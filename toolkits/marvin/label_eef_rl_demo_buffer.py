# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Interactively label Zadai demonstrations and materialize an RLT demo buffer.

This tool deliberately separates the inexpensive, resumable annotation pass
from GPU feature extraction::

    # A1: review videos and select at most 30 full expert episodes. Passing the
    #     ``videos`` subdirectory is accepted and resolved to the dataset root.
    python3 toolkits/marvin/label_eef_rl_demo_buffer.py annotate \
        --dataset /data/9_16_zadai_joint_clean/videos \
        --annotations /data/zadai_joint_expert_annotations.json \
        --max-expert-episodes 30

    # A2: run the matching Pi0.5/RLT Stage-1 model and write a
    #     TrajectoryReplayBuffer-compatible rank directory.
    python3 toolkits/marvin/label_eef_rl_demo_buffer.py export \
        --dataset /data/9_16_zadai_joint_clean \
        --annotations /data/zadai_joint_expert_annotations.json \
        --output /data/zadai_joint_demo_buffer/rank_0 \
        --stage1-checkpoint /checkpoints/zadai_277_stage1/actor \
        --norm-stats /checkpoints/zadai_277_stage1/norm_stats.json \
        --action-space joint --expected-episodes 30

Neither command edits the source dataset.  ``annotate`` atomically replaces
only its sidecar JSON.  ``export`` refuses an existing output directory.

Both LeRobot v2.1 (one parquet per episode) and v3 (episode slices in shared
parquet files) are supported. The source dataset is never modified.

Annotation keys
---------------
Space: play/pause; A/D: seek one frame; J/L: seek one second; ``[``: start an
RL interval; ``]``: end it; S: mark success at this frame; F: mark failure;
E: select the complete episode as a successful expert demonstration; N: save
and advance; B: save and go back; X: clear this episode; Q/Esc: save and quit.
Multiple ``[``/``]`` pairs are supported.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EEF_DIM = 20
JOINT_DIM = 16
DEFAULT_CHUNK_SIZE = 30
DEFAULT_PROMPT = "Insert the tail of the yellow cable tie into its head to fasten it."
VIEWS = (
    "observation.images.top",
    "observation.images.wrist_L",
    "observation.images.wrist_R",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class EpisodeSource:
    data_path: Path
    start: int
    stop: int
    video_locations: dict[str, tuple[int, int]]


def _dataset_root(dataset: Path) -> Path:
    dataset = dataset.resolve()
    return dataset.parent if dataset.name == "videos" else dataset


def _episode_sources(dataset: Path) -> dict[int, EpisodeSource]:
    import pyarrow.parquet as pq

    dataset = _dataset_root(dataset)
    files: dict[int, EpisodeSource] = {}
    for path in sorted((dataset / "data").rglob("episode_*.parquet")):
        try:
            episode = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        files[episode] = EpisodeSource(
            data_path=path,
            start=0,
            stop=pq.read_metadata(path).num_rows,
            video_locations={},
        )
    if files:
        return files

    episode_meta_files = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    if episode_meta_files:
        for row in pq.read_table(episode_meta_files).to_pylist():
            episode = int(row["episode_index"])
            data_path = dataset / "data" / (
                f"chunk-{int(row['data/chunk_index']):03d}/"
                f"file-{int(row['data/file_index']):03d}.parquet"
            )
            video_locations = {
                view: (
                    int(row[f"videos/{view}/chunk_index"]),
                    int(row[f"videos/{view}/file_index"]),
                )
                for view in VIEWS
            }
            files[episode] = EpisodeSource(
                data_path=data_path,
                start=int(row["dataset_from_index"]),
                stop=int(row["dataset_to_index"]),
                video_locations=video_locations,
            )
    if not files:
        raise FileNotFoundError(f"no episode metadata below {dataset}")
    return files


def _episode_table(source: EpisodeSource, columns: list[str] | None = None):
    import pyarrow.parquet as pq

    table = pq.read_table(source.data_path, columns=columns)
    return table.slice(source.start, source.stop - source.start)


def _video_path(
    dataset: Path, view: str, episode: int, source: EpisodeSource
) -> Path:
    dataset = _dataset_root(dataset)
    if view in source.video_locations:
        chunk, file_index = source.video_locations[view]
        path = (
            dataset
            / "videos"
            / view
            / f"chunk-{chunk:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = [
        path
        for path in (dataset / "videos").rglob(f"episode_{episode:06d}.mp4")
        if view in path.as_posix()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one {view} video for episode {episode}, got {candidates}"
        )
    return candidates[0]


def _load_annotations(path: Path, dataset: Path) -> dict[str, Any]:
    dataset = _dataset_root(dataset)
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if Path(value["dataset"]).resolve() != dataset.resolve():
            raise ValueError(
                f"annotation sidecar belongs to {value['dataset']}, not {dataset}"
            )
        return value
    return {
        "format": "zadai_rl_annotations_v2",
        "dataset": str(dataset.resolve()),
        "episodes": {},
    }


@dataclass
class EpisodeLabel:
    intervals: list[list[int]]
    open_start: int | None = None
    outcome: str = "unknown"
    terminal_frame: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "EpisodeLabel":
        value = value or {}
        return cls(
            intervals=[
                list(map(int, interval)) for interval in value.get("rl_intervals", [])
            ],
            open_start=value.get("open_rl_start"),
            outcome=str(value.get("outcome", "unknown")),
            terminal_frame=value.get("terminal_frame"),
        )

    def close_interval(self, frame: int) -> None:
        if self.open_start is None:
            raise ValueError("press '[' to start an RL interval first")
        if frame < self.open_start:
            raise ValueError("RL interval end precedes its start")
        self.intervals.append([self.open_start, frame])
        self.open_start = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rl_intervals": self.intervals,
            "open_rl_start": self.open_start,
            "outcome": self.outcome,
            "terminal_frame": self.terminal_frame,
        }


def _expert_episode_count(sidecar: dict[str, Any]) -> int:
    return sum(
        bool(value.get("rl_intervals")) and value.get("outcome") == "success"
        for value in sidecar["episodes"].values()
    )


def _is_expert_label(value: dict[str, Any]) -> bool:
    return bool(value.get("rl_intervals")) and value.get("outcome") == "success"


def annotate(
    dataset: Path,
    annotations: Path,
    start_episode: int | None,
    max_expert_episodes: int | None,
) -> None:
    if max_expert_episodes is not None and max_expert_episodes < 1:
        raise ValueError("max_expert_episodes must be >= 1")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("annotate requires opencv-python") from exc

    dataset = _dataset_root(dataset)
    episode_sources = _episode_sources(dataset)
    episode_ids = sorted(episode_sources)
    sidecar = _load_annotations(annotations, dataset)
    start_pos = 0
    if start_episode is not None:
        if start_episode not in episode_sources:
            raise ValueError(f"episode {start_episode} does not exist")
        start_pos = episode_ids.index(start_episode)
    else:
        for pos, episode in enumerate(episode_ids):
            if str(episode) not in sidecar["episodes"]:
                start_pos = pos
                break

    pos = start_pos
    while 0 <= pos < len(episode_ids):
        episode = episode_ids[pos]
        source = episode_sources[episode]
        frame_count = source.stop - source.start
        captures = [
            cv2.VideoCapture(str(_video_path(dataset, view, episode, source)))
            for view in VIEWS
        ]
        if any(not capture.isOpened() for capture in captures):
            raise RuntimeError(f"failed to open videos for episode {episode}")
        label = EpisodeLabel.from_dict(sidecar["episodes"].get(str(episode)))
        frame = 0
        playing = False
        direction = 1
        status = ""
        fps = max(1, int(round(captures[0].get(cv2.CAP_PROP_FPS) or 30)))
        try:
            while True:
                images = []
                for capture in captures:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, image = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f"cannot decode episode {episode}, frame {frame}"
                        )
                    images.append(cv2.resize(image, (426, 320)))
                canvas = np.concatenate(images, axis=1)
                in_rl = any(start <= frame <= end for start, end in label.intervals)
                if label.open_start is not None and frame >= label.open_start:
                    in_rl = True
                colour = (0, 220, 0) if in_rl else (220, 220, 220)
                lines = [
                    f"episode {episode} ({pos + 1}/{len(episode_ids)})  frame {frame}/{frame_count - 1}",
                    f"RL={in_rl} intervals={label.intervals} open={label.open_start}",
                    f"outcome={label.outcome} terminal={label.terminal_frame}  experts={_expert_episode_count(sidecar)}/{max_expert_episodes or '-'}  {status}",
                    "SPACE play | E full expert | [ start | ] end | S success | F fail | N next | Q quit",
                ]
                for row, line in enumerate(lines):
                    cv2.putText(
                        canvas,
                        line,
                        (12, 28 + row * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        colour,
                        2,
                    )
                cv2.imshow("Zadai RL annotation", canvas)
                key = cv2.waitKey(max(1, int(1000 / fps)) if playing else 0) & 0xFF
                status = ""
                if key == ord(" "):
                    playing = not playing
                elif key in (ord("a"), 81):
                    playing, frame = False, max(0, frame - 1)
                elif key in (ord("d"), 83):
                    playing, frame = False, min(frame_count - 1, frame + 1)
                elif key == ord("j"):
                    playing, frame = False, max(0, frame - fps)
                elif key == ord("l"):
                    playing, frame = False, min(frame_count - 1, frame + fps)
                elif key == ord("["):
                    label.open_start, status = frame, "RL interval started"
                elif key == ord("]"):
                    try:
                        label.close_interval(frame)
                    except ValueError as exc:
                        status = str(exc)
                elif key == ord("s"):
                    if label.open_start is not None:
                        label.close_interval(frame)
                    label.outcome, label.terminal_frame = "success", frame
                    status = "success terminal set"
                elif key == ord("f"):
                    if label.open_start is not None:
                        label.close_interval(frame)
                    label.outcome, label.terminal_frame = "failure", frame
                    status = "failure terminal set"
                elif key == ord("e"):
                    saved_label = sidecar["episodes"].get(str(episode), {})
                    selected_without_current = _expert_episode_count(sidecar) - int(
                        _is_expert_label(saved_label)
                    )
                    if (
                        max_expert_episodes is not None
                        and selected_without_current >= max_expert_episodes
                    ):
                        status = f"expert limit {max_expert_episodes} already reached"
                    else:
                        label = EpisodeLabel(
                            intervals=[[0, frame_count - 1]],
                            outcome="success",
                            terminal_frame=frame_count - 1,
                        )
                        status = "complete episode selected as expert"
                elif key == ord("x"):
                    label = EpisodeLabel([])
                    status = "episode label cleared"
                elif key in (ord("n"), ord("b"), ord("q"), 27):
                    candidate = label.to_dict()
                    old = sidecar["episodes"].get(str(episode), {})
                    candidate_count = (
                        _expert_episode_count(sidecar)
                        - int(_is_expert_label(old))
                        + int(_is_expert_label(candidate))
                    )
                    if (
                        max_expert_episodes is not None
                        and candidate_count > max_expert_episodes
                    ):
                        status = f"expert limit {max_expert_episodes} exceeded"
                        continue
                    sidecar["episodes"][str(episode)] = candidate
                    _atomic_json(annotations, sidecar)
                    if key in (ord("q"), 27):
                        return
                    direction = -1 if key == ord("b") else 1
                    break
                if playing:
                    frame += 1
                    if frame >= frame_count:
                        frame = frame_count - 1
                        playing = False
        finally:
            for capture in captures:
                capture.release()
        pos += direction
    cv2.destroyAllWindows()


def _matrix(table: Any, key: str, dim: int) -> np.ndarray:
    value = np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)
    if value.shape != (len(table), dim):
        raise ValueError(f"{key}: expected {(len(table), dim)}, got {value.shape}")
    return value


class EpisodeVideoReader:
    def __init__(self, dataset: Path, episode: int, source: EpisodeSource):
        import cv2

        self.cv2 = cv2
        self.captures = [
            cv2.VideoCapture(str(_video_path(dataset, view, episode, source)))
            for view in VIEWS
        ]
        if any(not capture.isOpened() for capture in self.captures):
            raise RuntimeError(f"failed to open videos for episode {episode}")

    def frame(self, index: int) -> list[np.ndarray]:
        result = []
        for capture in self.captures:
            capture.set(self.cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode frame {index}")
            result.append(self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB))
        return result

    def close(self) -> None:
        for capture in self.captures:
            capture.release()


def _load_feature_model(args: argparse.Namespace):
    try:
        import torch
        from omegaconf import OmegaConf

        from rlinf.models.embodiment.openpi import get_model
    except ImportError as exc:
        raise RuntimeError(
            "export requires the RLinf OpenPI environment (torch, omegaconf, openpi)"
        ) from exc
    joint = args.action_space == "joint"
    cfg = OmegaConf.create(
        {
            "model_type": "openpi",
            "precision": None,
            "is_lora": False,
            "model_path": str(args.stage1_checkpoint),
            "openpi_data": {
                "repo_id": "zadai_277_rlinf_v21" if joint else "zadai_200_eef_rot6d",
                "norm_stats_path": str(args.norm_stats),
            },
            "openpi": {
                "config_name": (
                    "pi05_zadai_ybs_joint_277" if joint else "pi05_zadai_eef_rot6d"
                ),
                "num_images_in_input": 3,
                "action_chunk": args.chunk_size,
                "action_horizon": args.ref_chunk_size,
                "action_env_dim": JOINT_DIM if joint else EEF_DIM,
                "num_steps": args.num_steps,
                "state_indices": [],
                "rlt_action_space": args.action_space,
                "use_rlt": True,
                "rlt_train_only": False,
                "rlt_prefix_seq_len": 1024,
                "rlt_image_only": False,
                "rlt_use_mask": True,
            },
        }
    )
    model = get_model(cfg).to(args.device).eval()
    model.requires_grad_(False)
    return model, torch


def _model_obs(
    model: Any, torch: Any, images: list[np.ndarray], state: np.ndarray, prompt: str
):
    env_obs = {
        "main_images": torch.from_numpy(images[0][None]),
        "wrist_images": None,
        "extra_view_images": torch.from_numpy(np.stack(images[1:], axis=0)[None]),
        "states": torch.from_numpy(state[None]),
        "task_descriptions": [prompt],
    }
    with torch.no_grad():
        return {
            key: value.detach().cpu().float()
            for key, value in model.extract_rlt_obs(env_obs).items()
        }


def _row(value: Any) -> Any:
    return value.unsqueeze(0) if getattr(value, "ndim", 0) >= 1 else value


def _transition_dict(
    torch: Any,
    curr: dict,
    nxt: dict,
    actions: np.ndarray,
    *,
    terminal: bool,
    success: bool,
    model_weights_id: str,
) -> dict:
    chunk = len(actions)
    action = torch.from_numpy(actions).float()
    rewards = torch.zeros((1, 1, chunk), dtype=torch.float32)
    dones = torch.zeros((1, 1, chunk), dtype=torch.bool)
    if terminal:
        dones[..., -1] = True
    if success:
        rewards[..., -1] = 1.0
    flat_action = action.reshape(1, 1, -1)
    flags = torch.ones_like(flat_action, dtype=torch.bool)
    curr_obs = {key: _row(value) for key, value in curr.items()}
    next_obs = {key: _row(value) for key, value in nxt.items()}
    forward_inputs = {
        "action": flat_action.clone(),
        **{key: value.clone() for key, value in curr_obs.items()},
        "record_transition": torch.ones((1, 1, 1), dtype=torch.bool),
        "actor_switch": torch.zeros((1, 1, 1), dtype=torch.bool),
        **{f"rlt_transition_{key}": value.clone() for key, value in curr_obs.items()},
    }
    return {
        "max_episode_length": 1,
        "model_weights_id": model_weights_id,
        "actions": flat_action,
        "intervene_flags": flags,
        "rewards": rewards,
        "terminations": dones.clone(),
        "truncations": torch.zeros_like(dones),
        "dones": dones,
        "versions": torch.zeros_like(flat_action),
        "forward_inputs": forward_inputs,
        "curr_obs": curr_obs,
        "next_obs": next_obs,
    }


def _chunk_starts(
    interval: list[int], chunk_size: int, chunk_stride: int | None = None
) -> list[int]:
    start, end = interval
    if start < 0 or end < start:
        raise ValueError(f"invalid RL interval {interval}")
    stride = chunk_size if chunk_stride is None else chunk_stride
    if chunk_size < 1 or stride < 1 or stride > chunk_size:
        raise ValueError("chunk size and stride must satisfy 1 <= stride <= size")
    if stride == chunk_size:
        return list(range(start, end + 1, chunk_size))
    last_start = max(start, end - chunk_size + 1)
    starts = list(range(start, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _validate_rlt_buffer(path: Path, chunk_size: int, action_dim: int) -> None:
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    buffer = TrajectoryReplayBuffer(
        enable_cache=True,
        cache_size=1,
        sample_window_size=1,
        auto_save=False,
    )
    try:
        buffer.load_checkpoint(str(path))
        sample = buffer.sample(1)
        if sample["actions"].shape[-1] != chunk_size * action_dim:
            raise ValueError(
                "exported action shape does not match the configured Joint/EEF chunk"
            )
    finally:
        buffer.close()


def export(args: argparse.Namespace) -> None:
    from rlinf.envs.realworld.zadai.action_codec import (
        DEFAULT_EEF_XYZ_HIGH,
        DEFAULT_EEF_XYZ_LOW,
        absolute_to_policy,
    )

    if args.expected_episodes is not None and args.expected_episodes < 1:
        raise ValueError("expected_episodes must be >= 1")
    if args.chunk_stride is not None and not 1 <= args.chunk_stride <= args.chunk_size:
        raise ValueError("chunk stride must satisfy 1 <= stride <= chunk size")
    dataset = _dataset_root(args.dataset)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output exists; refusing to modify it: {output}")
    if dataset == output or dataset in output.parents:
        raise ValueError("output must not be inside the source dataset")
    sidecar = _load_annotations(args.annotations, dataset)
    episode_sources = _episode_sources(dataset)
    selected = {
        int(key): value
        for key, value in sidecar["episodes"].items()
        if value.get("rl_intervals")
        and (value.get("outcome") == "success" or args.include_failures)
    }
    if not selected:
        raise ValueError("no exportable annotated RL intervals")
    if args.expected_episodes is not None and len(selected) != args.expected_episodes:
        raise ValueError(
            f"expected {args.expected_episodes} annotated episodes, got {len(selected)}"
        )
    model, torch = _load_feature_model(args)
    action_dim = JOINT_DIM if args.action_space == "joint" else EEF_DIM
    model_weights_id = f"offline_{args.action_space}_demo"
    staging = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    index: dict[str, dict[str, Any]] = {}
    transition_id = 0
    try:
        for episode, label in sorted(selected.items()):
            if episode not in episode_sources:
                raise ValueError(f"annotation references missing episode {episode}")
            source = episode_sources[episode]
            table = _episode_table(source)
            states = _matrix(table, "observation.state", action_dim)
            actions = _matrix(table, "action", action_dim)
            reader = EpisodeVideoReader(dataset, episode, source)
            obs_cache: dict[int, dict[str, Any]] = {}

            def model_obs_at(frame: int) -> dict[str, Any]:
                if frame not in obs_cache:
                    obs_cache[frame] = _model_obs(
                        model,
                        torch,
                        reader.frame(frame),
                        states[frame],
                        args.prompt,
                    )
                return obs_cache[frame]

            try:
                for interval in label["rl_intervals"]:
                    interval_end = min(int(interval[1]), len(table) - 1)
                    for start in _chunk_starts(
                        [int(interval[0]), interval_end],
                        args.chunk_size,
                        args.chunk_stride,
                    ):
                        end = min(start + args.chunk_size - 1, interval_end)
                        action_chunk = actions[start : end + 1]
                        if len(action_chunk) < args.chunk_size:
                            action_chunk = np.concatenate(
                                [
                                    action_chunk,
                                    np.repeat(
                                        action_chunk[-1:],
                                        args.chunk_size - len(action_chunk),
                                        axis=0,
                                    ),
                                ]
                            )
                        # Source demonstrations are physical absolute targets;
                        # Stage 2 replay stores the normalized RLT policy domain.
                        action_chunk = np.asarray(
                            absolute_to_policy(action_chunk, args.action_space),
                            dtype=np.float32,
                        )
                        next_frame = min(end + 1, len(table) - 1)
                        curr = model_obs_at(start)
                        nxt = model_obs_at(next_frame)
                        terminal = label.get("terminal_frame")
                        is_terminal = (
                            terminal is not None and start <= int(terminal) <= end
                        )
                        success = label.get("outcome") == "success" and is_terminal
                        trajectory = _transition_dict(
                            torch,
                            curr,
                            curr if is_terminal else nxt,
                            action_chunk,
                            terminal=is_terminal,
                            success=success,
                            model_weights_id=model_weights_id,
                        )
                        filename = (
                            f"trajectory_{transition_id}_{model_weights_id}.pt"
                        )
                        torch.save(trajectory, staging / filename)
                        index[str(transition_id)] = {
                            "num_samples": 1,
                            "trajectory_id": transition_id,
                            "max_episode_length": 1,
                            "shape": [1, 1, args.chunk_size],
                            "model_weights_id": model_weights_id,
                            "source_episode": episode,
                            "source_frame_start": start,
                            "source_frame_end": end,
                            "outcome": label.get("outcome", "unknown"),
                        }
                        transition_id += 1
            finally:
                reader.close()
        _atomic_json(
            staging / "metadata.json",
            {
                "trajectory_format": "pt",
                "size": transition_id,
                "total_samples": transition_id,
                "trajectory_counter": transition_id,
                "seed": 1234,
            },
        )
        _atomic_json(
            staging / "trajectory_index.json",
            {
                "trajectory_index": index,
                "trajectory_id_list": list(range(transition_id)),
            },
        )
        shutil.copy2(args.annotations, staging / "source_annotations.json")
        _atomic_json(
            staging / "demo_buffer_manifest.json",
            {
                "format": "rlinf_rlt_demo_buffer_v2",
                "source_dataset": str(dataset),
                "source_annotations": str(args.annotations.resolve()),
                "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
                "norm_stats": str(args.norm_stats.resolve()),
                "action_space": args.action_space,
                "action_dim": action_dim,
                "source_action_representation": f"absolute_{args.action_space}",
                "replay_action_representation": f"normalized_{args.action_space}",
                "eef_xyz_low": (
                    DEFAULT_EEF_XYZ_LOW.tolist()
                    if args.action_space == "eef"
                    else None
                ),
                "eef_xyz_high": (
                    DEFAULT_EEF_XYZ_HIGH.tolist()
                    if args.action_space == "eef"
                    else None
                ),
                "chunk_size": args.chunk_size,
                "chunk_stride": args.chunk_stride or args.chunk_size,
                "source_episode_count": len(selected),
                "z_rl_compatibility": "must use the same Stage-1 checkpoint family",
            },
        )
        _validate_rlt_buffer(staging, args.chunk_size, action_dim)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"wrote {transition_id} {args.action_space} RLT transitions "
        f"from {len(selected)} episodes to {output}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    label_parser = subparsers.add_parser("annotate", help="keyboard video annotation")
    label_parser.add_argument("--dataset", type=Path, required=True)
    label_parser.add_argument("--annotations", type=Path, required=True)
    label_parser.add_argument("--start-episode", type=int)
    label_parser.add_argument("--max-expert-episodes", type=int)

    export_parser = subparsers.add_parser(
        "export", help="materialize an RLT demo buffer"
    )
    export_parser.add_argument("--dataset", type=Path, required=True)
    export_parser.add_argument("--annotations", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    export_parser.add_argument("--norm-stats", type=Path, required=True)
    export_parser.add_argument(
        "--action-space", choices=("joint", "eef"), default="eef"
    )
    export_parser.add_argument("--expected-episodes", type=int)
    export_parser.add_argument("--device", default="cuda")
    export_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    export_parser.add_argument(
        "--chunk-stride",
        type=int,
        help="frame stride between overlapping chunks (default: chunk-size)",
    )
    export_parser.add_argument("--ref-chunk-size", type=int, default=50)
    export_parser.add_argument("--num-steps", type=int, default=10)
    export_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    export_parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "annotate":
            annotate(
                args.dataset,
                args.annotations.resolve(),
                args.start_episode,
                args.max_expert_episodes,
            )
        else:
            export(args)
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
