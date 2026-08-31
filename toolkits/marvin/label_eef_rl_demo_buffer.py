# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Interactively label EEF demonstrations and materialize an RLT demo buffer.

This tool deliberately separates the inexpensive, resumable annotation pass
from GPU feature extraction::

    # A1: review videos, mark one or more critical (RL) intervals, and label
    #     the outcome at the selected terminal frame.
    python3 toolkits/marvin/label_eef_rl_demo_buffer.py annotate \
        --dataset /data/zadai_200_eef_rot6d_rlinf_v21 \
        --annotations /data/zadai_200_eef_annotations.json

    # A2: run the jointly trained EEF Pi0.5/RLT Stage-1 model and write a
    #     TrajectoryReplayBuffer-compatible rank directory.
    python3 toolkits/marvin/label_eef_rl_demo_buffer.py export \
        --dataset /data/zadai_200_eef_rot6d_rlinf_v21 \
        --annotations /data/zadai_200_eef_annotations.json \
        --output /data/zadai_200_eef_demo_buffer/rank_0 \
        --stage1-checkpoint /checkpoints/eef_stage1/actor \
        --norm-stats /data/zadai_200_eef_norm_stats/norm_stats.json

Neither command edits the source dataset.  ``annotate`` atomically replaces
only its sidecar JSON.  ``export`` refuses an existing output directory.

Annotation keys
---------------
Space: play/pause; A/D: seek one frame; J/L: seek one second; ``[``: start an
RL interval; ``]``: end it; S: mark success at this frame; F: mark failure;
N: save and advance; B: save and go back; X: clear this episode; Q/Esc: save
and quit.  Multiple ``[``/``]`` pairs are supported.
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


def _episode_files(dataset: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted((dataset / "data").rglob("episode_*.parquet")):
        try:
            episode = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        files[episode] = path
    if not files:
        raise FileNotFoundError(f"no episode parquet files below {dataset / 'data'}")
    return files


def _video_path(dataset: Path, view: str, episode: int) -> Path:
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
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if Path(value["dataset"]).resolve() != dataset.resolve():
            raise ValueError(
                f"annotation sidecar belongs to {value['dataset']}, not {dataset}"
            )
        return value
    return {
        "format": "zadai_eef_rl_annotations_v1",
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


def annotate(dataset: Path, annotations: Path, start_episode: int | None) -> None:
    try:
        import cv2
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("annotate requires opencv-python and pyarrow") from exc

    episode_files = _episode_files(dataset)
    episode_ids = sorted(episode_files)
    sidecar = _load_annotations(annotations, dataset)
    start_pos = 0
    if start_episode is not None:
        if start_episode not in episode_files:
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
        table = pq.read_table(episode_files[episode], columns=["frame_index"])
        frame_count = len(table)
        captures = [
            cv2.VideoCapture(str(_video_path(dataset, view, episode))) for view in VIEWS
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
                    f"outcome={label.outcome} terminal={label.terminal_frame}  {status}",
                    "SPACE play | [ start | ] end | S success | F fail | N next | Q quit",
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
                cv2.imshow("Zadai EEF RL annotation", canvas)
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
                elif key == ord("x"):
                    label = EpisodeLabel([])
                    status = "episode label cleared"
                elif key in (ord("n"), ord("b"), ord("q"), 27):
                    sidecar["episodes"][str(episode)] = label.to_dict()
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
    def __init__(self, dataset: Path, episode: int):
        import cv2

        self.cv2 = cv2
        self.captures = [
            cv2.VideoCapture(str(_video_path(dataset, view, episode))) for view in VIEWS
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
    cfg = OmegaConf.create(
        {
            "model_type": "openpi",
            "precision": None,
            "is_lora": False,
            "model_path": str(args.stage1_checkpoint),
            "openpi_data": {
                "repo_id": "zadai_200_eef_rot6d",
                "norm_stats_path": str(args.norm_stats),
            },
            "openpi": {
                "config_name": "pi05_zadai_eef_rot6d",
                "num_images_in_input": 3,
                "action_chunk": args.chunk_size,
                "action_horizon": args.ref_chunk_size,
                "action_env_dim": EEF_DIM,
                "num_steps": args.num_steps,
                "state_indices": [],
                "rlt_action_space": "eef",
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
        "model_weights_id": "offline_eef_demo",
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


def _chunk_starts(interval: list[int], chunk_size: int) -> list[int]:
    start, end = interval
    if start < 0 or end < start:
        raise ValueError(f"invalid RL interval {interval}")
    return list(range(start, end + 1, chunk_size))


def export(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    from rlinf.envs.realworld.zadai.action_codec import eef_absolute_to_normalized

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output exists; refusing to modify it: {output}")
    if dataset == output or dataset in output.parents:
        raise ValueError("output must not be inside the source dataset")
    sidecar = _load_annotations(args.annotations, dataset)
    episode_files = _episode_files(dataset)
    selected = {
        int(key): value
        for key, value in sidecar["episodes"].items()
        if value.get("rl_intervals")
        and (value.get("outcome") == "success" or args.include_failures)
    }
    if not selected:
        raise ValueError("no exportable annotated RL intervals")
    model, torch = _load_feature_model(args)
    staging = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    index: dict[str, dict[str, Any]] = {}
    transition_id = 0
    try:
        for episode, label in sorted(selected.items()):
            if episode not in episode_files:
                raise ValueError(f"annotation references missing episode {episode}")
            table = pq.read_table(episode_files[episode])
            states = _matrix(table, "observation.state", EEF_DIM)
            actions = _matrix(table, "action", EEF_DIM)
            reader = EpisodeVideoReader(dataset, episode)
            try:
                for interval in label["rl_intervals"]:
                    interval_end = min(int(interval[1]), len(table) - 1)
                    for start in _chunk_starts(
                        [int(interval[0]), interval_end], args.chunk_size
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
                        # Stage 1 and the source dataset stay in physical
                        # absolute EEF space. Stage 2 replay stores exactly the
                        # normalized domain produced/consumed by the RLT MLP.
                        action_chunk = np.asarray(
                            eef_absolute_to_normalized(action_chunk),
                            dtype=np.float32,
                        )
                        next_frame = min(end + 1, len(table) - 1)
                        curr = _model_obs(
                            model,
                            torch,
                            reader.frame(start),
                            states[start],
                            args.prompt,
                        )
                        nxt = _model_obs(
                            model,
                            torch,
                            reader.frame(next_frame),
                            states[next_frame],
                            args.prompt,
                        )
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
                        )
                        filename = f"trajectory_{transition_id}_offline_eef_demo.pt"
                        torch.save(trajectory, staging / filename)
                        index[str(transition_id)] = {
                            "num_samples": 1,
                            "trajectory_id": transition_id,
                            "max_episode_length": 1,
                            "shape": [1, 1, args.chunk_size],
                            "model_weights_id": "offline_eef_demo",
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
            staging / "eef_buffer_manifest.json",
            {
                "format": "rlinf_rlt_eef_demo_buffer_v1",
                "source_dataset": str(dataset),
                "source_annotations": str(args.annotations.resolve()),
                "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
                "norm_stats": str(args.norm_stats.resolve()),
                "source_action_representation": "absolute_bilateral_eef_xyz_rot6d_gripper_20d",
                "replay_action_representation": "normalized_bilateral_eef_xyz_rot6d_gripper_20d",
                "chunk_size": args.chunk_size,
                "z_rl_compatibility": "must use the same Stage-1 checkpoint family",
            },
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"wrote {transition_id} EEF RLT transitions to {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    label_parser = subparsers.add_parser("annotate", help="keyboard video annotation")
    label_parser.add_argument("--dataset", type=Path, required=True)
    label_parser.add_argument("--annotations", type=Path, required=True)
    label_parser.add_argument("--start-episode", type=int)

    export_parser = subparsers.add_parser(
        "export", help="materialize an RLT demo buffer"
    )
    export_parser.add_argument("--dataset", type=Path, required=True)
    export_parser.add_argument("--annotations", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    export_parser.add_argument("--norm-stats", type=Path, required=True)
    export_parser.add_argument("--device", default="cuda")
    export_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    export_parser.add_argument("--ref-chunk-size", type=int, default=50)
    export_parser.add_argument("--num-steps", type=int, default=10)
    export_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    export_parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "annotate":
            annotate(
                args.dataset.resolve(), args.annotations.resolve(), args.start_episode
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
