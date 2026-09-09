# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert the Zadai LeRobot v3 bundle to RLinf's pinned v2.1 layout.

The conversion is structural only: Parquet rows are split by episode and MP4
files are linked or copied without decoding/re-encoding.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROMPT = "Insert the tail of the yellow cable tie into its head to fasten it."
CAMERA_KEYS = (
    "observation.images.top",
    "observation.images.wrist_L",
    "observation.images.wrist_R",
)
JOINT_ACTION_NAMES = [
    *(f"Joint{i}_L" for i in range(1, 8)),
    *(f"Joint{i}_R" for i in range(1, 8)),
    "gripper_L",
    "gripper_R",
]


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def _feature_names(info: dict, key: str) -> list[str]:
    names = info["features"][key].get("names")
    return list(names.get("axes", [])) if isinstance(names, dict) else list(names or [])


def _validate_source(source: Path) -> tuple[dict, pa.Table, dict, dict]:
    meta = source / "meta"
    info = _read_json(meta / "info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"expected LeRobot v3.0, got {info.get('codebase_version')!r}")
    total_episodes = info.get("total_episodes")
    if info.get("fps") != 30:
        raise ValueError(f"expected a 30 FPS Zadai dataset, got {info.get('fps')!r}")
    if not isinstance(total_episodes, int) or total_episodes <= 0:
        raise ValueError(
            f"expected a positive integer total_episodes, got {total_episodes!r}"
        )
    state_shape = info["features"]["observation.state"]["shape"]
    action_shape = info["features"]["action"]["shape"]
    if state_shape != action_shape or len(state_shape) != 1:
        raise ValueError(
            "observation.state and action must have the same one-dimensional shape"
        )
    state_names = _feature_names(info, "observation.state")
    action_names = _feature_names(info, "action")
    if state_names != action_names or len(state_names) != state_shape[0]:
        raise ValueError("observation.state and action names must match their shape")
    if state_shape == [16] and state_names != JOINT_ACTION_NAMES:
        raise ValueError("unexpected 16-D Zadai joint order")
    if tuple(key for key in info["features"] if key in CAMERA_KEYS) != CAMERA_KEYS:
        raise ValueError("expected top, wrist_L and wrist_R video features")

    task_table = pq.read_table(meta / "tasks.parquet")
    tasks = task_table.to_pylist()
    expected_tasks = [
        {"task_index": task_index, "task": PROMPT} for task_index in range(len(tasks))
    ]
    if not tasks or tasks != expected_tasks:
        raise ValueError(f"unexpected task metadata: {tasks!r}")

    episode_table = pq.read_table(meta / "episodes/chunk-000/file-000.parquet")
    episodes = episode_table.to_pydict()
    if episodes["episode_index"] != list(range(total_episodes)):
        raise ValueError(
            "episode indices must be contiguous from 0 to "
            f"{total_episodes - 1}"
        )

    data_files = sorted((source / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError("no source data Parquet files found")
    data = pa.concat_tables([pq.read_table(path) for path in data_files])
    if data.num_rows != info["total_frames"]:
        raise ValueError(
            f"frame count mismatch: {data.num_rows} != {info['total_frames']}"
        )
    return info, data, episodes, {"tasks": tasks, "episode_table": episode_table}


def _episode_stats(episode_table: pa.Table) -> list[dict]:
    rows = episode_table.to_pylist()
    result = []
    for row in rows:
        stats: dict[str, dict[str, object]] = {}
        for key, value in row.items():
            if not key.startswith("stats/"):
                continue
            feature, stat = key.removeprefix("stats/").rsplit("/", 1)
            if feature == "task_index" and stat == "count":
                value = [row["length"]]
            stats.setdefault(feature, {})[stat] = value
        result.append({"episode_index": row["episode_index"], "stats": stats})
    return result


def _materialize_video(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def convert(source: Path, output: Path, video_mode: str) -> None:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if output == source or source in output.parents:
        raise ValueError("output must not be the source or a child of the source")

    info, data, episodes, extra = _validate_source(source)
    chunks_size = int(info.get("chunks_size", 1000))
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (temp / "meta").mkdir()
        lengths = [int(value) for value in episodes["length"]]
        starts = [int(value) for value in episodes["dataset_from_index"]]
        ends = [int(value) for value in episodes["dataset_to_index"]]
        for episode_index, (start, end, length) in enumerate(
            zip(starts, ends, lengths, strict=True)
        ):
            if end - start != length:
                raise ValueError(f"episode {episode_index}: invalid frame bounds")
            episode = data.slice(start, length)
            if set(episode["episode_index"].to_pylist()) != {episode_index}:
                raise ValueError(
                    f"episode {episode_index}: Parquet frame indices do not match metadata"
                )
            # LeRobot v3 Parquet files may embed Hugging Face feature metadata
            # containing the newer ``List`` feature type. RLinf's pinned
            # datasets version cannot deserialize it, but can infer the same
            # Arrow columns correctly when this optional metadata is absent.
            episode = episode.replace_schema_metadata(None)
            chunk = episode_index // chunks_size
            path = temp / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode, path, compression="zstd")

            for camera in CAMERA_KEYS:
                source_video = (
                    source / f"videos/{camera}/chunk-000/file-{episode_index:03d}.mp4"
                )
                if not source_video.is_file():
                    raise FileNotFoundError(source_video)
                target_video = (
                    temp
                    / f"videos/chunk-{chunk:03d}/{camera}/episode_{episode_index:06d}.mp4"
                )
                _materialize_video(source_video, target_video, video_mode)

        target_info = dict(info)
        target_info.update(
            {
                "codebase_version": "v2.1",
                "total_videos": len(CAMERA_KEYS) * info["total_episodes"],
                "total_chunks": (info["total_episodes"] + chunks_size - 1)
                // chunks_size,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }
        )
        _write_json(temp / "meta/info.json", target_info)
        shutil.copy2(source / "meta/stats.json", temp / "meta/stats.json")
        _write_jsonl(temp / "meta/tasks.jsonl", extra["tasks"])
        _write_jsonl(
            temp / "meta/episodes.jsonl",
            [
                {"episode_index": index, "tasks": tasks, "length": length}
                for index, (tasks, length) in enumerate(
                    zip(episodes["tasks"], lengths, strict=True)
                )
            ],
        )
        _write_jsonl(
            temp / "meta/episodes_stats.jsonl", _episode_stats(extra["episode_table"])
        )

        if (
            sum(
                pq.read_metadata(path).num_rows
                for path in (temp / "data").rglob("*.parquet")
            )
            != info["total_frames"]
        ):
            raise RuntimeError("converted Parquet frame count verification failed")
        first_output_data = next((temp / "data").rglob("*.parquet"))
        if pq.read_schema(first_output_data).metadata:
            raise RuntimeError("converted Parquet schema metadata was not stripped")
        if len(list((temp / "videos").rglob("*.mp4"))) != target_info["total_videos"]:
            raise RuntimeError("converted video count verification failed")
        temp.rename(output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="source Zadai LeRobot v3 dataset root"
    )
    parser.add_argument(
        "output", type=Path, help="new RLinf-compatible LeRobot v2.1 dataset root"
    )
    parser.add_argument(
        "--video-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="reuse MP4 files without re-encoding (default: hardlink, copy fallback)",
    )
    args = parser.parse_args()
    convert(args.source, args.output, args.video_mode)
    print(f"Converted dataset: {args.output.resolve()}")


if __name__ == "__main__":
    main()
