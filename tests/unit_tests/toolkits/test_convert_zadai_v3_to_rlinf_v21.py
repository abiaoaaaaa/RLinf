# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the Zadai LeRobot v3 to v2.1 structural converter."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from toolkits.lerobot.convert_zadai_v3_to_rlinf_v21 import (
    CAMERA_KEYS,
    JOINT_ACTION_NAMES,
    PROMPT,
    convert,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_source(root: Path, total_episodes: int = 2) -> None:
    features = {
        "observation.state": {"shape": [16], "names": JOINT_ACTION_NAMES},
        "action": {"shape": [16], "names": JOINT_ACTION_NAMES},
        **{key: {"shape": [1, 1, 3]} for key in CAMERA_KEYS},
    }
    _write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "fps": 30,
            "total_episodes": total_episodes,
            "total_frames": total_episodes,
            "chunks_size": 1000,
            "features": features,
        },
    )
    _write_json(root / "meta/stats.json", {})
    tasks_path = root / "meta/tasks.parquet"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"task_index": [0], "task": [PROMPT]}), tasks_path
    )
    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(total_episodes)),
                "length": [1] * total_episodes,
                "dataset_from_index": list(range(total_episodes)),
                "dataset_to_index": list(range(1, total_episodes + 1)),
                "tasks": [[PROMPT]] * total_episodes,
            }
        ),
        episode_path,
    )
    data_path = root / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data = pa.table({"episode_index": list(range(total_episodes))})
    data = data.replace_schema_metadata(
        {b"huggingface": b'{"info":{"features":{"x":{"_type":"List"}}}}'}
    )
    pq.write_table(data, data_path)
    for camera in CAMERA_KEYS:
        camera_dir = root / "videos" / camera / "chunk-000"
        camera_dir.mkdir(parents=True, exist_ok=True)
        for episode_index in range(total_episodes):
            (camera_dir / f"file-{episode_index:03d}.mp4").write_bytes(b"video")


def test_convert_accepts_non_200_episode_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_source(source)

    convert(source, output, "hardlink")

    info = json.loads((output / "meta/info.json").read_text())
    assert info["total_episodes"] == 2
    assert len(list((output / "data").rglob("*.parquet"))) == 2
    assert len(list((output / "videos").rglob("*.mp4"))) == 6
    first_parquet = output / "data/chunk-000/episode_000000.parquet"
    assert pq.read_schema(first_parquet).metadata is None
    assert (output / "videos/chunk-000/observation.images.top/episode_000001.mp4").stat().st_ino == (
        source / "videos/observation.images.top/chunk-000/file-001.mp4"
    ).stat().st_ino
