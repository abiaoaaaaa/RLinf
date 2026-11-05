# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from toolkits.marvin.convert_online_buffer_joint_to_eef import convert_trajectory
from toolkits.marvin.label_eef_rl_demo_buffer import (
    EpisodeLabel,
    _chunk_starts,
    _dataset_root,
    _episode_sources,
    _episode_table,
    _transition_dict,
    _validate_rlt_buffer,
    _video_path,
)


class _FakeFk:
    def poses(self, joints, joint_velocities=None):
        del joint_velocities
        joints = np.asarray(joints)
        poses = np.zeros((len(joints), 14), dtype=np.float32)
        poses[:, :3] = joints[:, :3]
        poses[:, 3:7] = [0, 0, 0, 1]
        poses[:, 7:10] = joints[:, 7:10]
        poses[:, 10:14] = [0, 0, 0, 1]
        return poses, None


def test_episode_label_preserves_unfinished_interval():
    label = EpisodeLabel(intervals=[[10, 20]], open_start=30)

    encoded = label.to_dict()
    decoded = EpisodeLabel.from_dict(encoded)

    assert encoded["rl_intervals"] == [[10, 20]]
    assert decoded.open_start == 30
    assert _chunk_starts([10, 75], 30) == [10, 40, 70]
    assert _chunk_starts([0, 299], 30, 5) == list(range(0, 271, 5))
    assert _chunk_starts([0, 298], 30, 5)[-2:] == [265, 269]
    assert _chunk_starts([0, 20], 30, 5) == [0]


def test_demo_transition_has_success_only_at_terminal_step(tmp_path):
    curr = {
        "z_rl": torch.zeros(1, 8),
        "proprio": torch.zeros(1, 20),
        "ref_chunk": torch.zeros(1, 30, 20),
    }
    actions = np.zeros((30, 20), dtype=np.float32)

    transition = _transition_dict(
        torch,
        curr,
        curr,
        actions,
        terminal=True,
        success=True,
        model_weights_id="offline_eef_demo",
    )

    assert transition["actions"].shape == (1, 1, 600)
    assert transition["intervene_flags"].all()
    assert transition["rewards"].sum().item() == 1
    assert transition["rewards"][..., -1].item() == 1
    assert transition["dones"].sum().item() == 1
    assert transition["dones"][..., -1].item()

    torch.save(transition, tmp_path / "trajectory_0_offline_eef_demo.pt")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "trajectory_format": "pt",
                "size": 1,
                "total_samples": 1,
                "trajectory_counter": 1,
                "seed": 1234,
            }
        )
    )
    (tmp_path / "trajectory_index.json").write_text(
        json.dumps(
            {
                "trajectory_index": {
                    "0": {
                        "num_samples": 1,
                        "trajectory_id": 0,
                        "max_episode_length": 1,
                        "shape": [1, 1, 30],
                        "model_weights_id": "offline_eef_demo",
                    }
                },
                "trajectory_id_list": [0],
            }
        )
    )
    _validate_rlt_buffer(tmp_path, chunk_size=30, action_dim=20)


def test_lerobot_v3_episode_slices_and_video_paths(tmp_path):
    dataset = tmp_path / "dataset"
    data_path = dataset / "data/chunk-000/file-000.parquet"
    episode_path = dataset / "meta/episodes/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    episode_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "frame_index": [0, 1, 0, 1, 2],
                "episode_index": [0, 0, 1, 1, 1],
            }
        ),
        data_path,
    )
    episode_rows = {
        "episode_index": [0, 1],
        "data/chunk_index": [0, 0],
        "data/file_index": [0, 0],
        "dataset_from_index": [0, 2],
        "dataset_to_index": [2, 5],
    }
    for view in (
        "observation.images.top",
        "observation.images.wrist_L",
        "observation.images.wrist_R",
    ):
        episode_rows[f"videos/{view}/chunk_index"] = [0, 0]
        episode_rows[f"videos/{view}/file_index"] = [0, 1]
        video_dir = dataset / "videos" / view / "chunk-000"
        video_dir.mkdir(parents=True)
        (video_dir / "file-000.mp4").touch()
        (video_dir / "file-001.mp4").touch()
    pq.write_table(pa.table(episode_rows), episode_path)

    sources = _episode_sources(dataset / "videos")

    assert _dataset_root(dataset / "videos") == dataset
    assert len(_episode_table(sources[0])) == 2
    assert len(_episode_table(sources[1])) == 3
    assert _video_path(
        dataset, "observation.images.top", 1, sources[1]
    ).name == "file-001.mp4"


def test_joint_replay_conversion_changes_only_geometry_fields():
    source = {
        "actions": torch.zeros(1, 1, 32),
        "intervene_flags": torch.ones(1, 1, 32, dtype=torch.bool),
        "versions": torch.zeros(1, 1, 32),
        "rewards": torch.tensor([[[0.0, 1.0]]]),
        "dones": torch.tensor([[[False, True]]]),
        "curr_obs": {
            "z_rl": torch.arange(8).reshape(1, 1, 8).float(),
            "proprio": torch.zeros(1, 1, 16),
            "ref_chunk": torch.zeros(1, 1, 2, 16),
        },
        "next_obs": {
            "z_rl": torch.ones(1, 1, 8),
            "proprio": torch.zeros(1, 1, 16),
            "ref_chunk": torch.zeros(1, 1, 2, 16),
        },
        "forward_inputs": {
            "action": torch.zeros(1, 1, 32),
            "z_rl": torch.ones(1, 1, 8),
            "proprio": torch.zeros(1, 1, 16),
            "ref_chunk": torch.zeros(1, 1, 2, 16),
            "rlt_transition_proprio": torch.zeros(1, 1, 16),
            "rlt_transition_ref_chunk": torch.zeros(1, 1, 2, 16),
        },
    }

    converted = convert_trajectory(source, torch, _FakeFk())

    assert source["actions"].shape == (1, 1, 32)
    assert converted["actions"].shape == (1, 1, 40)
    assert converted["intervene_flags"].shape == (1, 1, 40)
    assert converted["curr_obs"]["proprio"].shape == (1, 1, 20)
    assert converted["curr_obs"]["ref_chunk"].shape == (1, 1, 2, 20)
    assert torch.equal(converted["curr_obs"]["z_rl"], source["curr_obs"]["z_rl"])
    assert torch.equal(converted["rewards"], source["rewards"])
    assert torch.equal(converted["dones"], source["dones"])
