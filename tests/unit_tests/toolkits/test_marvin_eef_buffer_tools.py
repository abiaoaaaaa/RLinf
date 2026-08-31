# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import torch

from toolkits.marvin.convert_online_buffer_joint_to_eef import convert_trajectory
from toolkits.marvin.label_eef_rl_demo_buffer import (
    EpisodeLabel,
    _chunk_starts,
    _transition_dict,
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


def test_demo_transition_has_success_only_at_terminal_step():
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
    )

    assert transition["actions"].shape == (1, 1, 600)
    assert transition["intervene_flags"].all()
    assert transition["rewards"].sum().item() == 1
    assert transition["rewards"][..., -1].item() == 1
    assert transition["dones"].sum().item() == 1
    assert transition["dones"][..., -1].item()


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
