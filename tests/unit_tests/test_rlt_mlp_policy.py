# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def _obs():
    return {
        "z_rl": torch.randn(2, 2),
        "proprio": torch.randn(2, 3),
        "ref_chunk": torch.tensor(
            [
                [[0.1, -0.2], [0.3, -0.4], [0.5, -0.6], [0.7, -0.8]],
                [[-0.1, 0.2], [-0.3, 0.4], [-0.5, 0.6], [-0.7, 0.8]],
            ]
        ),
    }


def test_residual_actor_starts_at_reference_and_respects_per_joint_bound():
    policy = RLTMLPPolicy(
        z_dim=2,
        proprio_dim=3,
        action_dim=2,
        num_action_chunks=3,
        ref_num_action_chunks=4,
        max_residual=[0.03, 0.0],
        residual_action=True,
    )
    obs = _obs()
    reference = obs["ref_chunk"][:, :3].reshape(2, -1)

    initial, _, _ = policy.sac_forward(obs, deterministic=True)
    torch.testing.assert_close(initial, reference)

    with torch.no_grad():
        policy.actor_mean.bias.fill_(20.0)
    shifted, _, _ = policy.sac_forward(obs, deterministic=True)
    delta = shifted.reshape(2, 3, 2) - reference.reshape(2, 3, 2)
    assert torch.all(torch.abs(delta[..., 0]) <= 0.030001)
    torch.testing.assert_close(delta[..., 1], torch.zeros_like(delta[..., 1]))


def test_residual_actor_requires_one_bound_per_action_dimension():
    with pytest.raises(ValueError, match="one value per action dimension"):
        RLTMLPPolicy(
            z_dim=2,
            proprio_dim=3,
            action_dim=2,
            num_action_chunks=3,
            residual_action=True,
            max_residual=[0.03, 0.03, 0.03],
        )
