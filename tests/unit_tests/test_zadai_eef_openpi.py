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
"""Tests for the Zadai EEF OpenPI data path."""

from __future__ import annotations

import contextlib

import numpy as np
import torch
from omegaconf import OmegaConf
from openpi.models.model import ModelType
from scipy.spatial.transform import Rotation

from rlinf.models.embodiment.openpi.policies.zadai_eef_rot6d_policy import (
    ZadaiEefRot6dInputs,
)
from rlinf.models.embodiment.openpi.transforms import (
    DUAL_ARM_ROT6D_LAYOUT,
    RigidBodyDeltaActions,
)
from rlinf.utils.rot6d import matrix_to_rot6d
from toolkits.marvin.calculate_zadai_eef_norm_stats import (
    body_delta_action_chunks,
)


def _state(
    left_xyz: list[float], left_rotation: np.ndarray | None = None
) -> np.ndarray:
    identity = np.eye(3) if left_rotation is None else left_rotation
    value = np.zeros(20, dtype=np.float32)
    value[0:3] = left_xyz
    value[3:9] = matrix_to_rot6d(identity)
    value[10:13] = [0.2, -0.1, 0.5]
    value[13:19] = matrix_to_rot6d(np.eye(3))
    return value


def test_policy_maps_three_images_and_pads_to_model_dim() -> None:
    transform = ZadaiEefRot6dInputs(
        action_dim=32,
        model_type=ModelType.PI0,
        default_prompt="task",
    )
    image = np.zeros((3, 8, 10), dtype=np.uint8)
    action = np.stack([_state([0.1, 0.2, 0.3])] * 4)

    result = transform(
        {
            "observation/state": _state([0.1, 0.2, 0.3]),
            "observation/top": image,
            "observation/wrist_L": image,
            "observation/wrist_R": image,
            "actions": action,
        }
    )

    assert result["state"].shape == (32,)
    assert result["actions"].shape == (4, 32)
    assert set(result["image"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    assert result["image"]["base_0_rgb"].shape == (8, 10, 3)
    assert result["prompt"] == "task"


def test_fast_norm_delta_matches_training_transform() -> None:
    states = np.stack(
        [
            _state([0.4, 0.1, 0.2]),
            _state(
                [0.0, 0.0, 0.0], Rotation.from_euler("z", 90, degrees=True).as_matrix()
            ),
        ]
    )
    actions = np.stack(
        [
            np.stack([_state([0.5, 0.2, 0.3]), _state([0.6, 0.1, 0.2])]),
            np.stack([_state([1.0, 0.0, 0.0]), _state([0.0, 1.0, 0.0])]),
        ]
    )

    padded_states, fast_delta = body_delta_action_chunks(states, actions, 32)

    transform = RigidBodyDeltaActions(DUAL_ARM_ROT6D_LAYOUT)
    for batch_index in range(len(states)):
        expected = transform(
            {
                "state": padded_states[batch_index],
                "actions": np.pad(actions[batch_index], ((0, 0), (0, 12))),
            }
        )["actions"]
        np.testing.assert_allclose(fast_delta[batch_index], expected, atol=1e-6)
    np.testing.assert_allclose(fast_delta[1, 0, 0:3], [0.0, -1.0, 0.0], atol=1e-6)


def test_sft_worker_forwards_real_action_loss_flag() -> None:
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    class FakeModel:
        kwargs: dict | None = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {"loss": torch.tensor(2.0), "rlt_loss": torch.tensor(1.0)}

    worker = object.__new__(FSDPVlaSftWorker)
    worker.cfg = OmegaConf.create(
        {"actor": {"model": {"openpi": {"sft_use_action_chunk_loss": True}}}}
    )
    worker.amp_context = contextlib.nullcontext()
    worker.model = FakeModel()

    loss, metrics = worker.get_train_model_output(batch="batch")

    assert loss.item() == 2.0
    assert metrics == {"loss": 2.0, "rlt_loss": 1.0}
    assert worker.model.kwargs["data"] == "batch"
    assert worker.model.kwargs["use_action_chunk_loss"] is True
