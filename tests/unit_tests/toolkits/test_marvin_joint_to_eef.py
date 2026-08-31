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
"""Tests for the Marvin joint-to-EEF dataset conversion core."""

from __future__ import annotations

import numpy as np

from toolkits.marvin.convert_joint_lerobot_to_eef import (
    array_stats,
    convert_arrays,
    eef_quat_to_rot6d,
)


class _FakeFK:
    """Deterministic FK that makes the source of each target observable."""

    def poses(
        self, joints: np.ndarray, joint_velocities: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        poses = np.zeros((len(joints), 14), dtype=np.float32)
        poses[:, 0:3] = joints[:, 0:3]
        poses[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
        poses[:, 7:10] = joints[:, 7:10]
        poses[:, 10:14] = [0.0, 0.0, 0.0, 1.0]
        twists = None
        if joint_velocities is not None:
            twists = np.concatenate(
                [
                    joint_velocities[:, 0:6],
                    joint_velocities[:, 7:13],
                ],
                axis=1,
            ).astype(np.float32)
        return poses, twists


def test_eef_quat_to_rot6d_canonical_layout() -> None:
    poses = np.zeros((1, 14), dtype=np.float32)
    poses[0, :7] = [1, 2, 3, 0, 0, 0, 1]
    poses[0, 7:] = [4, 5, 6, 0, 0, 0, 1]

    result = eef_quat_to_rot6d(poses, np.array([[0.25, 0.75]], dtype=np.float32))

    identity_rot6d = [1, 0, 0, 0, 1, 0]
    expected = np.array(
        [[1, 2, 3, *identity_rot6d, 0.25, 4, 5, 6, *identity_rot6d, 0.75]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result, expected)


def test_convert_arrays_runs_fk_on_source_action() -> None:
    state = np.zeros((2, 16), dtype=np.float32)
    action = np.zeros((2, 16), dtype=np.float32)
    velocity = np.arange(32, dtype=np.float32).reshape(2, 16)
    state[:, 0:3] = [[1, 2, 3], [10, 20, 30]]
    state[:, 7:10] = [[4, 5, 6], [40, 50, 60]]
    action[:, 0:3] = [[7, 8, 9], [70, 80, 90]]
    action[:, 7:10] = [[11, 12, 13], [110, 120, 130]]
    state[:, 14:16] = [[0.1, 0.2], [0.3, 0.4]]
    action[:, 14:16] = [[0.5, 0.6], [0.7, 0.8]]

    converted = convert_arrays(state, action, velocity, _FakeFK())

    np.testing.assert_allclose(converted["observation.state"][:, 0:3], state[:, 0:3])
    np.testing.assert_allclose(converted["action"][:, 0:3], action[:, 0:3])
    np.testing.assert_allclose(converted["action"][:, 10:13], action[:, 7:10])
    np.testing.assert_allclose(converted["action"][:, [9, 19]], action[:, 14:16])
    np.testing.assert_allclose(
        converted["action.joint_position_source"], action[:, :14]
    )
    assert np.all(converted["annotation.is_rl_phase"] == -1)
    assert np.all(converted["annotation.outcome"] == -1)


def test_array_stats_retains_vector_shape_and_count() -> None:
    values = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)

    stats = array_stats(values)

    assert stats["count"] == [3]
    assert stats["min"] == [1.0, 4.0]
    assert stats["max"] == [3.0, 6.0]
    assert stats["mean"] == [2.0, 5.0]
