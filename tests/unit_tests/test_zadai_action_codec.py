# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import numpy as np
import torch

from rlinf.envs.realworld.zadai.action_codec import (
    ACTION_HIGH,
    ACTION_LOW,
    DEFAULT_EEF_XYZ_HIGH,
    DEFAULT_EEF_XYZ_LOW,
    absolute_to_normalized,
    clamp_rot6d_angle,
    eef_absolute_to_normalized,
    eef_normalized_to_absolute,
    normalized_to_absolute,
    quaternion_xyzw_to_rot6d,
    rot6d_angular_distance,
    rot6d_to_quaternion_xyzw,
)


def test_zadai_action_codec_round_trip():
    physical = np.stack([ACTION_LOW, ACTION_HIGH, (ACTION_LOW + ACTION_HIGH) / 2.0])
    normalized = absolute_to_normalized(physical)
    np.testing.assert_allclose(normalized[0], -1.0)
    np.testing.assert_allclose(normalized[1], 1.0)
    np.testing.assert_allclose(normalized_to_absolute(normalized), physical)

    tensor = torch.as_tensor(physical)
    assert torch.allclose(
        normalized_to_absolute(absolute_to_normalized(tensor)), tensor
    )


def test_zadai_eef_codec_round_trip_preserves_valid_pose():
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    physical = np.concatenate(
        [
            (DEFAULT_EEF_XYZ_LOW[0] + DEFAULT_EEF_XYZ_HIGH[0]) / 2,
            identity,
            [0.25],
            (DEFAULT_EEF_XYZ_LOW[1] + DEFAULT_EEF_XYZ_HIGH[1]) / 2,
            identity,
            [0.75],
        ]
    ).astype(np.float32)

    normalized = eef_absolute_to_normalized(physical)
    decoded = eef_normalized_to_absolute(normalized)

    assert normalized.shape == (20,)
    np.testing.assert_allclose(decoded, physical, atol=1e-6)


def test_zadai_rot6d_quaternion_round_trip():
    quaternion = np.asarray([0.2, -0.3, 0.1, 0.9], dtype=np.float32)
    quaternion /= np.linalg.norm(quaternion)
    round_trip = rot6d_to_quaternion_xyzw(quaternion_xyzw_to_rot6d(quaternion))

    assert abs(float(np.dot(quaternion, round_trip))) > 1.0 - 1e-5


def test_zadai_rot6d_geodesic_clamp():
    identity = quaternion_xyzw_to_rot6d(np.asarray([0, 0, 0, 1], dtype=np.float32))
    half_turn = quaternion_xyzw_to_rot6d(np.asarray([0, 0, 1, 0], dtype=np.float32))

    clamped = clamp_rot6d_angle(half_turn, identity, max_angle_rad=0.1)

    np.testing.assert_allclose(
        rot6d_angular_distance(identity, clamped), 0.1, atol=1e-5
    )
