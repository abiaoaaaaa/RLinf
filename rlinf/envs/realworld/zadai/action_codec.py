# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Action-domain codecs shared by Zadai OpenPI, RLT, and the HTTP env."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

ControlSpace = Literal["joint", "eef"]

JOINT_ACTION_DIM = 16
EEF_ACTION_DIM = 20
JOINT_LOW = np.full(14, -np.pi, dtype=np.float32)
JOINT_HIGH = np.full(14, np.pi, dtype=np.float32)
ACTION_LOW = np.concatenate([JOINT_LOW, [0.0, 0.0]]).astype(np.float32)
ACTION_HIGH = np.concatenate([JOINT_HIGH, [1.0, 1.0]]).astype(np.float32)

# Conservative task-local Cartesian boxes. They cover the converted Zadai
# demonstrations with margin and can later be made task-specific without
# changing the Stage 2 action semantics.
DEFAULT_EEF_XYZ_LOW = np.asarray(
    [[0.20, -0.10, 0.65], [0.15, -0.45, 0.65]], dtype=np.float32
)
DEFAULT_EEF_XYZ_HIGH = np.asarray(
    [[0.70, 0.55, 1.10], [0.60, 0.10, 1.12]], dtype=np.float32
)


def action_dim(control_space: ControlSpace) -> int:
    """Return the public Stage 2 action dimension for a control space."""
    if control_space == "joint":
        return JOINT_ACTION_DIM
    if control_space == "eef":
        return EEF_ACTION_DIM
    raise ValueError(f"unsupported Zadai control_space: {control_space!r}")


def _constants_like(value, array: np.ndarray):
    if torch.is_tensor(value):
        return torch.as_tensor(array, device=value.device, dtype=value.dtype)
    return array


def absolute_to_normalized(action):
    """Map physical absolute targets to the RLT ``[-1, 1]`` domain."""
    low = _constants_like(action, ACTION_LOW)
    high = _constants_like(action, ACTION_HIGH)
    return 2.0 * (action - low) / (high - low) - 1.0


def normalized_to_absolute(action):
    """Map RLT ``[-1, 1]`` targets to physical absolute targets."""
    low = _constants_like(action, ACTION_LOW)
    high = _constants_like(action, ACTION_HIGH)
    return (action + 1.0) * 0.5 * (high - low) + low


def canonicalize_rot6d(rot6d):
    """Project first-two-column rot6d values onto a valid SO(3) frame."""
    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d must end in 6 values, got {rot6d.shape}")
    if torch.is_tensor(rot6d):
        a1, a2 = rot6d[..., :3], rot6d[..., 3:]
        norm1 = torch.linalg.vector_norm(a1, dim=-1, keepdim=True)
        fallback1 = torch.zeros_like(a1)
        fallback1[..., 0] = 1.0
        b1 = torch.where(norm1 > 1e-8, a1 / norm1.clamp_min(1e-8), fallback1)
        orthogonal = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        norm2 = torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True)
        axis = torch.nn.functional.one_hot(
            torch.argmin(torch.abs(b1), dim=-1), num_classes=3
        ).to(dtype=rot6d.dtype)
        fallback2 = axis - (b1 * axis).sum(dim=-1, keepdim=True) * b1
        fallback2 = fallback2 / torch.linalg.vector_norm(
            fallback2, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        b2 = torch.where(
            norm2 > 1e-8,
            orthogonal / norm2.clamp_min(1e-8),
            fallback2,
        )
        return torch.cat([b1, b2], dim=-1)
    value = np.asarray(rot6d)
    a1, a2 = value[..., :3], value[..., 3:]
    norm1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    fallback1 = np.zeros_like(a1)
    fallback1[..., 0] = 1.0
    b1 = np.where(norm1 > 1e-8, a1 / np.maximum(norm1, 1e-8), fallback1)
    orthogonal = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    norm2 = np.linalg.norm(orthogonal, axis=-1, keepdims=True)
    axis = np.eye(3, dtype=value.dtype)[np.argmin(np.abs(b1), axis=-1)]
    fallback2 = axis - np.sum(b1 * axis, axis=-1, keepdims=True) * b1
    fallback2 /= np.maximum(np.linalg.norm(fallback2, axis=-1, keepdims=True), 1e-8)
    b2 = np.where(
        norm2 > 1e-8,
        orthogonal / np.maximum(norm2, 1e-8),
        fallback2,
    )
    return np.concatenate([b1, b2], axis=-1)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert the Zadai first-two-columns representation to rotation matrices."""
    value = canonicalize_rot6d(np.asarray(rot6d, dtype=np.float32))
    first, second = value[..., :3], value[..., 3:]
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-1)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Encode the first two columns of a rotation matrix."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in (3, 3), got {matrix.shape}")
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def quaternion_xyzw_to_rot6d(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized xyzw quaternions to the dataset rot6d convention."""
    q = np.asarray(quaternion, dtype=np.float32)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion must end in 4 values, got {q.shape}")
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    x, y, z, w = np.moveaxis(q, -1, 0)
    matrix = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return matrix_to_rot6d(matrix)


def rot6d_to_quaternion_xyzw(rot6d: np.ndarray) -> np.ndarray:
    """Convert dataset rot6d values to unit xyzw quaternions."""
    matrix = rot6d_to_matrix(rot6d)
    flat = matrix.reshape(-1, 3, 3)
    output = np.empty((len(flat), 4), dtype=np.float32)
    for index, rotation in enumerate(flat):
        trace = float(np.trace(rotation))
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2.0
            q = np.asarray(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    0.25 * scale,
                ]
            )
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = (
                    np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
                )
                q = np.asarray(
                    [
                        0.25 * scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                        (rotation[2, 1] - rotation[1, 2]) / scale,
                    ]
                )
            elif axis == 1:
                scale = (
                    np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
                )
                q = np.asarray(
                    [
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        0.25 * scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                        (rotation[0, 2] - rotation[2, 0]) / scale,
                    ]
                )
            else:
                scale = (
                    np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
                )
                q = np.asarray(
                    [
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                        0.25 * scale,
                        (rotation[1, 0] - rotation[0, 1]) / scale,
                    ]
                )
        output[index] = q / max(float(np.linalg.norm(q)), 1e-8)
    return output.reshape(matrix.shape[:-2] + (4,))


def clamp_rot6d_angle(
    target: np.ndarray, reference: np.ndarray, max_angle_rad: float
) -> np.ndarray:
    """Clamp a target rotation to a geodesic angle around a reference."""
    if max_angle_rad <= 0:
        raise ValueError("max_angle_rad must be positive")
    target_q = rot6d_to_quaternion_xyzw(np.asarray(target, dtype=np.float32))
    reference_q = rot6d_to_quaternion_xyzw(np.asarray(reference, dtype=np.float32))
    dot = float(np.dot(reference_q, target_q))
    if dot < 0.0:
        target_q = -target_q
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    angle = 2.0 * float(np.arccos(dot))
    if angle <= max_angle_rad + 1e-8:
        return quaternion_xyzw_to_rot6d(target_q)
    fraction = max_angle_rad / angle
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    if sin_theta < 1e-7:
        interpolated = (1.0 - fraction) * reference_q + fraction * target_q
    else:
        interpolated = (
            np.sin((1.0 - fraction) * theta) / sin_theta * reference_q
            + np.sin(fraction * theta) / sin_theta * target_q
        )
    interpolated /= max(float(np.linalg.norm(interpolated)), 1e-8)
    return quaternion_xyzw_to_rot6d(interpolated)


def rot6d_angular_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest SO(3) angle between two rot6d rotations."""
    first_q = rot6d_to_quaternion_xyzw(np.asarray(first, dtype=np.float32))
    second_q = rot6d_to_quaternion_xyzw(np.asarray(second, dtype=np.float32))
    dot = float(np.clip(abs(np.dot(first_q, second_q)), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def eef_absolute_to_normalized(
    action,
    xyz_low: np.ndarray = DEFAULT_EEF_XYZ_LOW,
    xyz_high: np.ndarray = DEFAULT_EEF_XYZ_HIGH,
):
    """Encode physical absolute bilateral EEF targets for the RLT actor."""
    if action.shape[-1] != EEF_ACTION_DIM:
        raise ValueError(f"EEF action must end in 20 values, got {action.shape}")
    low = _constants_like(action, np.asarray(xyz_low, dtype=np.float32))
    high = _constants_like(action, np.asarray(xyz_high, dtype=np.float32))
    result = action.clone() if torch.is_tensor(action) else np.array(action, copy=True)
    for arm, offset in enumerate((0, 10)):
        result[..., offset : offset + 3] = (
            2.0 * (action[..., offset : offset + 3] - low[arm]) / (high[arm] - low[arm])
            - 1.0
        )
        result[..., offset + 3 : offset + 9] = canonicalize_rot6d(
            action[..., offset + 3 : offset + 9]
        )
        result[..., offset + 9] = 2.0 * action[..., offset + 9] - 1.0
    return result


def eef_normalized_to_absolute(
    action,
    xyz_low: np.ndarray = DEFAULT_EEF_XYZ_LOW,
    xyz_high: np.ndarray = DEFAULT_EEF_XYZ_HIGH,
):
    """Decode an RLT action to physical absolute bilateral EEF targets."""
    if action.shape[-1] != EEF_ACTION_DIM:
        raise ValueError(f"EEF action must end in 20 values, got {action.shape}")
    low = _constants_like(action, np.asarray(xyz_low, dtype=np.float32))
    high = _constants_like(action, np.asarray(xyz_high, dtype=np.float32))
    result = action.clone() if torch.is_tensor(action) else np.array(action, copy=True)
    for arm, offset in enumerate((0, 10)):
        result[..., offset : offset + 3] = (
            action[..., offset : offset + 3] + 1.0
        ) * 0.5 * (high[arm] - low[arm]) + low[arm]
        result[..., offset + 3 : offset + 9] = canonicalize_rot6d(
            action[..., offset + 3 : offset + 9]
        )
        result[..., offset + 9] = (action[..., offset + 9] + 1.0) * 0.5
    return result


def policy_to_absolute(action, control_space: ControlSpace):
    """Decode a joint or EEF Stage 2 policy action."""
    if control_space == "joint":
        return normalized_to_absolute(action)
    if control_space == "eef":
        return eef_normalized_to_absolute(action)
    raise ValueError(f"unsupported Zadai control_space: {control_space!r}")


def absolute_to_policy(action, control_space: ControlSpace):
    """Encode a joint or EEF physical action for Stage 2/replay."""
    if control_space == "joint":
        return absolute_to_normalized(action)
    if control_space == "eef":
        return eef_absolute_to_normalized(action)
    raise ValueError(f"unsupported Zadai control_space: {control_space!r}")
