# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Convert a persisted Zadai Stage-2 joint replay buffer to absolute EEF.

The source buffer is read-only.  A complete new ``TrajectoryReplayBuffer``
rank directory is created at ``--dst``; an existing destination is rejected.

Converted tensor fields:

* ``actions`` and ``forward_inputs/action``: normalized 16-D joint chunks to
  normalized 20-D bilateral EEF chunks;
* every ``proprio`` field: physical 16-D joint/gripper state to 20-D EEF state;
* every ``ref_chunk`` field: normalized joint reference to normalized EEF;
* action-shaped intervention/version masks are expanded from 16 to 20 values.

Rewards, terminal flags, Q/RL metadata, and ``z_rl`` are copied exactly.  The
last point matters: old replay files do not contain camera images, so their
``z_rl`` cannot be recomputed.  The output manifest records this limitation.
Do not mix the converted buffer with live features from a different Stage-1
checkpoint unless you intentionally accept that representation shift.

Example::

    python3 toolkits/marvin/convert_online_buffer_joint_to_eef.py \
        --src replay_buffers/zadai_200_rlt_stage2/rank_0 \
        --dst replay_buffers/zadai_200_rlt_stage2_eef/rank_0 \
        --mjcf /path/to/marvin_pro_mink.xml
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.envs.realworld.zadai.action_codec import eef_absolute_to_normalized
from toolkits.marvin.convert_joint_lerobot_to_eef import (
    EEF_DIM,
    PinocchioMarvinFK,
    eef_quat_to_rot6d,
)

SOURCE_DIM = 16
ARM_DIM = 14
JOINT_LOW = np.full(ARM_DIM, -np.pi, dtype=np.float32)
JOINT_HIGH = np.full(ARM_DIM, np.pi, dtype=np.float32)
ACTION_LOW = np.concatenate([JOINT_LOW, [0.0, 0.0]]).astype(np.float32)
ACTION_HIGH = np.concatenate([JOINT_HIGH, [1.0, 1.0]]).astype(np.float32)


def _atomic_json(path: Path, value: Any) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def normalized_to_absolute(value: np.ndarray) -> np.ndarray:
    """Decode the existing Zadai 16-D RLT action domain."""
    value = np.asarray(value, dtype=np.float32)
    return (value + 1.0) * 0.5 * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW


def _convert_physical_rows(rows: np.ndarray, fk: PinocchioMarvinFK) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != SOURCE_DIM:
        raise ValueError(f"expected physical rows (N, 16), got {rows.shape}")
    poses, _ = fk.poses(rows[:, :ARM_DIM])
    return eef_quat_to_rot6d(poses, rows[:, 14:16])


def _convert_tensor_last_dim(
    tensor: Any,
    torch: Any,
    fk: PinocchioMarvinFK,
    *,
    normalized: bool,
    output_policy: bool = False,
    flattened_chunk: bool = False,
) -> Any:
    array = tensor.detach().cpu().float().numpy()
    original_shape = array.shape
    if flattened_chunk:
        if array.shape[-1] % SOURCE_DIM:
            raise ValueError(
                f"flattened joint chunk has invalid final dimension {array.shape[-1]}"
            )
        rows = array.reshape(-1, SOURCE_DIM)
        chunk_len = original_shape[-1] // SOURCE_DIM
        output_shape = (*original_shape[:-1], chunk_len * EEF_DIM)
    else:
        if array.shape[-1] != SOURCE_DIM:
            raise ValueError(f"expected final dimension 16, got {array.shape}")
        rows = array.reshape(-1, SOURCE_DIM)
        output_shape = (*original_shape[:-1], EEF_DIM)
    physical = normalized_to_absolute(rows) if normalized else rows
    converted = _convert_physical_rows(physical, fk)
    if output_policy:
        converted = np.asarray(eef_absolute_to_normalized(converted), dtype=np.float32)
    converted = converted.reshape(output_shape)
    return torch.from_numpy(converted).to(dtype=tensor.dtype, device="cpu")


def _convert_action_mask(tensor: Any, torch: Any) -> Any:
    if tensor.shape[-1] % SOURCE_DIM:
        return tensor.detach().clone().cpu()
    shape = tensor.shape
    chunks = shape[-1] // SOURCE_DIM
    rows = tensor.detach().cpu().reshape(*shape[:-1], chunks, SOURCE_DIM)
    if tensor.dtype == torch.bool:
        step = rows.any(dim=-1, keepdim=True)
    else:
        step = rows[..., :1]
    return (
        step.expand(*shape[:-1], chunks, EEF_DIM)
        .reshape(*shape[:-1], chunks * EEF_DIM)
        .clone()
    )


def _convert_obs_dict(
    value: dict[str, Any], torch: Any, fk: PinocchioMarvinFK
) -> dict[str, Any]:
    result = copy.deepcopy(value)
    if torch.is_tensor(result.get("proprio")):
        result["proprio"] = _convert_tensor_last_dim(
            result["proprio"], torch, fk, normalized=False
        )
    if torch.is_tensor(result.get("ref_chunk")):
        result["ref_chunk"] = _convert_tensor_last_dim(
            result["ref_chunk"], torch, fk, normalized=True, output_policy=True
        )
    return result


def convert_trajectory(
    data: dict[str, Any], torch: Any, fk: PinocchioMarvinFK
) -> dict[str, Any]:
    """Return a converted deep copy of one serialized Trajectory dictionary."""
    result = copy.deepcopy(data)
    if torch.is_tensor(result.get("actions")):
        result["actions"] = _convert_tensor_last_dim(
            result["actions"],
            torch,
            fk,
            normalized=True,
            output_policy=True,
            flattened_chunk=True,
        )
    for key in ("intervene_flags", "versions"):
        if torch.is_tensor(result.get(key)) and result[key].shape[-1] % SOURCE_DIM == 0:
            result[key] = _convert_action_mask(result[key], torch)
    for key in ("curr_obs", "next_obs"):
        if isinstance(result.get(key), dict):
            result[key] = _convert_obs_dict(result[key], torch, fk)

    forward = result.get("forward_inputs")
    if isinstance(forward, dict):
        if torch.is_tensor(forward.get("action")):
            forward["action"] = _convert_tensor_last_dim(
                forward["action"],
                torch,
                fk,
                normalized=True,
                output_policy=True,
                flattened_chunk=True,
            )
        for key in ("proprio", "rlt_transition_proprio"):
            if torch.is_tensor(forward.get(key)):
                forward[key] = _convert_tensor_last_dim(
                    forward[key], torch, fk, normalized=False
                )
        for key in ("ref_chunk", "rlt_transition_ref_chunk"):
            if torch.is_tensor(forward.get(key)):
                forward[key] = _convert_tensor_last_dim(
                    forward[key],
                    torch,
                    fk,
                    normalized=True,
                    output_policy=True,
                )
    return result


def _validate_source(src: Path) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    for name in ("metadata.json", "trajectory_index.json"):
        if not (src / name).is_file():
            raise FileNotFoundError(f"source buffer is missing {name}: {src}")
    metadata = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
    index = json.loads((src / "trajectory_index.json").read_text(encoding="utf-8"))
    files = sorted(src.glob("trajectory_*.pt"))
    if len(files) != int(metadata["size"]):
        raise ValueError(
            f"metadata size={metadata['size']} but found {len(files)} trajectory files"
        )
    return metadata, index, files


def convert_buffer(src: Path, dst: Path, mjcf: Path) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("buffer conversion requires PyTorch") from exc

    src = src.resolve()
    dst = dst.resolve()
    if src == dst or src in dst.parents:
        raise ValueError("destination must not equal or be inside the source buffer")
    if dst.exists():
        raise FileExistsError(f"destination exists; refusing to modify it: {dst}")
    metadata, index, files = _validate_source(src)
    fk = PinocchioMarvinFK(mjcf)
    staging = dst.with_name(f".{dst.name}.tmp-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        for number, source_file in enumerate(files, start=1):
            trajectory = torch.load(source_file, map_location="cpu", weights_only=False)
            if not isinstance(trajectory, dict):
                raise ValueError(f"{source_file.name}: expected a dictionary")
            converted = convert_trajectory(trajectory, torch, fk)
            torch.save(converted, staging / source_file.name)
            if number % 100 == 0 or number == len(files):
                print(f"converted {number}/{len(files)} trajectories", flush=True)
        _atomic_json(staging / "metadata.json", metadata)
        _atomic_json(staging / "trajectory_index.json", index)
        _atomic_json(
            staging / "eef_buffer_manifest.json",
            {
                "format": "rlinf_rlt_eef_buffer_v1",
                "source_buffer": str(src),
                "source_buffer_unchanged": True,
                "mjcf": str(mjcf.resolve()),
                "source_action": "normalized_absolute_joint_16d",
                "target_action": "normalized_bilateral_eef_xyz_rot6d_gripper_20d",
                "target_proprio": "absolute_bilateral_eef_xyz_rot6d_gripper_20d",
                "z_rl": "copied_without_recomputation",
                "z_rl_compatibility": (
                    "safe only with the source Stage-1 feature representation; "
                    "source replay has no images for recomputation"
                ),
            },
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(dst)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"source preserved: {src}")
    print(f"EEF buffer:       {dst}")
    print(
        "warning: z_rl was preserved; see eef_buffer_manifest.json before mixing checkpoints"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    args = parser.parse_args()
    try:
        convert_buffer(args.src, args.dst, args.mjcf)
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
