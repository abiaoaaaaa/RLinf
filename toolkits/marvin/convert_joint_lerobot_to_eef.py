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
"""Convert a Marvin dual-arm LeRobot v3 dataset from joints to absolute EEF.

The source layout is the 16-D Marvin convention::

    [Joint1_L..Joint7_L, Joint1_R..Joint7_R, gripper_L, gripper_R]

The primary state and action are rewritten to the canonical 20-D layout::

    [L_xyz(3), L_rot6d(6), L_grip, R_xyz(3), R_rot6d(6), R_grip]

Both observation joints and commanded action joints are passed through the
same Pinocchio MJCF model.  In particular, the action is *not* approximated by
the next observation.  Absolute EEF targets remain the dataset source of truth;
the OpenPI data transform computes chunk-start body-frame SE(3) deltas during
training.

Videos are reused without decoding.  The default media mode creates hardlinks,
so source and destination must be on the same filesystem (``copy`` and
``symlink`` are also available).

Example::

    PYTHONPATH=. python3 toolkits/marvin/convert_joint_lerobot_to_eef.py \
        --src /path/to/zadai_200 \
        --dst /path/to/zadai_200_eef \
        --rlinf-v21-dst /path/to/zadai_200_eef_rlinf_v21 \
        --mjcf /path/to/Apex_Deploy-main/robot_node/marvin_description/mjcf/matrix/m6_696.xml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

SOURCE_DIM = 16
ARM_JOINT_DIM = 14
EEF_DIM = 20
EEF_QUAT_DIM = 14
EEF_TWIST_DIM = 12

LEFT_GRIPPER_INDEX = 14
RIGHT_GRIPPER_INDEX = 15
LEFT_FRAME = "left_tool"
RIGHT_FRAME = "right_tool"
ROOT_FRAME = "base_link"

JOINT_NAMES = [
    *(f"Joint{i}_L" for i in range(1, 8)),
    *(f"Joint{i}_R" for i in range(1, 8)),
]
EEF_NAMES = [
    *(f"left_eef_xyz_{axis}" for axis in "xyz"),
    *(f"left_eef_rot6d_{i}" for i in range(6)),
    "left_gripper",
    *(f"right_eef_xyz_{axis}" for axis in "xyz"),
    *(f"right_eef_rot6d_{i}" for i in range(6)),
    "right_gripper",
]
EEF_QUAT_NAMES = [
    *(f"left_eef_xyz_{axis}" for axis in "xyz"),
    *(f"left_eef_quat_{axis}" for axis in "xyzw"),
    *(f"right_eef_xyz_{axis}" for axis in "xyz"),
    *(f"right_eef_quat_{axis}" for axis in "xyzw"),
]
EEF_TWIST_NAMES = [
    *(f"left_eef_linear_velocity_{axis}" for axis in "xyz"),
    *(f"left_eef_angular_velocity_{axis}" for axis in "xyz"),
    *(f"right_eef_linear_velocity_{axis}" for axis in "xyz"),
    *(f"right_eef_angular_velocity_{axis}" for axis in "xyz"),
]

GENERATED_FEATURES = (
    "observation.state",
    "action",
    "observation.joint_position",
    "observation.joint_velocity",
    "observation.gripper_velocity",
    "observation.eef_pose_quat",
    "observation.eef_twist",
    "action.joint_position_source",
    "action.eef_pose_quat",
    "annotation.is_rl_phase",
    "annotation.outcome",
)


def matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    """Encode the first two rotation-matrix columns like ``rlinf.utils.rot6d``.

    This small local copy intentionally keeps the offline converter independent
    of importing the full ``rlinf`` package.  APEX conversion environments need
    Pinocchio and PyArrow, but do not otherwise need the training stack or Torch.
    """
    rotation = np.asarray(rotation)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3), got {rotation.shape}")
    return np.concatenate([rotation[..., :, 0], rotation[..., :, 1]], axis=-1).astype(
        np.float32
    )


class ForwardKinematics(Protocol):
    """Minimal FK interface used by the conversion core."""

    def poses(
        self, joints: np.ndarray, joint_velocities: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return bilateral ``xyz+xyzw`` poses and optional world-aligned twists."""


class PinocchioMarvinFK:
    """APEX-compatible Pinocchio FK for ``base_link`` to Marvin tool frames."""

    def __init__(self, mjcf: Path) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(
                "Pinocchio Python bindings are required. Run this script in the "
                "APEX/ROS environment that provides `import pinocchio`."
            ) from exc

        if not mjcf.is_file():
            raise FileNotFoundError(f"MJCF does not exist: {mjcf}")
        self.pin = pin
        self.model = pin.buildModelFromMJCF(str(mjcf.resolve()))
        self.data = self.model.createData()

        if self.model.nq != ARM_JOINT_DIM or self.model.nv != ARM_JOINT_DIM:
            raise ValueError(
                f"expected fixed-base 14-DoF MJCF, got nq={self.model.nq}, "
                f"nv={self.model.nv}"
            )
        actual_joints = list(self.model.names)[1:]
        if actual_joints != JOINT_NAMES:
            raise ValueError(
                "MJCF joint order differs from the dataset convention:\n"
                f"expected={JOINT_NAMES}\nactual={actual_joints}"
            )
        for frame in (ROOT_FRAME, LEFT_FRAME, RIGHT_FRAME):
            if not self.model.existFrame(frame):
                raise ValueError(f"MJCF is missing required frame {frame!r}")
        self.root_id = self.model.getFrameId(ROOT_FRAME)
        self.left_id = self.model.getFrameId(LEFT_FRAME)
        self.right_id = self.model.getFrameId(RIGHT_FRAME)

    def poses(
        self, joints: np.ndarray, joint_velocities: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Run FK per row using the same frames as APEX teleop and QP."""
        joints = np.asarray(joints, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[1] != ARM_JOINT_DIM:
            raise ValueError(f"expected joints shape (T, 14), got {joints.shape}")
        if joint_velocities is not None:
            joint_velocities = np.asarray(joint_velocities, dtype=np.float64)
            if joint_velocities.shape != joints.shape:
                raise ValueError(
                    "joint velocities must have the same shape as joints, got "
                    f"{joint_velocities.shape} vs {joints.shape}"
                )
        if not np.isfinite(joints).all():
            raise ValueError("joint positions contain NaN or infinity")

        poses = np.empty((len(joints), EEF_QUAT_DIM), dtype=np.float32)
        twists = (
            np.empty((len(joints), EEF_TWIST_DIM), dtype=np.float32)
            if joint_velocities is not None
            else None
        )
        pin = self.pin
        for row, q in enumerate(tqdm.tqdm(joints, desc="Pinocchio FK", leave=False)):
            if joint_velocities is None:
                pin.forwardKinematics(self.model, self.data, q)
            else:
                pin.forwardKinematics(self.model, self.data, q, joint_velocities[row])
            pin.updateFramePlacements(self.model, self.data)

            root = self.data.oMf[self.root_id]
            left = root.inverse() * self.data.oMf[self.left_id]
            right = root.inverse() * self.data.oMf[self.right_id]
            poses[row, :7] = np.asarray(pin.SE3ToXYZQUAT(left), dtype=np.float32)
            poses[row, 7:] = np.asarray(pin.SE3ToXYZQUAT(right), dtype=np.float32)

            if twists is not None:
                reference = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                left_v = pin.getFrameVelocity(
                    self.model, self.data, self.left_id, reference
                )
                right_v = pin.getFrameVelocity(
                    self.model, self.data, self.right_id, reference
                )
                twists[row] = np.concatenate(
                    [left_v.linear, left_v.angular, right_v.linear, right_v.angular]
                ).astype(np.float32)
        return poses, twists


def _as_matrix(column: pa.ChunkedArray, expected_dim: int, key: str) -> np.ndarray:
    """Convert either variable or fixed-size Arrow list columns to a matrix."""
    values = column.combine_chunks().to_pylist()
    out = np.asarray(values, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] != expected_dim:
        raise ValueError(f"{key}: expected shape (T, {expected_dim}), got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{key}: contains NaN or infinity")
    return out


def eef_quat_to_rot6d(eef_quat: np.ndarray, grippers: np.ndarray) -> np.ndarray:
    """Pack bilateral ``xyz+xyzw`` and grippers into canonical 20-D EEF."""
    eef_quat = np.asarray(eef_quat, dtype=np.float32)
    grippers = np.asarray(grippers, dtype=np.float32)
    if eef_quat.ndim != 2 or eef_quat.shape[1] != EEF_QUAT_DIM:
        raise ValueError(f"expected EEF poses shape (T, 14), got {eef_quat.shape}")
    if grippers.shape != (len(eef_quat), 2):
        raise ValueError(f"expected grippers shape (T, 2), got {grippers.shape}")

    left_rot6d = matrix_to_rot6d(_quat_xyzw_to_matrix(eef_quat[:, 3:7]))
    right_rot6d = matrix_to_rot6d(_quat_xyzw_to_matrix(eef_quat[:, 10:14]))
    return np.concatenate(
        [
            eef_quat[:, 0:3],
            left_rot6d,
            grippers[:, 0:1],
            eef_quat[:, 7:10],
            right_rot6d,
            grippers[:, 1:2],
        ],
        axis=1,
    ).astype(np.float32)


def _quat_xyzw_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert normalized xyzw quaternions without adding a SciPy dependency."""
    q = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(q, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            "FK returned non-unit quaternions; norm range "
            f"[{norms.min():.6f}, {norms.max():.6f}]"
        )
    x, y, z, w = (q[:, i] for i in range(4))
    matrices = np.empty((len(q), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def convert_arrays(
    state: np.ndarray,
    action: np.ndarray,
    velocity: np.ndarray,
    fk: ForwardKinematics,
) -> dict[str, np.ndarray]:
    """Convert one aligned source block and retain audit-friendly features."""
    state = np.asarray(state, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    expected = (len(state), SOURCE_DIM)
    for key, value in (("state", state), ("action", action), ("velocity", velocity)):
        if value.shape != expected:
            raise ValueError(f"{key}: expected shape {expected}, got {value.shape}")

    observation_pose, eef_twist = fk.poses(state[:, :14], velocity[:, :14])
    action_pose, _ = fk.poses(action[:, :14])
    if eef_twist is None:
        raise RuntimeError(
            "FK backend did not return twists for observation velocities"
        )

    return {
        "observation.state": eef_quat_to_rot6d(observation_pose, state[:, 14:16]),
        "action": eef_quat_to_rot6d(action_pose, action[:, 14:16]),
        "observation.joint_position": state[:, :14].copy(),
        "observation.joint_velocity": velocity[:, :14].copy(),
        "observation.gripper_velocity": velocity[:, 14:16].copy(),
        "observation.eef_pose_quat": observation_pose.astype(np.float32),
        "observation.eef_twist": eef_twist.astype(np.float32),
        "action.joint_position_source": action[:, :14].copy(),
        "action.eef_pose_quat": action_pose.astype(np.float32),
        # -1 means unlabelled.  The later annotation program must replace it.
        "annotation.is_rl_phase": np.full(len(state), -1, dtype=np.int8),
        "annotation.outcome": np.full(len(state), -1, dtype=np.int8),
    }


def array_stats(array: np.ndarray) -> dict[str, list]:
    """Compute LeRobot v3 numeric statistics for one feature."""
    value = np.asarray(array)
    if value.ndim == 1:
        value = value[:, None]
    value64 = value.astype(np.float64)
    quantiles = np.quantile(value64, [0.01, 0.1, 0.5, 0.9, 0.99], axis=0)
    return {
        "min": value64.min(axis=0).tolist(),
        "max": value64.max(axis=0).tolist(),
        "mean": value64.mean(axis=0).tolist(),
        "std": value64.std(axis=0).tolist(),
        "count": [int(len(value64))],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }


def _arrow_array(value: np.ndarray) -> pa.Array:
    if value.ndim == 1:
        return pa.array(value)
    return pa.array(value.tolist(), type=pa.list_(pa.float32()))


def _replace_data_columns(
    table: pa.Table, generated: dict[str, np.ndarray]
) -> pa.Table:
    source_columns = {name: table[name] for name in table.column_names}
    source_columns.pop("observation.velocity", None)
    source_columns.update(
        {key: _arrow_array(value) for key, value in generated.items()}
    )
    ordered = [
        *generated.keys(),
        *(
            name
            for name in table.column_names
            if name not in generated and name != "observation.velocity"
        ),
    ]
    result = pa.table({name: source_columns[name] for name in ordered})
    # The original pandas schema no longer matches after adding/replacing fields.
    metadata = dict(table.schema.metadata or {})
    metadata.pop(b"pandas", None)
    return result.replace_schema_metadata(metadata or None)


def _feature_info() -> dict[str, dict]:
    return {
        "observation.state": {"dtype": "float32", "shape": [20], "names": EEF_NAMES},
        "action": {"dtype": "float32", "shape": [20], "names": EEF_NAMES},
        "observation.joint_position": {
            "dtype": "float32",
            "shape": [14],
            "names": JOINT_NAMES,
        },
        "observation.joint_velocity": {
            "dtype": "float32",
            "shape": [14],
            "names": [f"{name}_velocity" for name in JOINT_NAMES],
        },
        "observation.gripper_velocity": {
            "dtype": "float32",
            "shape": [2],
            "names": ["gripper_L_velocity", "gripper_R_velocity"],
        },
        "observation.eef_pose_quat": {
            "dtype": "float32",
            "shape": [14],
            "names": EEF_QUAT_NAMES,
        },
        "observation.eef_twist": {
            "dtype": "float32",
            "shape": [12],
            "names": EEF_TWIST_NAMES,
        },
        "action.joint_position_source": {
            "dtype": "float32",
            "shape": [14],
            "names": JOINT_NAMES,
        },
        "action.eef_pose_quat": {
            "dtype": "float32",
            "shape": [14],
            "names": EEF_QUAT_NAMES,
        },
        "annotation.is_rl_phase": {
            "dtype": "int8",
            "shape": [1],
            "names": ["is_rl_phase_-1_unknown_0_demo_1_rl"],
        },
        "annotation.outcome": {
            "dtype": "int8",
            "shape": [1],
            "names": ["outcome_-1_unknown_0_failure_1_success"],
        },
    }


def _rewrite_info(src: Path, dst: Path, mjcf: Path) -> None:
    with src.open(encoding="utf-8") as handle:
        info = json.load(handle)
    features = info["features"]
    features.pop("observation.velocity", None)
    features.update(_feature_info())
    info["eef_conversion"] = {
        "version": "marvin_eef_rot6d_v1",
        "source_action": "joint_position_target_fk",
        "root_frame": ROOT_FRAME,
        "left_eef_frame": LEFT_FRAME,
        "right_eef_frame": RIGHT_FRAME,
        "position_unit": "meter",
        "joint_unit": "radian",
        "quaternion_order": "xyzw",
        "rot6d_convention": "first_two_rotation_matrix_columns",
        "training_delta": "inverse_chunk_start_observation_times_absolute_target",
        "mjcf_path_at_conversion": str(mjcf.resolve()),
    }
    with dst.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _rewrite_global_stats(
    src: Path, dst: Path, global_values: dict[str, list[np.ndarray]]
) -> None:
    with src.open(encoding="utf-8") as handle:
        stats = json.load(handle)
    stats.pop("observation.velocity", None)
    for key, pieces in global_values.items():
        stats[key] = array_stats(np.concatenate(pieces, axis=0))
    with dst.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _rewrite_episode_metadata(
    src: Path,
    dst: Path,
    episode_values: dict[int, dict[str, list[np.ndarray]]],
) -> None:
    table = pq.read_table(src)
    kept = {
        name: table[name]
        for name in table.column_names
        if not name.startswith("stats/observation.state/")
        and not name.startswith("stats/action/")
        and not name.startswith("stats/observation.velocity/")
    }
    episodes = np.asarray(table["episode_index"])
    generated_columns: dict[str, list] = defaultdict(list)
    for episode in episodes:
        values = episode_values[int(episode)]
        for feature in GENERATED_FEATURES:
            stats = array_stats(np.concatenate(values[feature], axis=0))
            for stat_name, stat_value in stats.items():
                generated_columns[f"stats/{feature}/{stat_name}"].append(stat_value)
    kept.update({name: pa.array(values) for name, values in generated_columns.items()})
    result = pa.table(kept)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(result, dst)


def _copy_file(src: str, dst: str, mode: str) -> str:
    if mode == "copy":
        return shutil.copy2(src, dst)
    if mode == "hardlink":
        os.link(src, dst)
        return dst
    raise AssertionError(mode)


def _reuse_videos(src: Path, dst: Path, mode: str) -> None:
    if not src.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
        return
    shutil.copytree(src, dst, copy_function=lambda a, b: _copy_file(a, b, mode))


def convert_dataset(
    src: Path,
    dst: Path,
    mjcf: Path,
    media_mode: str = "hardlink",
    overwrite: bool = False,
) -> None:
    """Convert a complete LeRobot v3 root without touching the source."""
    src = src.resolve()
    dst = dst.resolve()
    if src == dst or src in dst.parents:
        raise ValueError("destination must not equal or be inside the source dataset")
    for required in (src / "meta/info.json", src / "meta/stats.json", src / "data"):
        if not required.exists():
            raise FileNotFoundError(f"source is missing {required.relative_to(src)}")

    with (src / "meta/info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"this converter requires LeRobot v3.0, got {info.get('codebase_version')!r}"
        )
    for key in ("observation.state", "action", "observation.velocity"):
        shape = info.get("features", {}).get(key, {}).get("shape")
        if shape != [SOURCE_DIM]:
            raise ValueError(f"{key}: expected source shape [16], got {shape}")

    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"destination exists: {dst}; pass --overwrite to replace it"
            )
        shutil.rmtree(dst)
    (dst / "data").mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)

    fk = PinocchioMarvinFK(mjcf)
    global_values: dict[str, list[np.ndarray]] = defaultdict(list)
    episode_values: dict[int, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    parquet_files = sorted((src / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found under {src / 'data'}")

    for source_file in tqdm.tqdm(parquet_files, desc="data files"):
        table = pq.read_table(source_file)
        state = _as_matrix(table["observation.state"], SOURCE_DIM, "observation.state")
        action = _as_matrix(table["action"], SOURCE_DIM, "action")
        velocity = _as_matrix(
            table["observation.velocity"], SOURCE_DIM, "observation.velocity"
        )
        generated = convert_arrays(state, action, velocity, fk)
        converted_table = _replace_data_columns(table, generated)
        target_file = dst / "data" / source_file.relative_to(src / "data")
        target_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(converted_table, target_file, compression="zstd")

        episode_indices = np.asarray(table["episode_index"])
        for key, value in generated.items():
            global_values[key].append(value)
            for episode in np.unique(episode_indices):
                episode_values[int(episode)][key].append(
                    value[episode_indices == episode]
                )

    _rewrite_info(src / "meta/info.json", dst / "meta/info.json", mjcf)
    _rewrite_global_stats(
        src / "meta/stats.json", dst / "meta/stats.json", global_values
    )
    episodes_dir = src / "meta/episodes"
    if episodes_dir.exists():
        for source_file in sorted(episodes_dir.rglob("*.parquet")):
            target_file = dst / "meta/episodes" / source_file.relative_to(episodes_dir)
            _rewrite_episode_metadata(source_file, target_file, episode_values)
    tasks = src / "meta/tasks.parquet"
    if tasks.exists():
        shutil.copy2(tasks, dst / "meta/tasks.parquet")
    _reuse_videos(src / "videos", dst / "videos", media_mode)

    print(
        f"Converted {info['total_frames']} frames in {info['total_episodes']} episodes\n"
        f"source:      {src}\n"
        f"destination: {dst}\n"
        f"MJCF:        {mjcf.resolve()}\n"
        f"media mode:  {media_mode}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path, required=True, help="Source LeRobot v3 root"
    )
    parser.add_argument(
        "--dst", type=Path, required=True, help="Fresh output dataset root"
    )
    parser.add_argument(
        "--rlinf-v21-dst",
        type=Path,
        help=(
            "also materialize an RLinf-pinned LeRobot v2.1 dataset at this path; "
            "the EEF v3 dataset at --dst remains the conversion source of truth"
        ),
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        required=True,
        help="Exact runtime APEX MJCF used by the robot (for example matrix/m6_696.xml)",
    )
    parser.add_argument(
        "--media-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to reuse unchanged videos (default: hardlink)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing destination before conversion; never changes --src",
    )
    args = parser.parse_args()
    try:
        convert_dataset(
            args.src,
            args.dst,
            args.mjcf,
            media_mode=args.media_mode,
            overwrite=args.overwrite,
        )
        if args.rlinf_v21_dst is not None:
            from toolkits.lerobot.convert_zadai_v3_to_rlinf_v21 import (
                convert as convert_v3_to_v21,
            )

            convert_v3_to_v21(args.dst, args.rlinf_v21_dst, args.media_mode)
            print(f"RLinf v2.1:  {args.rlinf_v21_dst.resolve()}")
    except (
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
