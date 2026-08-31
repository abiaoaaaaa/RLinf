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
"""Calculate exact OpenPI stats for Zadai EEF body-frame action chunks.

This avoids decoding the three video streams.  It reproduces LeRobot action
chunk sampling: each anchor uses the following ``action_horizon`` absolute EEF
targets, with the final action repeated at the episode boundary.  Each target
is converted with ``inv(T_observation_at_anchor) @ T_absolute_target`` and both
state/actions are padded from 20 to the Pi0.5 model dimension (normally 32).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openpi.shared.normalize as normalize
import pyarrow.parquet as pq
import tqdm

from rlinf.utils.rot6d import matrix_to_rot6d, pose_to_SE3, se3_body_delta

EEF_DIM = 20
POSE_SLICES = (
    (slice(0, 3), slice(3, 9)),
    (slice(10, 13), slice(13, 19)),
)


def body_delta_action_chunks(
    states: np.ndarray, absolute_actions: np.ndarray, model_action_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pad states and convert ``(B,H,20)`` absolute targets to body deltas."""
    states = np.asarray(states, dtype=np.float32)
    absolute_actions = np.asarray(absolute_actions, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != EEF_DIM:
        raise ValueError(f"expected states (B,20), got {states.shape}")
    if absolute_actions.ndim != 3 or absolute_actions.shape[::2] != (
        len(states),
        EEF_DIM,
    ):
        raise ValueError(f"expected actions (B,H,20), got {absolute_actions.shape}")
    if model_action_dim < EEF_DIM:
        raise ValueError(
            f"model_action_dim must be at least 20, got {model_action_dim}"
        )

    padded_states = np.zeros((len(states), model_action_dim), dtype=np.float32)
    padded_states[:, :EEF_DIM] = states
    delta_actions = np.zeros(
        (*absolute_actions.shape[:2], model_action_dim), dtype=np.float32
    )
    delta_actions[..., :EEF_DIM] = absolute_actions

    for xyz_slice, rot6d_slice in POSE_SLICES:
        state_transform = pose_to_SE3(states[:, xyz_slice], states[:, rot6d_slice])
        target_transform = pose_to_SE3(
            absolute_actions[..., xyz_slice], absolute_actions[..., rot6d_slice]
        )
        delta_transform = se3_body_delta(state_transform[:, None], target_transform)
        delta_actions[..., xyz_slice] = delta_transform[..., :3, 3]
        delta_actions[..., rot6d_slice] = matrix_to_rot6d(delta_transform[..., :3, :3])
    return padded_states, delta_actions


def _matrix(table, key: str) -> np.ndarray:
    value = np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != EEF_DIM:
        raise ValueError(f"{key}: expected (T,20), got {value.shape}")
    return value


def calculate(
    dataset: Path,
    output_dir: Path,
    action_horizon: int = 50,
    model_action_dim: int = 32,
    batch_size: int = 2048,
) -> Path:
    """Calculate and save ``norm_stats.json`` without loading any images."""
    dataset = dataset.resolve()
    info_path = dataset / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"not a LeRobot dataset: {dataset}")
    with info_path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    for key in ("observation.state", "action"):
        if info["features"][key]["shape"] != [EEF_DIM]:
            raise ValueError(f"{key} must be 20-D EEF data")

    parquet_files = sorted((dataset / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet data under {dataset / 'data'}")
    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    frame_count = 0
    for parquet_file in tqdm.tqdm(parquet_files, desc="episodes/files"):
        table = pq.read_table(
            parquet_file, columns=["observation.state", "action", "episode_index"]
        )
        all_states = _matrix(table, "observation.state")
        all_actions = _matrix(table, "action")
        episode_indices = np.asarray(table["episode_index"], dtype=np.int64)
        for episode in np.unique(episode_indices):
            mask = episode_indices == episode
            episode_states = all_states[mask]
            episode_actions = all_actions[mask]
            offsets = np.arange(action_horizon, dtype=np.int64)
            for start in range(0, len(episode_states), batch_size):
                stop = min(start + batch_size, len(episode_states))
                anchors = np.arange(start, stop, dtype=np.int64)
                target_indices = np.minimum(
                    anchors[:, None] + offsets[None, :], len(episode_actions) - 1
                )
                padded_states, delta_actions = body_delta_action_chunks(
                    episode_states[anchors],
                    episode_actions[target_indices],
                    model_action_dim,
                )
                stats["state"].update(padded_states)
                stats["actions"].update(delta_actions)
                frame_count += len(anchors)

    expected_frames = int(info["total_frames"])
    if frame_count != expected_frames:
        raise RuntimeError(
            f"processed {frame_count} frames, expected {expected_frames}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(
        output_dir,
        {key: running.get_statistics() for key, running in stats.items()},
    )
    output = output_dir / "norm_stats.json"
    print(f"Wrote {output} from {frame_count} anchors, horizon={action_horizon}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--model-action-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    calculate(
        args.dataset,
        args.output_dir,
        action_horizon=args.action_horizon,
        model_action_dim=args.model_action_dim,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
