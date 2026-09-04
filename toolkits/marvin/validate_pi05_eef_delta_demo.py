"""Validate the Pi0.5 EEF-delta deployment path using one recorded demo chunk.

The tool does not ask Pi0.5 to predict an action.  Instead it uses the demo's
absolute future actions as a known-good Pi0.5 output, applies the exact
OpenPI ``RigidBodyDeltaActions`` transform used at training time, normalizes
and unnormalizes those deltas with the Pi0.5 action statistics, and decodes
them with ``RigidBodyAbsoluteActions``.  The recovered absolute EEF targets
then pass through the same Stage-2 Zadai environment action codec, safety
projection, and HTTP payload conversion used at deployment.

This isolates action-representation / control-path correctness from model
prediction quality.  ``--send --confirm`` submits the first 30 recovered
targets to VLAHost and records measured EEF tracking errors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.envs.realworld.zadai.action_codec import (
    absolute_to_policy,
    quaternion_xyzw_to_rot6d,
    rot6d_angular_distance,
    shift_eef_tool_frame,
)
from rlinf.envs.realworld.zadai.zadai_env import Zadai200Env
from rlinf.models.embodiment.openpi.transforms import (
    DUAL_ARM_ROT6D_LAYOUT,
    RigidBodyAbsoluteActions,
    RigidBodyDeltaActions,
)

EEF_DIM = 20


def _episode_path(dataset: Path, episode: int) -> Path:
    matches = sorted((dataset / "data").rglob(f"episode_{episode:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one parquet for episode {episode}, found {len(matches)}"
        )
    return matches[0]


def _matrix(table: Any, name: str) -> np.ndarray:
    if name not in table.column_names:
        raise KeyError(f"dataset is missing {name!r}")
    value = np.asarray(table[name].to_pylist(), dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != EEF_DIM or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite [N, {EEF_DIM}], got {value.shape}")
    return value


def _load_action_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    stats = raw.get("norm_stats", raw).get("actions")
    if not isinstance(stats, dict):
        raise KeyError("norm stats must contain norm_stats.actions")
    mean = np.asarray(stats["mean"], dtype=np.float32)[:EEF_DIM]
    std = np.asarray(stats["std"], dtype=np.float32)[:EEF_DIM]
    if mean.shape != (EEF_DIM,) or std.shape != (EEF_DIM,):
        raise ValueError("action norm stats must contain at least 20 values")
    return mean, std


def _normalization_roundtrip(
    anchor_state: np.ndarray, absolute_actions: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact transform ordering: abs -> body delta -> z-score -> abs."""
    delta = RigidBodyDeltaActions(DUAL_ARM_ROT6D_LAYOUT)(
        {"state": anchor_state.copy(), "actions": absolute_actions.copy()}
    )["actions"]
    normalized_delta = (delta - mean) / (std + 1e-6)
    restored_delta = normalized_delta * (std + 1e-6) + mean
    restored_absolute = RigidBodyAbsoluteActions(DUAL_ARM_ROT6D_LAYOUT)(
        {"state": anchor_state.copy(), "actions": restored_delta.copy()}
    )["actions"]
    return delta.astype(np.float32), normalized_delta.astype(np.float32), restored_absolute.astype(np.float32)


def _pose_errors(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    return {
        "left_position_m": float(np.linalg.norm(expected[:3] - actual[:3])),
        "right_position_m": float(np.linalg.norm(expected[10:13] - actual[10:13])),
        "left_rotation_rad": float(rot6d_angular_distance(expected[3:9], actual[3:9])),
        "right_rotation_rad": float(rot6d_angular_distance(expected[13:19], actual[13:19])),
        "left_gripper": float(abs(expected[9] - actual[9])),
        "right_gripper": float(abs(expected[19] - actual[19])),
    }


def _state_pose(state: dict[str, Any], side: str) -> np.ndarray:
    pose = state.get(f"eef_{side}")
    if not isinstance(pose, dict):
        raise ValueError(f"VLAHost state lacks eef_{side}")
    position, orientation = pose.get("position"), pose.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        raise ValueError(f"VLAHost eef_{side} is malformed")
    values = np.asarray(
        [*[position.get(axis) for axis in "xyz"], *[orientation.get(axis) for axis in "xyzw"]],
        dtype=np.float32,
    )
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError(f"VLAHost eef_{side} is not finite")
    return np.concatenate([values[:3], quaternion_xyzw_to_rot6d(values[3:])])


def _gripper(state: dict[str, Any], side: str) -> float:
    value = state.get(f"gripper_{side}")
    if isinstance(value, dict):
        value = value.get("position")
    if isinstance(value, list):
        value = value[0] if value else 0.0
    value = float(value or 0.0)
    if not np.isfinite(value):
        raise ValueError(f"VLAHost gripper_{side} is not finite")
    return value


def _live_policy_state(state: dict[str, Any], tool_offset_m: float) -> np.ndarray:
    runtime = np.concatenate(
        [_state_pose(state, "left"), [_gripper(state, "left")], _state_pose(state, "right"), [_gripper(state, "right")]]
    ).astype(np.float32)
    return shift_eef_tool_frame(runtime, -tool_offset_m).astype(np.float32)


def _max_error(records: list[dict[str, float]], key: str) -> float | None:
    values = [item[key] for item in records if item.get(key) is not None]
    return float(max(values)) if values else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    import pyarrow.parquet as pq

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = _episode_path(dataset, args.episode)
    table = pq.read_table(path, columns=["observation.state", "action"])
    states = _matrix(table, "observation.state")
    actions = _matrix(table, "action")
    if args.frame < 0 or args.frame >= len(actions):
        raise ValueError(f"--frame must be in [0, {len(actions) - 1}]")
    available = len(actions) - args.frame
    if args.horizon > available:
        raise ValueError(f"only {available} future actions remain at frame {args.frame}")
    if args.execute_steps > args.horizon:
        raise ValueError("--execute-steps cannot exceed --horizon")

    anchor = states[args.frame]
    absolute_chunk = actions[args.frame : args.frame + args.horizon]
    mean, std = _load_action_stats(args.norm_stats.resolve())
    delta, normalized_delta, recovered = _normalization_roundtrip(
        anchor, absolute_chunk, mean, std
    )
    recovery_errors = [_pose_errors(src, dst) for src, dst in zip(absolute_chunk, recovered, strict=True)]
    normalization_error = float(np.max(np.abs(absolute_chunk - recovered)))

    processor = Zadai200Env(
        {
            "control_space": "eef",
            "is_dummy": True,
            "action_transport": "chunk",
            "step_frequency": float(args.frequency),
            "enforce_action_delta_limit": True,
            "enforce_chunk_action_delta_limit": True,
            "max_eef_translation_delta": float(args.max_step),
            "max_eef_chunk_translation_delta": float(args.max_chunk),
            "max_eef_rotation_delta_rad": float(args.max_rotation),
            "max_eef_angular_velocity_rad_s": float(args.max_angular_velocity),
            "max_eef_chunk_rotation_delta_rad": float(args.max_chunk_rotation),
            "eef_policy_to_runtime_tool_offset_m": float(args.tool_offset),
        }
    )
    try:
        reference = anchor.copy()
        live_initial_error = None
        session = None
        if args.send:
            import requests

            session = requests.Session()
            response = session.get(args.url.rstrip("/") + "/state", timeout=args.timeout)
            response.raise_for_status()
            reference = _live_policy_state(response.json(), args.tool_offset)
            live_initial_error = _pose_errors(anchor, reference)
            if not args.allow_initial_mismatch and (
                max(live_initial_error["left_position_m"], live_initial_error["right_position_m"])
                > args.initial_position_tolerance
                or max(live_initial_error["left_rotation_rad"], live_initial_error["right_rotation_rad"])
                > args.initial_rotation_tolerance
            ):
                raise RuntimeError(
                    "live robot pose does not match the demo anchor; refusing to send. "
                    "Move/align to the demo start or use --allow-initial-mismatch explicitly. "
                    f"errors={live_initial_error}"
                )

        chunk_reference = reference.copy()
        prepared: list[dict[str, Any]] = []
        for absolute in recovered[: args.execute_steps]:
            normalized = np.asarray(absolute_to_policy(absolute, "eef"), dtype=np.float32)
            step = processor._prepare_physical_action(
                normalized, reference, chunk_reference=chunk_reference
            )
            prepared.append(step)
            reference = step["physical_action"]

        trace: list[dict[str, Any]] = []
        sent = False
        if session is not None:
            payload = [processor._physical_action_payload(step["physical_action"]) for step in prepared]
            response = session.post(
                args.url.rstrip("/") + "/action_chunk",
                json={"source_hz": float(args.frequency), "actions": payload},
                timeout=args.timeout,
            )
            response.raise_for_status()
            result = response.json()
            if not result.get("success", False):
                raise RuntimeError(f"VLAHost rejected chunk: {result}")
            sent = True
            deadline = time.monotonic() + max(30.0, 4.0 * len(prepared) / args.frequency)
            while True:
                response = session.get(args.url.rstrip("/") + "/state", timeout=args.timeout)
                response.raise_for_status()
                state = response.json()
                intervention = state.get("intervention") if isinstance(state.get("intervention"), dict) else {}
                step_id = intervention.get("policy_step_id")
                target_index = step_id if isinstance(step_id, int) and 0 <= step_id < len(prepared) else len(prepared) - 1
                measured = _live_policy_state(state, args.tool_offset)
                trace.append(
                    {
                        "elapsed_s": time.monotonic() - (deadline - max(30.0, 4.0 * len(prepared) / args.frequency)),
                        "policy_step_id": step_id,
                        "target_index": target_index,
                        "tracking_error": _pose_errors(prepared[target_index]["physical_action"], measured),
                        "need_new_chunk": state.get("need_new_chunk"),
                    }
                )
                if int(state.get("need_new_chunk", 0)) == 1:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("VLAHost did not complete the validation chunk")
                time.sleep(1.0 / args.trace_frequency)

        steps = []
        for index, (source, delta_value, normalized_value, restored, step) in enumerate(
            zip(
                absolute_chunk[: args.execute_steps],
                delta[: args.execute_steps],
                normalized_delta[: args.execute_steps],
                recovered[: args.execute_steps],
                prepared,
                strict=True,
            )
        ):
            steps.append(
                {
                    "index": index,
                    "absolute_demo": source.tolist(),
                    "pi05_body_delta": delta_value.tolist(),
                    "pi05_normalized_delta": normalized_value.tolist(),
                    "pi05_decoded_absolute": restored.tolist(),
                    "stage2_physical": step["physical_action"].tolist(),
                    "stage2_safety_delta": step["safety_delta"].tolist(),
                    "stage2_safety_clipped": step["safety_clipped"],
                    "vlahost_payload": processor._physical_action_payload(step["physical_action"]),
                }
            )
        (output / "steps.json").write_text(json.dumps(steps, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output / "tracking.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary = {
            "dataset": str(dataset),
            "episode": args.episode,
            "anchor_frame": args.frame,
            "pi05_action_horizon": args.horizon,
            "stage2_steps_executed": args.execute_steps,
            "stage1_transform": "absolute EEF -> body-frame SE(3) delta -> z-score -> inverse z-score -> absolute EEF",
            "norm_stats": str(args.norm_stats.resolve()),
            "max_absolute_recovery_error": normalization_error,
            "max_recovery_left_position_m": _max_error(recovery_errors, "left_position_m"),
            "max_recovery_right_position_m": _max_error(recovery_errors, "right_position_m"),
            "max_recovery_left_rotation_rad": _max_error(recovery_errors, "left_rotation_rad"),
            "max_recovery_right_rotation_rad": _max_error(recovery_errors, "right_rotation_rad"),
            "stage2_safety_clipped_steps": int(sum(step["safety_clipped"] for step in prepared)),
            "tool_offset_m": args.tool_offset,
            "sent": sent,
            "live_initial_error": live_initial_error,
            "max_live_left_position_error_m": _max_error([item["tracking_error"] for item in trace], "left_position_m"),
            "max_live_right_position_error_m": _max_error([item["tracking_error"] for item in trace], "right_position_m"),
            "max_live_left_rotation_error_rad": _max_error([item["tracking_error"] for item in trace], "left_rotation_rad"),
            "max_live_right_rotation_error_rad": _max_error([item["tracking_error"] for item in trace], "right_rotation_rad"),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return summary
    finally:
        processor.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--execute-steps", type=int, default=30)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--max-step", type=float, default=0.05)
    parser.add_argument("--max-chunk", type=float, default=0.40)
    parser.add_argument("--max-rotation", type=float, default=0.20)
    parser.add_argument("--max-angular-velocity", type=float, default=3.0)
    parser.add_argument("--max-chunk-rotation", type=float, default=1.50)
    parser.add_argument("--tool-offset", type=float, default=0.0)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--trace-frequency", type=float, default=100.0)
    parser.add_argument("--initial-position-tolerance", type=float, default=0.03)
    parser.add_argument("--initial-rotation-tolerance", type=float, default=0.20)
    parser.add_argument("--allow-initial-mismatch", action="store_true")
    args = parser.parse_args()
    if args.send and not args.confirm:
        parser.error("--send requires --confirm")
    if args.horizon <= 0 or args.execute_steps <= 0:
        parser.error("--horizon and --execute-steps must be positive")
    if args.frequency <= 0 or args.trace_frequency <= 0:
        parser.error("frequencies must be positive")
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
