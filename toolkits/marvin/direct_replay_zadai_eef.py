# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Replay recorded Zadai EEF poses without the RLinf action pipeline.

This diagnostic reads physical 20-D EEF values directly from a LeRobot
episode, converts only the rot6d representation to an xyzw quaternion, and
sends one absolute waypoint at a time to VLAHost.  It deliberately does not
normalize, denormalize, accumulate deltas, project workspaces, or form policy
chunks.

By default ``observation.state`` is replayed because it is the EEF trajectory
that the robot actually reached during data collection.  Use ``--source
action`` to inspect the recorded controller targets instead.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.envs.realworld.zadai.action_codec import (
    quaternion_xyzw_to_rot6d,
    rot6d_angular_distance,
    rot6d_to_quaternion_xyzw,
    shift_eef_tool_frame,
)

EEF_DIM = 20


def _episode_files(dataset: Path) -> dict[int, Path]:
    episodes: dict[int, Path] = {}
    for path in sorted((dataset / "data").rglob("episode_*.parquet")):
        try:
            episodes[int(path.stem.rsplit("_", 1)[-1])] = path
        except ValueError:
            continue
    if not episodes:
        raise FileNotFoundError(f"no episode parquet files below {dataset / 'data'}")
    return episodes


def _matrix(table: Any, column: str) -> np.ndarray:
    if column not in table.column_names:
        raise KeyError(f"dataset is missing column {column!r}")
    result = np.asarray(table[column].to_pylist(), dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != EEF_DIM:
        raise ValueError(f"{column} must have shape [N, 20], got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{column} contains non-finite values")
    return result


def _payload(value: np.ndarray) -> dict[str, Any]:
    return {
        "eef_left": np.concatenate(
            [value[:3], rot6d_to_quaternion_xyzw(value[3:9])]
        ).tolist(),
        "eef_right": np.concatenate(
            [value[10:13], rot6d_to_quaternion_xyzw(value[13:19])]
        ).tolist(),
        "gripper_left": float(value[9]),
        "gripper_right": float(value[19]),
    }


def _measured_eef(state: dict[str, Any], side: str) -> np.ndarray | None:
    pose = state.get(f"eef_{side}")
    if not isinstance(pose, dict):
        return None
    position = pose.get("position")
    orientation = pose.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        return None
    xyz = [position.get(axis) for axis in "xyz"]
    quat = [orientation.get(axis) for axis in "xyzw"]
    values = np.asarray(xyz + quat, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        return None
    return np.concatenate([values[:3], quaternion_xyzw_to_rot6d(values[3:])])


def _eef_error(
    target: np.ndarray, measured: np.ndarray | None
) -> dict[str, float] | None:
    if measured is None:
        return None
    return {
        "position_error_m": float(np.linalg.norm(target[:3] - measured[:3])),
        "rotation_error_rad": float(rot6d_angular_distance(target[3:9], measured[3:9])),
    }


def _target_errors(value: np.ndarray, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "left": _eef_error(value[:9], _measured_eef(state, "left")),
        "right": _eef_error(value[10:19], _measured_eef(state, "right")),
    }


def _max_source_steps(values: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for side, offset in (("left", 0), ("right", 10)):
        xyz_steps = np.linalg.norm(
            np.diff(values[:, offset : offset + 3], axis=0), axis=1
        )
        rotation_steps = [
            rot6d_angular_distance(previous, current)
            for previous, current in zip(
                values[:-1, offset + 3 : offset + 9],
                values[1:, offset + 3 : offset + 9],
                strict=True,
            )
        ]
        result[f"max_{side}_position_step_m"] = float(np.max(xyz_steps, initial=0.0))
        result[f"max_{side}_rotation_step_rad"] = float(
            max(rotation_steps, default=0.0)
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
        import requests
    except ImportError as exc:
        raise RuntimeError("this tool requires pyarrow and requests") from exc

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    episode_path = _episode_files(dataset)[args.episode]
    source_column = "observation.state" if args.source == "state" else "action"
    joint_column = (
        "observation.joint_position"
        if args.source == "state"
        else "action.joint_position_source"
    )
    available_columns = set(pq.read_schema(episode_path).names)
    columns = [source_column]
    if joint_column in available_columns:
        columns.append(joint_column)
    table = pq.read_table(episode_path, columns=columns)
    all_values = _matrix(table, source_column)
    expected_joints = None
    if joint_column in table.column_names:
        expected_joints = np.asarray(table[joint_column].to_pylist(), dtype=np.float64)
        if expected_joints.shape != (len(all_values), 14):
            raise ValueError(
                f"{joint_column} must have shape [N, 14], got {expected_joints.shape}"
            )
    stop = (
        len(all_values)
        if args.stop_frame is None
        else min(args.stop_frame, len(all_values))
    )
    policy_values = all_values[args.start_frame : stop]
    values = shift_eef_tool_frame(
        policy_values, args.eef_policy_to_runtime_tool_offset_m
    ).astype(np.float32)
    if expected_joints is not None:
        expected_joints = expected_joints[args.start_frame : stop]
    values = values[args.start_frame : stop]
    if not len(values):
        raise ValueError("selected frame range is empty")

    sends_path = output / "sends.jsonl"
    sends_path.write_text("", encoding="utf-8")
    source_steps = _max_source_steps(values)
    summary: dict[str, Any] = {
        "dataset": str(dataset),
        "episode": args.episode,
        "source": source_column,
        "start_frame": args.start_frame,
        "stop_frame_exclusive": stop,
        "frames_selected": len(values),
        "requested_frequency_hz": float(args.frequency),
        "eef_policy_to_runtime_tool_offset_m": float(
            args.eef_policy_to_runtime_tool_offset_m
        ),
        **source_steps,
        "trajectory_frames_sent": 0,
        "alignment_requests": 0,
        "final_hold_requests": 0,
        "server_clamped": 0,
        "deadline_misses": 0,
        "max_deadline_lag_s": 0.0,
        "request_latency_s": [],
        "measured_joint_error_max_rad": [],
        "commanded_joint_error_max_rad": [],
    }

    session = requests.Session()
    base_url = args.url.rstrip("/")
    period = 1.0 / float(args.frequency)
    request_sequence = 0
    trajectory_send_times: list[float] = []

    def send_and_record(
        value: np.ndarray,
        frame: int,
        phase: str,
        scheduled_at: float,
        expected_joint: np.ndarray | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal request_sequence
        request_sequence += 1
        now = time.monotonic()
        if now < scheduled_at:
            time.sleep(scheduled_at - now)
        started = time.monotonic()
        lag = max(0.0, started - scheduled_at)
        payload = _payload(value)
        response = session.post(
            base_url + "/action_chunk",
            json={"source_hz": float(args.frequency), "actions": [payload]},
            timeout=float(args.timeout),
        )
        response.raise_for_status()
        result = response.json()
        finished = time.monotonic()
        if not result.get("success", False):
            raise RuntimeError(f"VLAHost rejected frame {frame}: {result}")
        state_response = session.get(base_url + "/state", timeout=float(args.timeout))
        state_response.raise_for_status()
        state = state_response.json()
        n_clamped = int(result.get("n_clamped", 0))
        summary["server_clamped"] += n_clamped
        summary["deadline_misses"] += int(lag > period)
        summary["max_deadline_lag_s"] = max(summary["max_deadline_lag_s"], lag)
        summary["request_latency_s"].append(finished - started)
        measured_joint = state.get("joint_states", {}).get("positions")
        intervention = state.get("intervention")
        if not isinstance(intervention, dict):
            intervention = {}
        commanded_joint = intervention.get("executed_action")
        measured_joint_error = None
        commanded_joint_error = None
        if expected_joint is not None and measured_joint is not None:
            measured_array = np.asarray(measured_joint, dtype=np.float64)
            if measured_array.shape == (14,):
                measured_joint_error = (measured_array - expected_joint).tolist()
        if expected_joint is not None and commanded_joint is not None:
            commanded_array = np.asarray(commanded_joint[:14], dtype=np.float64)
            if commanded_array.shape == (14,):
                commanded_joint_error = (commanded_array - expected_joint).tolist()
        if phase == "trajectory":
            if measured_joint_error is not None:
                summary["measured_joint_error_max_rad"].append(
                    float(np.max(np.abs(measured_joint_error)))
                )
            if commanded_joint_error is not None:
                summary["commanded_joint_error_max_rad"].append(
                    float(np.max(np.abs(commanded_joint_error)))
                )
        record = {
            "request_sequence": request_sequence,
            "phase": phase,
            "episode": args.episode,
            "frame": frame,
            "scheduled_monotonic_s": scheduled_at,
            "actual_monotonic_s": started,
            "deadline_lag_s": lag,
            "request_latency_s": finished - started,
            "target_eef_rot6d": value.tolist(),
            "policy_frame_eef_rot6d": shift_eef_tool_frame(
                value, -args.eef_policy_to_runtime_tool_offset_m
            ).tolist(),
            "recorded_joint_reference": (
                expected_joint.tolist() if expected_joint is not None else None
            ),
            "measured_minus_recorded_joint_rad": measured_joint_error,
            "commanded_minus_recorded_joint_rad": commanded_joint_error,
            "http_payload": payload,
            "vlahost_response": result,
            "eef_error": _target_errors(value, state),
            "measured_eef_left": state.get("eef_left"),
            "measured_eef_right": state.get("eef_right"),
            "measured_joint_positions": measured_joint,
            "server_intervention": intervention,
            "need_new_chunk": state.get("need_new_chunk"),
        }
        with sends_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if n_clamped and phase == "trajectory" and not args.allow_clamp:
            raise RuntimeError(
                f"VLAHost clamped frame {frame}; direct replay stopped. "
                "Inspect sends.jsonl or pass --allow-clamp to continue."
            )
        return result, state

    # Repeatedly command frame zero before starting the clock. This is an
    # explicit alignment phase and is kept separate from trajectory metrics.
    align_started = time.monotonic()
    align_deadline = align_started + float(args.align_timeout)
    align_next = align_started
    while True:
        first_joint = expected_joints[0] if expected_joints is not None else None
        _, state = send_and_record(
            values[0], args.start_frame, "alignment", align_next, first_joint
        )
        summary["alignment_requests"] += 1
        errors = _target_errors(values[0], state)
        reached = all(
            error is not None
            and error["position_error_m"] <= args.align_position_tolerance
            and error["rotation_error_rad"] <= args.align_rotation_tolerance
            for error in errors.values()
        )
        if reached:
            break
        if time.monotonic() >= align_deadline:
            raise TimeoutError(
                "initial EEF alignment did not reach tolerance; inspect sends.jsonl"
            )
        align_next += period

    trajectory_started = time.monotonic() + period
    for index, value in enumerate(values):
        frame = args.start_frame + index
        scheduled = trajectory_started + index * period
        expected_joint = expected_joints[index] if expected_joints is not None else None
        send_and_record(value, frame, "trajectory", scheduled, expected_joint)
        trajectory_send_times.append(time.monotonic())
        summary["trajectory_frames_sent"] += 1

    hold_started = time.monotonic() + period
    hold_count = int(math.ceil(float(args.final_hold) * args.frequency))
    for index in range(hold_count):
        scheduled = hold_started + index * period
        final_joint = expected_joints[-1] if expected_joints is not None else None
        send_and_record(values[-1], stop - 1, "final_hold", scheduled, final_joint)
        summary["final_hold_requests"] += 1

    latencies = np.asarray(summary.pop("request_latency_s"), dtype=np.float64)
    summary["request_latency_mean_s"] = float(np.mean(latencies))
    summary["request_latency_p95_s"] = float(np.percentile(latencies, 95))
    summary["request_latency_max_s"] = float(np.max(latencies))
    for key in (
        "measured_joint_error_max_rad",
        "commanded_joint_error_max_rad",
    ):
        joint_errors = np.asarray(summary.pop(key), dtype=np.float64)
        prefix = key.removesuffix("_rad")
        summary[f"{prefix}_mean_rad"] = (
            float(np.mean(joint_errors)) if len(joint_errors) else None
        )
        summary[f"{prefix}_p95_rad"] = (
            float(np.percentile(joint_errors, 95)) if len(joint_errors) else None
        )
        summary[f"{prefix}_max_rad"] = (
            float(np.max(joint_errors)) if len(joint_errors) else None
        )
    if len(trajectory_send_times) > 1:
        send_intervals = np.diff(trajectory_send_times)
        summary["effective_frequency_hz"] = float(1.0 / np.mean(send_intervals))
        summary["send_interval_p95_s"] = float(np.percentile(send_intervals, 95))
        summary["send_interval_max_s"] = float(np.max(send_intervals))
    else:
        summary["effective_frequency_hz"] = None
        summary["send_interval_p95_s"] = None
        summary["send_interval_max_s"] = None
    elapsed = time.monotonic() - trajectory_started
    summary["trajectory_and_hold_elapsed_s"] = elapsed
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--source", choices=("state", "action"), default="state")
    parser.add_argument("--frequency", type=float, default=20.0)
    parser.add_argument(
        "--eef-policy-to-runtime-tool-offset-m",
        type=float,
        default=0.0,
        help=(
            "shift policy-frame EEF targets along local +X before sending; "
            "use 0.095 for legacy 50 mm data on the M6-696 145 mm runtime"
        ),
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--align-timeout", type=float, default=15.0)
    parser.add_argument("--align-position-tolerance", type=float, default=0.015)
    parser.add_argument("--align-rotation-tolerance", type=float, default=0.12)
    parser.add_argument("--final-hold", type=float, default=1.0)
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--allow-clamp",
        action="store_true",
        help="continue if VLAHost changes a trajectory target",
    )
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not (args.send and args.confirm):
        parser.error("hardware replay requires both --send and --confirm")
    if args.frequency <= 0:
        parser.error("--frequency must be positive")
    if not np.isfinite(args.eef_policy_to_runtime_tool_offset_m):
        parser.error("--eef-policy-to-runtime-tool-offset-m must be finite")
    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative")
    if args.stop_frame is not None and args.stop_frame <= args.start_frame:
        parser.error("--stop-frame must be greater than --start-frame")
    if args.align_timeout <= 0:
        parser.error("--align-timeout must be positive")
    if args.align_position_tolerance <= 0 or args.align_rotation_tolerance <= 0:
        parser.error("alignment tolerances must be positive")
    if args.final_hold < 0:
        parser.error("--final-hold must be non-negative")
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
