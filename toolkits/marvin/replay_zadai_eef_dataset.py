"""Audit and optionally replay the complete Zadai EEF LeRobot dataset.

The default mode is read-only.  It runs every episode through the actual
Stage-2 EEF action path (absolute physical target -> normalized policy action
-> ``Zadai200Env._prepare_physical_action`` -> the environment's wire payload)
and writes JSONL diagnostics plus a summary. ``--send`` is an explicit
hardware operation that submits those exact Stage-2 payloads to VLAHost; it is
never enabled by default.

Examples:
    python toolkits/marvin/replay_zadai_eef_dataset.py audit \
        --dataset /data/zadai_200_eef_rot6d_rlinf_v21 \
        --output /tmp/zadai_eef_audit

    # Hardware replay (only after reviewing the audit):
    python toolkits/marvin/replay_zadai_eef_dataset.py replay \
        --dataset /data/zadai_200_eef_rot6d_rlinf_v21 \
        --output /tmp/zadai_eef_replay --episode 0 --send
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
    policy_to_absolute,
    rot6d_angular_distance,
    quaternion_xyzw_to_rot6d,
    shift_eef_tool_frame,
)
from rlinf.envs.realworld.zadai.zadai_env import Zadai200Env

EEF_DIM = 20
DEFAULT_CHUNK = 30


def _matrix(table: Any, name: str) -> np.ndarray:
    if name not in table.column_names:
        raise KeyError(f"dataset is missing column {name!r}")
    values = table[name].to_pylist()
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != EEF_DIM:
        raise ValueError(f"{name} must have shape [N, 20], got {result.shape}")
    return result


def _episode_files(dataset: Path) -> dict[int, Path]:
    result = {}
    for path in sorted((dataset / "data").rglob("episode_*.parquet")):
        try:
            result[int(path.stem.rsplit("_", 1)[-1])] = path
        except ValueError:
            continue
    if not result:
        raise FileNotFoundError(f"no episode parquet files below {dataset / 'data'}")
    return result


def _state_eef_vector(state: dict[str, Any], side: str) -> np.ndarray | None:
    """Convert one VLAHost state pose to the replayer's 9-D EEF format."""
    pose = state.get(f"eef_{side}")
    if not isinstance(pose, dict):
        return None
    position = pose.get("position")
    orientation = pose.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        return None
    values = np.asarray(
        [*[position.get(axis) for axis in "xyz"],
         *[orientation.get(axis) for axis in "xyzw"]],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        return None
    return np.concatenate([values[:3], quaternion_xyzw_to_rot6d(values[3:])])


def _state_gripper_position(state: dict[str, Any], side: str) -> float:
    value = state.get(f"gripper_{side}")
    if isinstance(value, dict):
        value = value.get("position")
    if isinstance(value, list):
        value = value[0] if value else 0.0
    result = float(value or 0.0)
    if not np.isfinite(result):
        raise ValueError(f"VLAHost gripper_{side} position is not finite")
    return result


def _runtime_policy_reference(
    state: dict[str, Any], tool_offset_m: float
) -> np.ndarray:
    left = _state_eef_vector(state, "left")
    right = _state_eef_vector(state, "right")
    if left is None or right is None:
        raise ValueError("VLAHost /state is missing finite bilateral EEF poses")
    runtime = np.concatenate(
        [
            left,
            [_state_gripper_position(state, "left")],
            right,
            [_state_gripper_position(state, "right")],
        ]
    ).astype(np.float32)
    return shift_eef_tool_frame(runtime, -float(tool_offset_m)).astype(np.float32)


def _eef_error(target: np.ndarray, measured: np.ndarray | None) -> dict[str, float] | None:
    """Return Cartesian and geodesic EEF errors for one measured pose."""
    if measured is None:
        return None
    return {
        "position_error_m": float(np.linalg.norm(target[:3] - measured[:3])),
        "rotation_error_rad": float(
            rot6d_angular_distance(target[3:9], measured[3:9])
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("this tool requires pyarrow") from exc

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "steps.jsonl"
    execution_path = output / "execution.jsonl"
    trace_path = output / "execution_trace.jsonl"
    episodes = _episode_files(dataset)
    selected = episodes if args.episode is None else {args.episode: episodes[args.episode]}
    summary: dict[str, Any] = {
        "dataset": str(dataset), "episodes": 0, "frames": 0,
        "roundtrip_max_abs_error": 0.0, "workspace_violations": 0,
        "step_clipped": 0, "max_xyz_step_m": 0.0, "max_rot_step_rad": 0.0,
        "nonfinite": 0, "action_out_of_normalized_range": 0,
    }
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
    low = processor._physical_action_low
    high = processor._physical_action_high
    summary["stage2_action_processing"] = {
        "implementation": "Zadai200Env._prepare_physical_action",
        "control_space": "eef",
        "step_frequency_hz": float(args.frequency),
        "max_eef_translation_delta_m": float(args.max_step),
        "max_eef_chunk_translation_delta_m": float(args.max_chunk),
        "max_eef_rotation_delta_rad": float(args.max_rotation),
        "max_eef_angular_velocity_rad_s": float(args.max_angular_velocity),
        "max_eef_chunk_rotation_delta_rad": float(args.max_chunk_rotation),
        "eef_policy_to_runtime_tool_offset_m": float(args.tool_offset),
        "initial_reference": (
            "VLAHost /state" if args.mode == "replay" else "observation.state[0]"
        ),
    }
    session = None
    if args.mode == "replay":
        import requests

        session = requests.Session()
    chunks_sent = 0

    def send_chunk(chunk: list[np.ndarray], episode: int, frame_start: int) -> None:
        nonlocal chunks_sent
        if session is None or not chunk:
            return
        payload = [processor._physical_action_payload(value) for value in chunk]
        response = session.post(
            args.url.rstrip("/") + "/action_chunk",
            json={"source_hz": float(args.frequency), "actions": payload},
            timeout=float(args.timeout),
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(f"VLAHost rejected replay chunk: {result}")
        chunks_sent += 1

        # VLAHost replaces an active chunk when a new one arrives.  Do not
        # use len(chunk) / source_hz here: the server may prepend a blend
        # waypoint and interpolate at a different playback rate.  Wait for
        # the server's completion signal so the next chunk cannot preempt the
        # tail of this one.
        completion_deadline = time.monotonic() + max(
            30.0, 4.0 * len(chunk) / float(args.frequency)
        )
        trace_started = time.monotonic()
        trace_sample = 0

        def record_state(
            state: dict[str, Any], phase: str, *, hold_request_sent: bool = False
        ) -> dict[str, Any]:
            nonlocal trace_sample
            elapsed = time.monotonic() - trace_started
            intervention = state.get("intervention")
            if not isinstance(intervention, dict):
                intervention = {}
            policy_step_id = intervention.get("policy_step_id")
            if isinstance(policy_step_id, int) and 0 <= policy_step_id < len(chunk):
                source_index = policy_step_id
                target_source = "server_policy_step_id"
            elif phase == "settle":
                source_index = len(chunk) - 1
                target_source = "chunk_final_target"
            else:
                source_index = min(
                    int(elapsed * float(args.frequency)), len(chunk) - 1
                )
                target_source = "elapsed_time_estimate"
            joint_states = state.get("joint_states")
            if not isinstance(joint_states, dict):
                joint_states = {}
            measured_left = _state_eef_vector(state, "left")
            measured_right = _state_eef_vector(state, "right")
            target_value = chunk[source_index]
            errors = {
                "left": _eef_error(target_value[:9], measured_left),
                "right": _eef_error(target_value[10:19], measured_right),
            }
            record = {
                "episode": episode,
                "frame_start": frame_start,
                "chunk_id": result.get("chunk_id"),
                "trace_sample": trace_sample,
                "phase": phase,
                "hold_request_sent": hold_request_sent,
                "elapsed_s": elapsed,
                "source_hz": float(args.frequency),
                "target_source": target_source,
                "target_step": source_index,
                "server_policy_step_id": policy_step_id,
                "target_eef": {
                    "left": target_value[:9].tolist(),
                    "right": target_value[10:19].tolist(),
                    "gripper_left": float(target_value[9]),
                    "gripper_right": float(target_value[19]),
                },
                "eef_error": errors,
                "server_policy_action_eef": intervention.get("policy_action_eef"),
                "server_executed_action_eef": intervention.get("executed_action_eef"),
                "server_executed_action_joint": intervention.get("executed_action"),
                "measured_eef_left": state.get("eef_left"),
                "measured_eef_right": state.get("eef_right"),
                "measured_joint_positions": joint_states.get("positions"),
                "need_new_chunk": state.get("need_new_chunk"),
                "intervention": intervention,
            }
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            trace_sample += 1
            return record

        final_playback_record = None
        while True:
            if time.monotonic() >= completion_deadline:
                raise TimeoutError(
                    "VLAHost chunk did not finish before timeout; refusing to "
                    "submit the next replay chunk"
                )
            state_response = session.get(
                args.url.rstrip("/") + "/state", timeout=float(args.timeout)
            )
            state_response.raise_for_status()
            state = state_response.json()
            final_playback_record = record_state(state, "playback")
            if int(state.get("need_new_chunk", 0)) == 1:
                break
            time.sleep(1.0 / float(args.trace_frequency))

        settle_deadline = time.monotonic() + float(args.settle_time)
        hold_frequency = float(args.settle_hold_frequency)
        hold_period = 1.0 / hold_frequency if hold_frequency > 0 else None
        next_hold_at = time.monotonic()
        hold_requests = 0
        final_settle_record = final_playback_record
        while time.monotonic() < settle_deadline:
            hold_request_sent = False
            now = time.monotonic()
            if hold_period is not None and now >= next_hold_at:
                hold_response = session.post(
                    args.url.rstrip("/") + "/action",
                    json=payload[-1],
                    timeout=float(args.timeout),
                )
                hold_response.raise_for_status()
                hold_result = hold_response.json()
                if not hold_result.get("success", False):
                    raise RuntimeError(
                        f"VLAHost rejected final EEF hold action: {hold_result}"
                    )
                if hold_result.get("control_space") != "eef":
                    raise RuntimeError(
                        f"VLAHost returned unexpected hold control space: {hold_result}"
                    )
                hold_request_sent = True
                hold_requests += 1
                next_hold_at += hold_period
                if next_hold_at <= now:
                    missed = int((now - next_hold_at) / hold_period) + 1
                    next_hold_at += missed * hold_period
            state_response = session.get(
                args.url.rstrip("/") + "/state", timeout=float(args.timeout)
            )
            state_response.raise_for_status()
            state = state_response.json()
            final_settle_record = record_state(
                state, "settle", hold_request_sent=hold_request_sent
            )
            sleep_for = min(
                1.0 / float(args.trace_frequency),
                max(0.0, settle_deadline - time.monotonic()),
            )
            if hold_period is not None:
                sleep_for = min(
                    sleep_for, max(0.0, next_hold_at - time.monotonic())
                )
            if sleep_for > 0:
                time.sleep(sleep_for)

        measured = None
        try:
            state_response = session.get(
                args.url.rstrip("/") + "/state", timeout=float(args.timeout)
            )
            state_response.raise_for_status()
            state = state_response.json()
            measured = {
                "eef_left": state.get("eef_left"),
                "eef_right": state.get("eef_right"),
                "need_new_chunk": state.get("need_new_chunk"),
                "intervention": state.get("intervention"),
                "joint_positions": state.get("joint_states", {}).get("positions"),
            }
        except Exception as exc:  # keep diagnostics usable if state polling fails
            measured = {"error": repr(exc)}
        with execution_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "episode": episode,
                "frame_start": frame_start,
                "frame_end": frame_start + len(chunk) - 1,
                "chunk_size": len(chunk),
                "target_first": chunk[0].tolist(),
                "target_last": chunk[-1].tolist(),
                "vlahost_response": result,
                "error_at_playback_end": (
                    final_playback_record.get("eef_error")
                    if final_playback_record is not None
                    else None
                ),
                "error_after_settle": (
                    final_settle_record.get("eef_error")
                    if final_settle_record is not None
                    else None
                ),
                "settle_time_s": float(args.settle_time),
                "settle_hold_frequency_hz": hold_frequency,
                "settle_hold_requests": hold_requests,
                "measured_after_chunk": measured,
            }, ensure_ascii=False) + "\n")
    execution_path.write_text("", encoding="utf-8")
    trace_path.write_text("", encoding="utf-8")
    with detail_path.open("w", encoding="utf-8") as log:
        for episode, path in sorted(selected.items()):
            table = pq.read_table(path, columns=["observation.state", "action"])
            states = _matrix(table, "observation.state")
            actions = _matrix(table, "action")
            # Stage 2 receives a measured EEF observation before generating its
            # first action chunk. Hardware replay therefore anchors to the live
            # VLAHost state; read-only audit uses the matching recorded state.
            # Subsequent chunks carry the last commanded target exactly as the
            # environment does.
            if session is None:
                previous = states[0].copy()
            else:
                state_response = session.get(
                    args.url.rstrip("/") + "/state", timeout=float(args.timeout)
                )
                state_response.raise_for_status()
                previous = _runtime_policy_reference(
                    state_response.json(), args.tool_offset
                )
            chunk_ref = previous.copy()
            episode_clipped = 0
            sent_chunk: list[np.ndarray] = []
            stopped_early = False
            for frame, (state, action) in enumerate(zip(states, actions, strict=True)):
                if (
                    args.mode == "replay"
                    and args.max_chunks is not None
                    and chunks_sent >= args.max_chunks
                ):
                    stopped_early = True
                    break
                summary["frames"] += 1
                if not np.isfinite(action).all() or not np.isfinite(state).all():
                    summary["nonfinite"] += 1
                normalized = np.asarray(
                    absolute_to_policy(action, "eef"), dtype=np.float32
                )
                if np.any((normalized < -1.00001) | (normalized > 1.00001)):
                    summary["action_out_of_normalized_range"] += 1
                roundtrip = np.asarray(
                    policy_to_absolute(normalized, "eef"), dtype=np.float32
                )
                summary["roundtrip_max_abs_error"] = max(
                    summary["roundtrip_max_abs_error"], float(np.max(np.abs(roundtrip - action)))
                )
                if frame % args.chunk_size == 0:
                    chunk_ref = previous.copy()
                prepared = processor._prepare_physical_action(
                    normalized, previous, chunk_reference=chunk_ref
                )
                projected = prepared["physical_action"]
                clipped = bool(prepared["safety_clipped"])
                workspace_bad = bool(np.any(action < low - 1e-5) or np.any(action > high + 1e-5))
                summary["workspace_violations"] += int(workspace_bad)
                summary["step_clipped"] += int(clipped)
                episode_clipped += int(clipped)
                if previous is not None:
                    xyz_step = max(
                        float(np.linalg.norm(action[o : o + 3] - previous[o : o + 3]))
                        for o in (0, 10)
                    )
                    rot_step = max(
                        rot6d_angular_distance(action[o + 3 : o + 9], previous[o + 3 : o + 9])
                        for o in (0, 10)
                    )
                    summary["max_xyz_step_m"] = max(summary["max_xyz_step_m"], xyz_step)
                    summary["max_rot_step_rad"] = max(summary["max_rot_step_rad"], rot_step)
                log.write(json.dumps({
                    "episode": episode, "frame": frame,
                    "state": state.tolist(), "action_physical": action.tolist(),
                    "action_normalized": normalized.tolist(),
                    "action_roundtrip": roundtrip.tolist(),
                    "action_projected": projected.tolist(),
                    "stage2_safety_delta": prepared["safety_delta"].tolist(),
                    "wire_payload": processor._physical_action_payload(projected),
                    "clipped": clipped,
                    "workspace_violation": workspace_bad,
                }, ensure_ascii=False) + "\n")
                previous = projected
                sent_chunk.append(projected)
                if len(sent_chunk) >= args.chunk_size:
                    send_chunk(sent_chunk, episode, frame - len(sent_chunk) + 1)
                    sent_chunk = []
            if sent_chunk and not stopped_early:
                send_chunk(sent_chunk, episode, len(actions) - len(sent_chunk))
            summary["episodes"] += 1
            log.write(json.dumps({"episode_end": episode, "episode_clipped": episode_clipped}) + "\n")
    summary["chunks_sent"] = chunks_sent
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    processor.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "replay"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--max-step", type=float, default=0.05)
    parser.add_argument("--max-chunk", type=float, default=0.40)
    parser.add_argument("--max-rotation", type=float, default=0.10)
    parser.add_argument("--max-angular-velocity", type=float, default=3.0)
    parser.add_argument("--max-chunk-rotation", type=float, default=1.50)
    parser.add_argument(
        "--tool-offset",
        type=float,
        default=0.0,
        help="policy-to-runtime local +X tool-frame offset; corrected m6_696 data uses 0.0",
    )
    parser.add_argument("--send", action="store_true", help="send projected chunks to VLAHost (hardware operation)")
    parser.add_argument("--confirm", action="store_true", help="required together with --send")
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.0,
        help="seconds to wait and trace after each chunk finishes",
    )
    parser.add_argument(
        "--trace-frequency",
        type=float,
        default=100.0,
        help="state sampling frequency during playback and settle periods",
    )
    parser.add_argument(
        "--settle-hold-frequency",
        type=float,
        default=0.0,
        help=(
            "repeat the final absolute EEF target at this frequency during "
            "the settle period; zero disables active hold"
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="stop hardware replay after this many chunks",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.frequency <= 0:
        parser.error("--frequency must be positive")
    if args.settle_time < 0:
        parser.error("--settle-time must be non-negative")
    if args.trace_frequency <= 0:
        parser.error("--trace-frequency must be positive")
    if args.settle_hold_frequency < 0:
        parser.error("--settle-hold-frequency must be non-negative")
    if args.settle_hold_frequency > 0 and args.settle_time <= 0:
        parser.error("--settle-hold-frequency requires positive --settle-time")
    if args.max_chunks is not None and args.max_chunks <= 0:
        parser.error("--max-chunks must be positive")
    if args.mode == "replay" and not (args.send and args.confirm):
        raise SystemExit("replay mode requires both --send and --confirm; use audit for read-only diagnostics")
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
