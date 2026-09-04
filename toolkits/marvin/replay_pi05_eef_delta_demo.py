"""Replay a complete EEF demo through Pi0.5's body-frame delta action path.

At every 30-step deployment boundary this tool computes the recorded Pi0.5
body-frame delta chunk, applies Pi0.5 action normalization and its inverse,
then (by default) decodes against the matching recorded state. This recovers
the original absolute demo trajectory while exercising the Pi0.5 delta codec.
The optional live-state decode mode is intentionally experimental because a
tracking error can move later targets away from the recorded safety-checked
trajectory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.envs.realworld.zadai.action_codec import absolute_to_policy
from rlinf.envs.realworld.zadai.zadai_env import Zadai200Env
from rlinf.models.embodiment.openpi.transforms import (
    DUAL_ARM_ROT6D_LAYOUT,
    RigidBodyAbsoluteActions,
    RigidBodyDeltaActions,
)
from validate_pi05_eef_delta_demo import (
    EEF_DIM,
    _episode_path,
    _live_policy_state,
    _load_action_stats,
    _matrix,
    _pose_errors,
)


def _delta_chunk(
    recorded_state: np.ndarray,
    recorded_actions: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Pi0.5 normalized delta and its exact unnormalized delta."""
    delta = RigidBodyDeltaActions(DUAL_ARM_ROT6D_LAYOUT)(
        {"state": recorded_state.copy(), "actions": recorded_actions.copy()}
    )["actions"]
    normalized = (delta - mean) / (std + 1e-6)
    decoded = normalized * (std + 1e-6) + mean
    return normalized.astype(np.float32), decoded.astype(np.float32)


def _decode_live(live_state: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return RigidBodyAbsoluteActions(DUAL_ARM_ROT6D_LAYOUT)(
        {"state": live_state.copy(), "actions": delta.copy()}
    )["actions"].astype(np.float32)


def _wait_for_chunk(
    session: Any,
    url: str,
    timeout: float,
    trace_frequency: float,
    frequency: float,
    targets: list[dict[str, Any]],
    tool_offset: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    deadline = time.monotonic() + max(30.0, 4.0 * len(targets) / frequency)
    trace: list[dict[str, Any]] = []
    last_live: np.ndarray | None = None
    started = time.monotonic()
    while True:
        response = session.get(url.rstrip("/") + "/state", timeout=timeout)
        response.raise_for_status()
        state = response.json()
        live = _live_policy_state(state, tool_offset)
        last_live = live
        intervention = state.get("intervention")
        intervention = intervention if isinstance(intervention, dict) else {}
        step_id = intervention.get("policy_step_id")
        index = step_id if isinstance(step_id, int) and 0 <= step_id < len(targets) else len(targets) - 1
        trace.append(
            {
                "elapsed_s": time.monotonic() - started,
                "policy_step_id": step_id,
                "target_index": index,
                "tracking_error": _pose_errors(targets[index]["physical_action"], live),
                "need_new_chunk": state.get("need_new_chunk"),
            }
        )
        if int(state.get("need_new_chunk", 0)) == 1:
            return trace, live
        if time.monotonic() >= deadline:
            raise TimeoutError("VLAHost did not complete the Pi0.5 delta chunk")
        time.sleep(1.0 / trace_frequency)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import pyarrow.parquet as pq
    import requests

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(_episode_path(dataset, args.episode), columns=["observation.state", "action"])
    states = _matrix(table, "observation.state")
    actions = _matrix(table, "action")
    if args.start_frame < 0 or args.start_frame >= len(actions):
        raise ValueError(f"--start-frame must be in [0, {len(actions) - 1}]")
    mean, std = _load_action_stats(args.norm_stats.resolve())
    processor = Zadai200Env(
        {
            "control_space": "eef", "is_dummy": True, "action_transport": "chunk",
            "step_frequency": float(args.frequency),
            "enforce_action_delta_limit": True, "enforce_chunk_action_delta_limit": True,
            "max_eef_translation_delta": float(args.max_step),
            "max_eef_chunk_translation_delta": float(args.max_chunk),
            "max_eef_rotation_delta_rad": float(args.max_rotation),
            "max_eef_angular_velocity_rad_s": float(args.max_angular_velocity),
            "max_eef_chunk_rotation_delta_rad": float(args.max_chunk_rotation),
            "eef_policy_to_runtime_tool_offset_m": float(args.tool_offset),
        }
    )
    try:
        session = requests.Session()
        response = session.get(args.url.rstrip("/") + "/state", timeout=args.timeout)
        response.raise_for_status()
        live_state = _live_policy_state(response.json(), args.tool_offset)
        initial_error = _pose_errors(states[args.start_frame], live_state)
        if not args.allow_initial_mismatch and (
            max(initial_error["left_position_m"], initial_error["right_position_m"]) > args.initial_position_tolerance
            or max(initial_error["left_rotation_rad"], initial_error["right_rotation_rad"]) > args.initial_rotation_tolerance
        ):
            raise RuntimeError(
                "live robot pose does not match the demo start; refusing to send. "
                f"errors={initial_error}"
            )

        chunk_summaries: list[dict[str, Any]] = []
        all_trace: list[dict[str, Any]] = []
        frame = args.start_frame
        chunk_id = 0
        status = "completed"

        def write_progress() -> dict[str, Any]:
            summary = {
                "status": status,
                "dataset": str(dataset), "episode": args.episode,
                "start_frame": args.start_frame, "frames_replayed": frame - args.start_frame,
                "chunks_replayed": chunk_id, "pi05_action_horizon": args.horizon,
                "stage2_execute_steps": args.execute_steps,
                "control_path": (
                    "Pi0.5 delta -> norm -> denorm -> recorded-state absolute EEF "
                    "-> Stage2 -> VLAHost"
                    if not args.live_delta_reference else
                    "EXPERIMENTAL: Pi0.5 delta -> norm -> denorm -> live-state absolute EEF "
                    "-> Stage2 -> VLAHost"
                ),
                "initial_error": initial_error,
                "tool_offset_m": args.tool_offset,
                "total_stage2_safety_clipped_steps": int(
                    sum(x["stage2_safety_clipped_steps"] for x in chunk_summaries)
                ),
            }
            (output / "chunks.json").write_text(
                json.dumps(chunk_summaries, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (output / "tracking.json").write_text(
                json.dumps(all_trace, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            return summary

        try:
            while frame < len(actions):
                if args.max_chunks is not None and chunk_id >= args.max_chunks:
                    break
                actual_steps = min(args.execute_steps, len(actions) - frame)
                source = actions[frame : min(frame + args.horizon, len(actions))]
            # Pi0.5 always predicts ``horizon`` values. Match dataset boundary
            # padding for the final short chunk; only actual_steps are sent.
                if len(source) < args.horizon:
                    source = np.concatenate(
                        [source, np.repeat(source[-1:], args.horizon - len(source), axis=0)]
                    )
                normalized_delta, decoded_delta = _delta_chunk(states[frame], source, mean, std)
                decode_state = live_state if args.live_delta_reference else states[frame]
                decoded_absolute = _decode_live(decode_state, decoded_delta)
                reference = live_state.copy()
                chunk_reference = reference.copy()
                prepared: list[dict[str, Any]] = []
                for absolute in decoded_absolute[:actual_steps]:
                    step = processor._prepare_physical_action(
                        np.asarray(absolute_to_policy(absolute, "eef"), dtype=np.float32),
                        reference, chunk_reference=chunk_reference,
                    )
                    prepared.append(step)
                    reference = step["physical_action"]
                payload = [processor._physical_action_payload(step["physical_action"]) for step in prepared]
                response = session.post(
                    args.url.rstrip("/") + "/action_chunk",
                    json={"source_hz": float(args.frequency), "actions": payload}, timeout=args.timeout,
                )
                response.raise_for_status()
                result = response.json()
                if not result.get("success", False):
                    raise RuntimeError(f"VLAHost rejected chunk {chunk_id}: {result}")
                chunk_record = {
                    "status": "submitted_waiting_for_completion",
                    "chunk_id": chunk_id, "frame_start": frame,
                    "frame_end": frame + actual_steps - 1,
                    "vlahost_chunk_id": result.get("chunk_id"),
                    "stage2_safety_clipped_steps": int(sum(x["safety_clipped"] for x in prepared)),
                    "demo_target_vs_delta_decoded_at_start": _pose_errors(source[0], decoded_absolute[0]),
                }
                # Persist before state polling. If VLAHost becomes unavailable
                # or the operator interrupts, the submitted physical range is
                # still available for incident analysis.
                chunk_summaries.append(chunk_record)
                write_progress()
                trace, live_state = _wait_for_chunk(
                    session, args.url, args.timeout, args.trace_frequency, args.frequency,
                    prepared, args.tool_offset
                )
                for item in trace:
                    item["chunk_id"] = chunk_id
                    item["frame_start"] = frame
                all_trace.extend(trace)
                chunk_record["status"] = "completed"
                chunk_record["final_tracking_error"] = trace[-1]["tracking_error"]
                frame += actual_steps
                chunk_id += 1
                write_progress()
        except KeyboardInterrupt:
            status = "interrupted_after_completed_chunk"
            return write_progress()
        except Exception:
            status = "failed_after_completed_chunk"
            write_progress()
            raise
        return write_progress()
    finally:
        processor.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--execute-steps", type=int, default=30)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--max-step", type=float, default=0.05)
    parser.add_argument("--max-chunk", type=float, default=0.40)
    parser.add_argument("--max-rotation", type=float, default=0.20)
    parser.add_argument("--max-angular-velocity", type=float, default=3.0)
    parser.add_argument("--max-chunk-rotation", type=float, default=1.50)
    parser.add_argument("--tool-offset", type=float, default=0.0)
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--trace-frequency", type=float, default=100.0)
    parser.add_argument("--initial-position-tolerance", type=float, default=0.03)
    parser.add_argument("--initial-rotation-tolerance", type=float, default=0.20)
    parser.add_argument("--allow-initial-mismatch", action="store_true")
    parser.add_argument(
        "--live-delta-reference", action="store_true",
        help="experimental: decode demo delta relative to each live state; default preserves demo absolute trajectory",
    )
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not (args.send and args.confirm):
        parser.error("hardware replay requires both --send and --confirm")
    if args.horizon <= 0 or args.execute_steps <= 0 or args.execute_steps > args.horizon:
        parser.error("require 0 < --execute-steps <= --horizon")
    if args.max_chunks is not None and args.max_chunks <= 0:
        parser.error("--max-chunks must be positive")
    if args.frequency <= 0 or args.trace_frequency <= 0:
        parser.error("frequencies must be positive")
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
