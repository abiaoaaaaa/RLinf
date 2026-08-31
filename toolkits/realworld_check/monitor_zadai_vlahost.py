#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Monitor Zadai VLAHost state and intervention traces without sending actions."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

ACTION_FIELDS = (
    "policy_action",
    "vr_action",
    "human_action",
    "executed_action",
    "policy_action_eef",
    "human_action_eef",
    "executed_action_eef",
)
INTERVENTION_STATE_FIELDS = (
    "session_enabled",
    "active",
    "intervene_flag",
    "phase",
    "source",
    "input_mode",
    "footkey",
    "enable_left",
    "enable_right",
    "vr_connected",
    "refresh_generation",
)


def _finite_vector(value: Any, expected_size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == expected_size
        and all(
            isinstance(item, (int, float)) and math.isfinite(item) for item in value
        )
    )


def _finite_mapping(value: Any, names: tuple[str, ...]) -> bool:
    return isinstance(value, dict) and all(
        isinstance(value.get(name), (int, float)) and math.isfinite(value[name])
        for name in names
    )


def validate_state(state: Any) -> list[str]:
    """Return schema errors for one Zadai VLAHost ``/state`` response."""
    errors: list[str] = []
    if not isinstance(state, dict):
        return [f"state must be an object, got {type(state).__name__}"]

    joint_states = state.get("joint_states")
    if not isinstance(joint_states, dict):
        errors.append("joint_states must be an object")
    elif not _finite_vector(joint_states.get("positions"), 14):
        errors.append("joint_states.positions must contain 14 finite values")

    for side in ("left", "right"):
        pose = state.get(f"eef_{side}")
        if not isinstance(pose, dict):
            errors.append(f"eef_{side} must be an object")
            continue
        position = pose.get("position")
        orientation = pose.get("orientation")
        if not _finite_mapping(position, ("x", "y", "z")):
            errors.append(f"eef_{side}.position must contain finite x/y/z")
        if not _finite_mapping(orientation, ("x", "y", "z", "w")):
            errors.append(f"eef_{side}.orientation must contain finite x/y/z/w")

    capabilities = state.get("capabilities")
    if not isinstance(capabilities, dict):
        errors.append("capabilities must be an object")
    else:
        spaces = capabilities.get("control_spaces")
        if not isinstance(spaces, list) or "eef" not in spaces:
            errors.append("capabilities.control_spaces must include eef")
        if capabilities.get("eef_action_format") != "absolute_xyz_xyzw_per_arm":
            errors.append(
                "capabilities.eef_action_format must be absolute_xyz_xyzw_per_arm"
            )

    for side in ("left", "right"):
        gripper = state.get(f"gripper_{side}")
        if not isinstance(gripper, dict):
            errors.append(f"gripper_{side} must be an object")
            continue
        position = gripper.get("position")
        if not isinstance(position, (int, float)) or not math.isfinite(position):
            errors.append(f"gripper_{side}.position must be finite")

    quad_image = state.get("quad_image")
    if not isinstance(quad_image, dict):
        errors.append("quad_image must be an object")
    elif not quad_image.get("stream_url"):
        errors.append("quad_image.stream_url is missing")

    intervention = state.get("intervention")
    if not isinstance(intervention, dict):
        errors.append("intervention must be an object")
        return errors
    for name in ACTION_FIELDS:
        value = intervention.get(name)
        if value is not None and not _finite_vector(value, 16):
            errors.append(f"intervention.{name} must be null or 16 finite values")
    return errors


def intervention_signature(intervention: dict[str, Any]) -> tuple[Any, ...]:
    """Build the compact state used to detect intervention transitions."""
    return tuple(intervention.get(name) for name in INTERVENTION_STATE_FIELDS)


def _format_intervention(intervention: dict[str, Any]) -> str:
    values = [
        f"session={intervention.get('session_enabled')}",
        f"active={intervention.get('active')}",
        f"flag={intervention.get('intervene_flag')}",
        f"phase={intervention.get('phase')}",
        f"source={intervention.get('source')}",
        f"mode={intervention.get('input_mode')}",
        f"footkey={intervention.get('footkey')}",
        f"vr={intervention.get('vr_connected')}",
        f"generation={intervention.get('refresh_generation')}",
    ]
    action_status = [
        f"{name}={'16D' if _finite_vector(intervention.get(name), 16) else 'null'}"
        for name in ACTION_FIELDS
    ]
    return " ".join(values + action_status)


@dataclass
class MonitorStats:
    """Aggregate diagnostics collected during one monitor run."""

    started_monotonic: float = field(default_factory=time.monotonic)
    samples: int = 0
    request_errors: int = 0
    schema_error_samples: int = 0
    state_changes: int = 0
    intervention_samples: int = 0
    max_latency_ms: float = 0.0
    phases: Counter[str] = field(default_factory=Counter)
    sources: Counter[str] = field(default_factory=Counter)
    input_modes: Counter[str] = field(default_factory=Counter)
    action_samples: Counter[str] = field(default_factory=Counter)

    def observe(self, intervention: dict[str, Any], latency_ms: float) -> None:
        """Update counters from one valid HTTP response."""
        self.samples += 1
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.phases[str(intervention.get("phase"))] += 1
        self.sources[str(intervention.get("source"))] += 1
        self.input_modes[str(intervention.get("input_mode"))] += 1
        if bool(intervention.get("intervene_flag")):
            self.intervention_samples += 1
        for name in ACTION_FIELDS:
            if _finite_vector(intervention.get(name), 16):
                self.action_samples[name] += 1

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable run summary."""
        elapsed = max(time.monotonic() - self.started_monotonic, 1e-9)
        return {
            "duration_s": elapsed,
            "samples": self.samples,
            "effective_hz": self.samples / elapsed,
            "request_errors": self.request_errors,
            "schema_error_samples": self.schema_error_samples,
            "state_changes": self.state_changes,
            "intervention_samples": self.intervention_samples,
            "max_latency_ms": self.max_latency_ms,
            "phases": dict(self.phases),
            "sources": dict(self.sources),
            "input_modes": dict(self.input_modes),
            "action_samples": dict(self.action_samples),
        }


class JsonlWriter:
    """Flush every monitoring record to disk immediately."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, event: str, **payload: Any) -> None:
        """Append one timestamped event."""
        record = {
            "wall_time": time.time(),
            "wall_time_iso": datetime.now().astimezone().isoformat(),
            "event": event,
            **payload,
        }
        self._handle.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("logs") / f"vlahost_intervention_monitor_{timestamp}.jsonl"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Continuously save Zadai VLAHost /state and intervention diagnostics. "
            "The monitor never sends robot actions."
        )
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("ZADAI_VLAHOST_URL", "http://127.0.0.1:8010"),
        help="VLAHost base URL (default: ZADAI_VLAHOST_URL or localhost:8010).",
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Polling rate.")
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="HTTP timeout in seconds."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds; otherwise run until Ctrl+C.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path.")
    parser.add_argument(
        "--print-every",
        type=float,
        default=2.0,
        help="Periodic console summary interval in seconds.",
    )
    parser.add_argument(
        "--manage-session",
        action="store_true",
        help=(
            "Enable and refresh /intervention/session while monitoring, then "
            "disable it on exit. This never sends robot actions."
        ),
    )
    parser.add_argument(
        "--session-refresh",
        type=float,
        default=0.5,
        help="Intervention session refresh interval in seconds.",
    )
    args = parser.parse_args()
    if args.hz <= 0:
        parser.error("--hz must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.print_every <= 0:
        parser.error("--print-every must be positive")
    if args.session_refresh <= 0:
        parser.error("--session-refresh must be positive")
    return args


def _set_session(
    session: requests.Session,
    base_url: str,
    enabled: bool,
    timeout: float,
) -> dict[str, Any]:
    response = session.post(
        f"{base_url}/intervention/session",
        json={"enabled": enabled},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not payload.get("success", False):
        raise RuntimeError(f"VLAHost rejected intervention session update: {payload}")
    return payload


def run_monitor(args: argparse.Namespace) -> int:
    """Run the monitor until its duration expires or it receives a signal."""
    base_url = args.url.rstrip("/")
    output = args.output or _default_output()
    writer = JsonlWriter(output)
    stats = MonitorStats()
    session = requests.Session()
    stopping = False
    session_enabled = False

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    writer.write(
        "monitor_start",
        url=base_url,
        polling_hz=args.hz,
        manage_session=args.manage_session,
    )
    print(f"VLAHost monitor: {base_url}/state")
    print(f"JSONL log: {output.resolve()}")
    print("Robot actions: DISABLED (this program never calls /action or /go_home)")

    try:
        if args.manage_session:
            payload = _set_session(session, base_url, True, args.timeout)
            session_enabled = True
            writer.write("session_update", enabled=True, response=payload)
            print("Intervention session: ENABLED and managed by this monitor")
        else:
            print("Intervention session: READ ONLY (use --manage-session to enable it)")

        period = 1.0 / args.hz
        next_poll = time.monotonic()
        next_print = next_poll
        next_refresh = next_poll
        previous_signature: tuple[Any, ...] | None = None

        while not stopping:
            now = time.monotonic()
            if (
                args.duration is not None
                and now - stats.started_monotonic >= args.duration
            ):
                break

            if args.manage_session and now >= next_refresh:
                payload = _set_session(session, base_url, True, args.timeout)
                writer.write("session_refresh", enabled=True, response=payload)
                next_refresh = now + args.session_refresh

            request_started = time.monotonic()
            try:
                response = session.get(f"{base_url}/state", timeout=args.timeout)
                response.raise_for_status()
                state = response.json()
                latency_ms = (time.monotonic() - request_started) * 1000.0
                errors = validate_state(state)
                intervention = (
                    state.get("intervention", {}) if isinstance(state, dict) else {}
                )
                if not isinstance(intervention, dict):
                    intervention = {}
                stats.observe(intervention, latency_ms)
                if errors:
                    stats.schema_error_samples += 1
                writer.write(
                    "state_sample",
                    sample_index=stats.samples,
                    request_latency_ms=latency_ms,
                    validation_errors=errors,
                    state=state,
                )

                signature = intervention_signature(intervention)
                if signature != previous_signature:
                    stats.state_changes += 1
                    writer.write(
                        "intervention_change",
                        sample_index=stats.samples,
                        intervention=intervention,
                    )
                    print(
                        f"[change #{stats.state_changes}] {_format_intervention(intervention)}"
                    )
                    previous_signature = signature

                if errors:
                    print(f"[schema warning] {'; '.join(errors)}", file=sys.stderr)
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                stats.request_errors += 1
                writer.write(
                    "request_error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                print(f"[request error] {type(exc).__name__}: {exc}", file=sys.stderr)

            now = time.monotonic()
            if now >= next_print:
                print(
                    f"[status] samples={stats.samples} errors={stats.request_errors} "
                    f"intervention_samples={stats.intervention_samples} "
                    f"max_latency_ms={stats.max_latency_ms:.1f}"
                )
                next_print = now + args.print_every

            next_poll += period
            delay = next_poll - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_poll = time.monotonic()
    finally:
        if args.manage_session and session_enabled:
            try:
                payload = _set_session(session, base_url, False, args.timeout)
                writer.write("session_update", enabled=False, response=payload)
                print("Intervention session: DISABLED")
            except Exception as exc:  # best effort during monitor shutdown
                writer.write(
                    "session_disable_error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                print(f"Failed to disable intervention session: {exc}", file=sys.stderr)
        summary = stats.summary()
        writer.write("monitor_summary", **summary)
        writer.close()
        session.close()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved: {output.resolve()}")
    return 0 if stats.request_errors == 0 and stats.schema_error_samples == 0 else 1


def main() -> int:
    """Run the command-line monitor."""
    return run_monitor(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
