#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Safely validate Zadai VLAHost EEF control and pedal intervention.

The default mode is read-only. Motion and intervention checks require both an
explicit subcommand and ``--acknowledge-robot-clear``. The motion check moves
both arms by at most 5 mm and always attempts to restore the measured start
pose. The intervention check streams a measured-pose hold chunk, observes a
pedal press/release, then submits a fresh EEF chunk to verify policy recovery.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import requests

from toolkits.realworld_check.monitor_zadai_vlahost import validate_state

MAX_TEST_TRANSLATION_M = 0.005


def pose_vector(state: dict[str, Any], side: str) -> list[float]:
    """Extract one ``xyz+xyzw`` vector from a VLAHost state."""
    pose = state[f"eef_{side}"]
    position = pose["position"]
    orientation = pose["orientation"]
    return [
        float(position["x"]),
        float(position["y"]),
        float(position["z"]),
        float(orientation["x"]),
        float(orientation["y"]),
        float(orientation["z"]),
        float(orientation["w"]),
    ]


def eef_action_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Build a bilateral absolute-EEF hold action from measured state."""
    action: dict[str, Any] = {
        "eef_left": pose_vector(state, "left"),
        "eef_right": pose_vector(state, "right"),
    }
    for side in ("left", "right"):
        gripper = state.get(f"gripper_{side}")
        if isinstance(gripper, dict) and isinstance(
            gripper.get("position"), (int, float)
        ):
            action[f"gripper_{side}"] = float(gripper["position"])
    return action


def position_error(state: dict[str, Any], target: dict[str, Any]) -> float:
    """Return maximum bilateral Cartesian position error in metres."""
    errors = []
    for side in ("left", "right"):
        measured = pose_vector(state, side)[:3]
        desired = target[f"eef_{side}"][:3]
        errors.append(math.dist(measured, desired))
    return max(errors)


class VlaHostClient:
    """Small typed wrapper around the validation-relevant HTTP endpoints."""

    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def state(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/state", timeout=self.timeout)
        response.raise_for_status()
        state = response.json()
        errors = validate_state(state)
        if errors:
            raise RuntimeError("invalid VLAHost state: " + "; ".join(errors))
        return state

    def action(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/action", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("success"):
            raise RuntimeError(f"VLAHost rejected EEF action: {result}")
        if result.get("control_space") != "eef":
            raise RuntimeError(f"unexpected control space: {result}")
        return result

    def chunk(
        self,
        payload: dict[str, Any],
        count: int = 10,
        observation_generation: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "source_hz": 10.0,
            "actions": [payload] * count,
        }
        if observation_generation is not None:
            body["observation_generation"] = observation_generation
        response = self.session.post(
            f"{self.base_url}/action_chunk",
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("success"):
            raise RuntimeError(f"VLAHost rejected EEF chunk: {result}")
        return result

    def intervention_session(self, enabled: bool) -> None:
        response = self.session.post(
            f"{self.base_url}/intervention/session",
            json={"enabled": enabled},
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("success"):
            raise RuntimeError(f"intervention session update failed: {result}")


def wait_for_position(
    client: VlaHostClient,
    target: dict[str, Any],
    tolerance_m: float,
    timeout_s: float,
) -> float:
    """Wait for the bilateral EEF target and return final position error."""
    deadline = time.monotonic() + timeout_s
    error = float("inf")
    while time.monotonic() < deadline:
        error = position_error(client.state(), target)
        if error <= tolerance_m:
            return error
        time.sleep(0.1)
    raise TimeoutError(
        f"EEF target did not converge: error={error:.6f} m, "
        f"tolerance={tolerance_m:.6f} m"
    )


def run_small_step(args: argparse.Namespace, client: VlaHostClient) -> None:
    """Execute a bounded bilateral translation and restore the start pose."""
    if not args.acknowledge_robot_clear:
        raise RuntimeError("motion requires --acknowledge-robot-clear")
    if not 0.0 < args.distance_m <= MAX_TEST_TRANSLATION_M:
        raise ValueError(f"--distance-m must be in (0, {MAX_TEST_TRANSLATION_M:.3f}]")
    start_state = client.state()
    start = eef_action_from_state(start_state)
    target = {
        key: list(value) if isinstance(value, list) else value
        for key, value in start.items()
    }
    axis_index = {"x": 0, "y": 1, "z": 2}[args.axis]
    for side in ("left", "right"):
        target[f"eef_{side}"][axis_index] += args.distance_m

    print(
        f"Executing reference-only bilateral +{args.distance_m * 1000:.1f} mm "
        f"{args.axis} step; restore is mandatory."
    )
    moved = False
    try:
        client.action(target)
        moved = True
        error = wait_for_position(client, target, args.tolerance_m, args.wait_timeout)
        print(f"Forward target reached, max Cartesian error={error * 1000:.2f} mm")
    finally:
        if moved:
            print("Restoring measured start pose...")
            client.action(start)
            error = wait_for_position(
                client, start, args.tolerance_m, args.wait_timeout
            )
            print(f"Start pose restored, max Cartesian error={error * 1000:.2f} mm")


def run_intervention(args: argparse.Namespace, client: VlaHostClient) -> None:
    """Validate pedal takeover and fresh EEF policy-chunk recovery."""
    if not args.acknowledge_robot_clear:
        raise RuntimeError("intervention check requires --acknowledge-robot-clear")
    hold = eef_action_from_state(client.state())
    seen_vr = False
    seen_release = False
    deadline = time.monotonic() + args.intervention_timeout
    client.intervention_session(True)
    try:
        client.chunk(hold)
        print(
            "Press and hold the foot pedal, move only within the cleared test area, then release."
        )
        next_refresh = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_refresh:
                client.intervention_session(True)
                next_refresh = now + 0.5
            state = client.state()
            intervention = state["intervention"]
            phase = intervention.get("phase")
            if intervention.get("intervene_flag") and phase == "vr":
                if not seen_vr:
                    print("Observed pedal takeover: phase=vr, source=human")
                seen_vr = True
            if seen_vr and phase in {"waiting_policy", "pending_policy"}:
                seen_release = True
                print(
                    f"Observed pedal release: phase={phase}; sending fresh EEF hold chunk"
                )
                hold = eef_action_from_state(state)
                client.chunk(
                    hold,
                    observation_generation=int(
                        intervention.get("refresh_generation", -1)
                    ),
                )
            if (
                seen_release
                and phase == "policy"
                and not intervention.get("intervene_flag")
            ):
                if intervention.get("policy_control_space") != "eef":
                    raise RuntimeError(
                        "policy recovered with a non-EEF control space: "
                        f"{intervention.get('policy_control_space')}"
                    )
                print("EEF policy recovery verified: phase=policy, control_space=eef")
                return
            time.sleep(0.05)
        raise TimeoutError(
            "did not observe the full pedal press -> release -> EEF policy sequence"
        )
    finally:
        client.intervention_session(False)
        print("Intervention session disabled")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "small-step", "intervention"))
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--acknowledge-robot-clear", action="store_true")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--distance-m", type=float, default=0.003)
    parser.add_argument("--tolerance-m", type=float, default=0.001)
    parser.add_argument("--wait-timeout", type=float, default=8.0)
    parser.add_argument("--intervention-timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    """Run the selected validation mode."""
    args = parse_args()
    client = VlaHostClient(args.url, args.timeout)
    try:
        state = client.state()
        print(
            "EEF preflight passed: frame="
            f"{state['capabilities'].get('eef_frame_id')}, joint=14D, bilateral pose ready"
        )
        if args.mode == "small-step":
            run_small_step(args, client)
        elif args.mode == "intervention":
            run_intervention(args, client)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
