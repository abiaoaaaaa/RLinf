# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the Zadai VLAHost intervention monitor."""

from toolkits.realworld_check.monitor_zadai_vlahost import (
    MonitorStats,
    intervention_signature,
    validate_state,
)


def _valid_state():
    return {
        "joint_states": {"positions": [0.0] * 14},
        "eef_left": {
            "position": {"x": 0.4, "y": 0.2, "z": 0.8},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "eef_right": {
            "position": {"x": 0.4, "y": -0.2, "z": 0.8},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "capabilities": {
            "control_spaces": ["joint", "eef"],
            "eef_action_format": "absolute_xyz_xyzw_per_arm",
            "eef_frame_id": "base_link",
        },
        "gripper_left": {"position": 0.1},
        "gripper_right": {"position": 0.2},
        "quad_image": {"stream_url": "/stream/quad.mjpg"},
        "intervention": {
            "session_enabled": True,
            "active": False,
            "intervene_flag": False,
            "phase": "policy",
            "source": "policy",
            "input_mode": 3,
            "footkey": 0,
            "enable_left": True,
            "enable_right": True,
            "vr_connected": True,
            "refresh_generation": 0,
            "policy_action": None,
            "vr_action": [0.0] * 16,
            "human_action": [0.0] * 16,
            "executed_action": None,
            "policy_action_eef": None,
            "human_action_eef": None,
            "executed_action_eef": None,
        },
    }


def test_validate_state_accepts_zadai_schema():
    assert validate_state(_valid_state()) == []


def test_validate_state_reports_wrong_action_and_joint_dimensions():
    state = _valid_state()
    state["joint_states"]["positions"] = [0.0] * 13
    state["intervention"]["executed_action"] = [0.0] * 15

    errors = validate_state(state)

    assert "joint_states.positions must contain 14 finite values" in errors
    assert "intervention.executed_action must be null or 16 finite values" in errors


def test_validate_state_requires_eef_capability_and_pose_shape():
    state = _valid_state()
    state["eef_left"]["orientation"] = {"x": 0.0, "y": 0.0, "z": 0.0}
    state["capabilities"]["control_spaces"] = ["joint"]

    errors = validate_state(state)

    assert "eef_left.orientation must contain finite x/y/z/w" in errors
    assert "capabilities.control_spaces must include eef" in errors


def test_intervention_signature_tracks_state_changes():
    before = _valid_state()["intervention"]
    after = dict(before, phase="vr", intervene_flag=True, footkey=1)

    assert intervention_signature(before) != intervention_signature(after)


def test_monitor_stats_summarizes_intervention_actions():
    stats = MonitorStats()
    intervention = _valid_state()["intervention"]
    intervention["intervene_flag"] = True
    intervention["executed_action"] = [0.5] * 16

    stats.observe(intervention, latency_ms=4.5)
    summary = stats.summary()

    assert summary["samples"] == 1
    assert summary["intervention_samples"] == 1
    assert summary["action_samples"]["executed_action"] == 1
    assert summary["max_latency_ms"] == 4.5
