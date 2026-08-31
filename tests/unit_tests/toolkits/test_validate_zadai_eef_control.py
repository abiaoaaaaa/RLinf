# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the safe Zadai EEF acceptance helper."""

from toolkits.realworld_check.validate_zadai_eef_control import (
    eef_action_from_state,
    pose_vector,
    position_error,
)


def _state():
    return {
        "eef_left": {
            "position": {"x": 0.4, "y": 0.2, "z": 0.8},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "eef_right": {
            "position": {"x": 0.4, "y": -0.2, "z": 0.8},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "gripper_left": {"position": 0.1},
        "gripper_right": {"position": 0.2},
    }


def test_build_hold_action_from_measured_state():
    action = eef_action_from_state(_state())

    assert pose_vector(_state(), "left") == [0.4, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0]
    assert action["eef_right"] == [0.4, -0.2, 0.8, 0.0, 0.0, 0.0, 1.0]
    assert action["gripper_left"] == 0.1


def test_position_error_uses_worst_arm():
    target = eef_action_from_state(_state())
    target["eef_left"][0] += 0.003
    target["eef_right"][2] += 0.004

    assert abs(position_error(_state(), target) - 0.004) < 1e-12
