# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.realworld_env import _stack_vector_info
from rlinf.envs.realworld.zadai.action_codec import (
    ACTION_HIGH,
    ACTION_LOW,
    eef_absolute_to_normalized,
    quaternion_xyzw_to_rot6d,
    rot6d_angular_distance,
    shift_eef_tool_frame,
)
from rlinf.envs.realworld.zadai.zadai_env import Zadai200Env, Zadai200RobotConfig


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"success": True, "chunk_id": 12, "execution": "started"}


class _Session:
    def __init__(self):
        self.posts = []

    def post(self, url, *, json=None, timeout=None):
        self.posts.append((url, json, timeout))
        return _Response()


def _make_env() -> Zadai200Env:
    env = object.__new__(Zadai200Env)
    env.config = Zadai200RobotConfig(
        server_url="http://127.0.0.1:8010",
        timeout=10.0,
        action_transport="chunk",
        intervention_session=False,
        enforce_action_delta_limit=False,
    )
    env.action_space = gym.spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)
    env._action_dim = 16
    env._physical_action_low = ACTION_LOW.copy()
    env._physical_action_high = ACTION_HIGH.copy()
    env._last_observed_physical = np.zeros(16, dtype=np.float32)
    env._last_intervention_refresh_generation = 7
    env._submitted_chunk = None
    env._submitted_chunk_index = 0
    env._submitted_chunk_started_at = None
    env._last_session_refresh = 0.0
    env._last_action_sent_at = None
    env._elapsed_steps = 0
    env._runtime_log_path = None
    env._session = _Session()
    return env


def test_begin_action_chunk_submits_entire_policy_output():
    env = _make_env()
    actions = np.zeros((30, 16), dtype=np.float32)

    env.begin_action_chunk(actions)

    assert len(env._session.posts) == 1
    url, payload, timeout = env._session.posts[0]
    assert url == "http://127.0.0.1:8010/action_chunk"
    assert timeout == 10.0
    assert payload["source_hz"] == 30.0
    assert payload["observation_generation"] == 7
    assert len(payload["actions"]) == 30
    assert env._submitted_chunk_index == 0
    assert len(env._submitted_chunk) == 30


def test_chunk_step_polls_state_without_posting_single_action(monkeypatch):
    env = _make_env()
    env.begin_action_chunk(np.zeros((2, 16), dtype=np.float32))
    obs = {
        "state": {
            "a_joint_position": np.zeros(14, dtype=np.float32),
            "b_gripper_position": np.zeros(2, dtype=np.float32),
        },
        "frames": {},
    }
    monkeypatch.setattr(env, "_refresh_session", lambda: None)
    monkeypatch.setattr(env, "_state", lambda: (obs, {}))
    monkeypatch.setattr(
        "rlinf.envs.realworld.zadai.zadai_env.time.sleep", lambda _: None
    )

    env.step(np.zeros(16, dtype=np.float32))
    env.step(np.zeros(16, dtype=np.float32))

    assert len(env._session.posts) == 1
    assert env._session.posts[0][0].endswith("/action_chunk")
    assert env._submitted_chunk is None


def test_action_chunk_limits_each_step_and_total_motion():
    env = _make_env()
    env.config.enforce_action_delta_limit = True
    env.config.max_action_delta = [0.10] * 14 + [None, None]
    env.config.enforce_chunk_action_delta_limit = True
    env.config.max_chunk_action_delta = [0.50] * 14 + [None, None]

    env.begin_action_chunk(np.ones((30, 16), dtype=np.float32))

    sent = np.stack([step["physical_action"] for step in env._submitted_chunk])
    assert np.max(np.abs(sent[:, :14])) <= 0.50 + 1e-6
    assert np.max(np.abs(np.diff(sent[:, :14], axis=0))) <= 0.10 + 1e-6
    assert np.max(np.abs(sent[0, :14])) <= 0.10 + 1e-6


def test_eef_action_payload_uses_absolute_xyz_xyzw_targets():
    env = Zadai200Env(
        {
            "control_space": "eef",
            "is_dummy": True,
            "enforce_action_delta_limit": False,
            "enforce_chunk_action_delta_limit": False,
        }
    )
    try:
        prepared = env._prepare_physical_action(
            np.zeros(20, dtype=np.float32), reference=None
        )
        payload = env._physical_action_payload(prepared["physical_action"])
    finally:
        env.close()

    assert len(payload["eef_left"]) == 7
    assert len(payload["eef_right"]) == 7
    np.testing.assert_allclose(np.linalg.norm(payload["eef_left"][3:]), 1.0)
    np.testing.assert_allclose(np.linalg.norm(payload["eef_right"][3:]), 1.0)
    assert payload["gripper_left"] == 0.5
    assert payload["gripper_right"] == 0.5


def test_eef_tool_frame_compatibility_roundtrip():
    identity = quaternion_xyzw_to_rot6d(np.asarray([0, 0, 0, 1]))
    value = np.concatenate(
        [[0.4, 0.2, 0.7], identity, [0.25], [0.3, -0.2, 0.8], identity, [0.75]]
    ).astype(np.float32)

    runtime = shift_eef_tool_frame(value, 0.095)
    restored = shift_eef_tool_frame(runtime, -0.095)

    np.testing.assert_allclose(runtime[:3], [0.495, 0.2, 0.7], atol=1e-6)
    np.testing.assert_allclose(runtime[10:13], [0.395, -0.2, 0.8], atol=1e-6)
    np.testing.assert_allclose(restored, value, atol=1e-6)


def test_eef_payload_applies_legacy_tool_compatibility_offset():
    env = Zadai200Env(
        {
            "control_space": "eef",
            "is_dummy": True,
            "eef_policy_to_runtime_tool_offset_m": 0.095,
            "enforce_action_delta_limit": False,
            "enforce_chunk_action_delta_limit": False,
        }
    )
    identity = quaternion_xyzw_to_rot6d(np.asarray([0, 0, 0, 1]))
    policy_action = np.concatenate(
        [[0.4, 0.2, 0.7], identity, [0.0], [0.3, -0.2, 0.8], identity, [0.0]]
    ).astype(np.float32)
    try:
        payload = env._physical_action_payload(policy_action)
    finally:
        env.close()

    np.testing.assert_allclose(payload["eef_left"][:3], [0.495, 0.2, 0.7])
    np.testing.assert_allclose(payload["eef_right"][:3], [0.395, -0.2, 0.8])


def test_eef_workspace_endpoints_are_serialized_strictly_inside_bounds():
    env = Zadai200Env(
        {
            "control_space": "eef",
            "is_dummy": True,
            "enforce_action_delta_limit": False,
            "enforce_chunk_action_delta_limit": False,
        }
    )
    try:
        prepared = env._prepare_physical_action(
            np.ones(20, dtype=np.float32), reference=None
        )
        payload = env._physical_action_payload(prepared["physical_action"])
    finally:
        env.close()

    assert payload["eef_left"][0] < 0.80
    assert payload["eef_right"][0] < 0.70


def test_eef_custom_workspace_clips_before_wire_serialization():
    env = Zadai200Env(
        {
            "control_space": "eef",
            "is_dummy": True,
            "eef_xyz_low": [[0.20, -0.10, 0.65], [0.15, -0.45, 0.65]],
            "eef_xyz_high": [[0.70, 0.55, 1.10], [0.60, 0.10, 1.12]],
            "enforce_action_delta_limit": False,
            "enforce_chunk_action_delta_limit": False,
        }
    )
    try:
        prepared = env._prepare_physical_action(
            np.ones(20, dtype=np.float32), reference=None
        )
        payload = env._physical_action_payload(prepared["physical_action"])
    finally:
        env.close()

    assert payload["eef_left"][0] < 0.70
    assert payload["eef_right"][0] < 0.60
    assert payload["eef_left"][2] < 1.10
    assert payload["eef_right"][2] < 1.12


def test_eef_wire_payload_clamps_out_of_workspace_measured_hold():
    env = Zadai200Env(
        {
            "control_space": "eef",
            "is_dummy": True,
            "enforce_action_delta_limit": False,
            "enforce_chunk_action_delta_limit": False,
        }
    )
    measured_hold = env._prepare_physical_action(
        np.zeros(20, dtype=np.float32), reference=None
    )["physical_action"]
    measured_hold[10] = 0.7004127264022827
    try:
        payload = env._physical_action_payload(measured_hold)
    finally:
        env.close()

    assert payload["eef_right"][0] < 0.70


def test_eef_rotation_obeys_angular_velocity_and_step_limits():
    env = Zadai200Env(
        {
            "control_space": "eef",
            "is_dummy": True,
            "step_frequency": 30.0,
            "max_eef_rotation_delta_rad": 0.10,
            "max_eef_angular_velocity_rad_s": 3.0,
            "enforce_chunk_action_delta_limit": False,
        }
    )
    identity = quaternion_xyzw_to_rot6d(np.asarray([0, 0, 0, 1]))
    half_turn = quaternion_xyzw_to_rot6d(np.asarray([0, 0, 1, 0]))
    reference = env._prepare_physical_action(
        np.zeros(20, dtype=np.float32), reference=None
    )["physical_action"]
    requested = reference.copy()
    requested[3:9] = half_turn
    try:
        prepared = env._prepare_physical_action(
            eef_absolute_to_normalized(requested), reference
        )
    finally:
        env.close()

    assert np.isclose(
        rot6d_angular_distance(identity, prepared["physical_action"][3:9]),
        3.0 / 30.0,
        atol=1e-5,
    )


def test_eef_translation_limit_uses_euclidean_distance():
    max_translation = 0.02
    env = Zadai200Env(
        {
            "control_space": "eef",
            "is_dummy": True,
            "max_eef_translation_delta": max_translation,
            "enforce_chunk_action_delta_limit": False,
        }
    )
    reference = env._prepare_physical_action(
        np.zeros(20, dtype=np.float32), reference=None
    )["physical_action"]
    requested = reference.copy()
    requested[:3] += 0.10
    requested[10:13] -= 0.10
    try:
        prepared = env._prepare_physical_action(
            eef_absolute_to_normalized(requested), reference
        )["physical_action"]
    finally:
        env.close()

    assert np.linalg.norm(prepared[:3] - reference[:3]) <= max_translation + 1e-6
    assert (
        np.linalg.norm(prepared[10:13] - reference[10:13])
        <= max_translation + 1e-6
    )


def test_vector_info_stacks_object_wrapped_sent_action():
    sent_action = np.empty(1, dtype=object)
    sent_action[0] = np.linspace(-1.0, 1.0, 16, dtype=np.float32)

    stacked = _stack_vector_info(sent_action, dtype=np.float32)

    assert stacked.shape == (1, 16)
    np.testing.assert_allclose(stacked[0], sent_action[0])
