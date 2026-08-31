# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest
import requests

from rlinf.envs.realworld.zadai.zadai_env import Zadai200Env, Zadai200RobotConfig


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"success": True}


class _Session:
    def __init__(self):
        self.posts = []

    def post(self, url, *, json=None, timeout=None):
        self.posts.append((url, json, timeout))
        return _Response()


class _FlakySession(_Session):
    def post(self, url, *, json=None, timeout=None):
        self.posts.append((url, json, timeout))
        if len(self.posts) == 1:
            raise requests.ConnectionError("stale keep-alive connection")
        return _Response()


def _make_env(config: Zadai200RobotConfig) -> Zadai200Env:
    env = object.__new__(Zadai200Env)
    env.config = config
    env._session = _Session()
    env._runtime_log_path = None
    return env


def test_reset_grippers_opens_both_sides():
    env = _make_env(
        Zadai200RobotConfig(
            server_url="http://127.0.0.1:8010",
            timeout=10.0,
        )
    )

    env._reset_grippers()

    assert env._session.posts == [
        (
            "http://127.0.0.1:8010/action",
            {"gripper_left": 0.0, "gripper_right": 0.0},
            10.0,
        )
    ]


def test_reset_grippers_can_be_disabled():
    env = _make_env(Zadai200RobotConfig(reset_gripper_position=None))

    env._reset_grippers()

    assert env._session.posts == []


def test_intervention_session_retries_one_stale_connection():
    env = _make_env(
        Zadai200RobotConfig(
            server_url="http://127.0.0.1:8010",
            timeout=10.0,
            intervention_session=True,
        )
    )
    env._session = _FlakySession()
    env._logger = type("Logger", (), {"warning": lambda *args, **kwargs: None})()

    env._post_session(True)

    assert env._session.posts == [
        (
            "http://127.0.0.1:8010/intervention/session",
            {"enabled": True},
            10.0,
        ),
        (
            "http://127.0.0.1:8010/intervention/session",
            {"enabled": True},
            10.0,
        ),
    ]


@pytest.mark.parametrize("position", [-0.01, 1.01])
def test_reset_gripper_position_rejects_out_of_range_value(position):
    with pytest.raises(ValueError, match="reset_gripper_position"):
        Zadai200RobotConfig(reset_gripper_position=position)
