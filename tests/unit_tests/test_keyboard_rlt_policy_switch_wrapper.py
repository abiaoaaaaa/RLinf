# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from inspect import unwrap
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.realworld.common.wrappers import (
    keyboard_rlt_policy_switch_wrapper as switch_module,
)
from rlinf.workers.env import env_worker as env_worker_module


class _FakeListener:
    def __init__(self):
        self.events: list[str] = []

    def pop_pressed_keys(self) -> list[str]:
        events, self.events = self.events, []
        return events


class _FakeEnv(gym.Env):
    def reset(self, *, seed=None, options=None):
        return {"state": 0}, {}

    def step(self, action):
        return {"state": 1}, 0.25, False, False, {}


@pytest.fixture
def wrapped_env(monkeypatch):
    monkeypatch.setattr(switch_module, "KeyboardListener", _FakeListener)
    return switch_module.KeyboardRLTPolicySwitchWrapper(_FakeEnv())


def test_keyboard_rlt_switch_and_success_reward(wrapped_env):
    wrapped_env.listener.events = ["b"]
    _, reward, terminated, _, info = wrapped_env.step(None)
    assert reward == 0.0
    assert not terminated
    assert info["rlt_switch_flags"]

    wrapped_env.listener.events = ["c"]
    _, reward, terminated, _, info = wrapped_env.step(None)
    assert reward == 1.0
    assert not terminated
    assert info["rlt_segment_done"]
    assert not info["rlt_switch_flags"]
    assert info["rlt_policy_switch_event"] == "success"

    wrapped_env.reset()
    assert not wrapped_env.rlt_switch_flags


@pytest.mark.parametrize(
    ("key", "event"),
    [("a", "failure"), ("q", "abort")],
)
def test_keyboard_rlt_failure_reward(wrapped_env, key, event):
    wrapped_env.listener.events = [key]
    _, reward, terminated, _, info = wrapped_env.step(None)
    assert reward == 0.0
    assert terminated
    assert not info["rlt_segment_done"]
    assert not info["rlt_switch_flags"]
    assert info["rlt_policy_switch_event"] == event


class _FakeChunkEnv:
    def chunk_step(self, actions):
        observations = [{"states": torch.zeros(1, 1)}] * 2
        rewards = torch.tensor([[0.0, 1.0]])
        env_terminations = torch.zeros((1, 2), dtype=torch.bool)
        truncations = torch.zeros_like(env_terminations)
        infos = {
            "rlt_segment_done": torch.tensor([[False, True]]),
            "rlt_switch_flags": torch.tensor([[True, False]]),
        }
        return observations, rewards, env_terminations, truncations, [{}, infos]


def test_rlt_segment_terminal_does_not_end_robot_episode(monkeypatch):
    monkeypatch.setattr(
        env_worker_module,
        "prepare_actions",
        lambda raw_chunk_actions, **kwargs: raw_chunk_actions,
    )
    worker = SimpleNamespace()
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "env_type": "realworld",
                    "auto_reset": True,
                    "ignore_terminations": False,
                }
            }
        }
    )
    worker.model_cfg = OmegaConf.create(
        {
            "model_type": "rlt_mlp_policy",
            "num_action_chunks": 2,
            "action_dim": 1,
        }
    )
    worker.env_list = [_FakeChunkEnv()]
    worker.use_external_reward_model = False

    output, _, _ = unwrap(env_worker_module.EnvWorker.env_interact_step)(
        worker,
        torch.zeros((1, 2, 1)),
        0,
    )

    assert not output.dones.any()
    assert torch.equal(output.terminations, torch.tensor([[False, True]]))
    assert torch.equal(output.rlt_switch_flags, torch.tensor([[True, False]]))
