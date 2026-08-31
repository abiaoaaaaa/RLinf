# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class KeyboardRLTPolicySwitchWrapper(gym.Wrapper):
    """Control the RLT phase and sparse terminal reward from the keyboard.

    ``b`` enters the RLT critical phase. ``c`` ends that phase with reward 1
    and returns control to the VLA without resetting. ``a``/``q`` terminate
    the robot episode with reward 0 so the existing auto-reset path can run.
    """

    PEDAL_DEBOUNCE_S = 0.2

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.listener = KeyboardListener()
        self._rlt_switch_flags = False
        self._last_press_ts: dict[str, float] = {}

    @property
    def rlt_switch_flags(self) -> bool:
        return self._rlt_switch_flags

    def reset(self, *, seed=None, options=None):
        self._rlt_switch_flags = False
        self._last_press_ts.clear()
        self.listener.pop_pressed_keys()
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, _, _, truncated, info = self.env.step(action)
        reward = 0.0
        terminated = False
        rlt_segment_done = False

        event: str | None = None
        for key in self.listener.pop_pressed_keys():
            now = time.monotonic()
            if now - self._last_press_ts.get(key, -math.inf) < self.PEDAL_DEBOUNCE_S:
                continue
            self._last_press_ts[key] = now

            if key == "b":
                if not self._rlt_switch_flags:
                    event = "enter_actor"
                    self._rlt_switch_flags = True
                    self._log_info(
                        "Pedal 'b' pressed; entering the RLT critical phase."
                    )
                else:
                    event = "actor_already_active"
            elif key == "c":
                event = "success"
                reward = 1.0
                rlt_segment_done = True
                self._rlt_switch_flags = False
                self._log_info("Key 'c' pressed; RLT segment succeeded.")
                break
            elif key in {"a", "q"}:
                event = "failure" if key == "a" else "abort"
                reward = 0.0
                terminated = True
                self._rlt_switch_flags = False
                self._log_info(
                    f"Key '{key}' pressed; terminating the episode with reward 0."
                )
                break

        info["rlt_switch_flags"] = self._rlt_switch_flags
        info["rlt_segment_done"] = rlt_segment_done
        info["rlt_policy_switch_event"] = event
        return obs, reward, terminated, truncated, info

    def _log_info(self, message: str) -> None:
        logger = getattr(self._base_env(), "_logger", None)
        if logger is not None:
            logger.info(message)

    def _base_env(self):
        return getattr(self.env, "unwrapped", self.env)
