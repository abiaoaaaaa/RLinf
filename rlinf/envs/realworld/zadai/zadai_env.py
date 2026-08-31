# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Minimal HTTP Gym environment for the Zadai/M6 VLAHost bridge."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import cv2
import gymnasium as gym
import numpy as np
import requests

from rlinf.envs.realworld.common.wrappers.keyboard_rlt_policy_switch_wrapper import (
    KeyboardRLTPolicySwitchWrapper,
)
from rlinf.utils.logging import get_logger
from rlinf.utils.runtime_log import append_runtime_event

from .action_codec import (
    ACTION_HIGH,
    ACTION_LOW,
    DEFAULT_EEF_XYZ_HIGH,
    DEFAULT_EEF_XYZ_LOW,
    absolute_to_policy,
    action_dim,
    clamp_rot6d_angle,
    policy_to_absolute,
    quaternion_xyzw_to_rot6d,
    rot6d_to_quaternion_xyzw,
)


def _image_from_state(
    session: requests.Session, base_url: str, image: dict[str, Any], timeout: float
) -> np.ndarray:
    if "data" in image:
        raw = base64.b64decode(image["data"])
    else:
        stream_url = image.get("stream_url")
        if not stream_url:
            raise ValueError("quad_image has neither data nor stream_url")
        url = urljoin(base_url.rstrip("/") + "/", str(stream_url).lstrip("/"))
        response = session.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        buf = bytearray()
        found = False
        try:
            for chunk in response.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                if not found:
                    start = chunk.find(b"\xff\xd8")
                    if start < 0:
                        continue
                    buf.extend(chunk[start:])
                    found = True
                else:
                    buf.extend(chunk)
                end = buf.rfind(b"\xff\xd9")
                if end >= 0:
                    buf = buf[: end + 2]
                    break
        finally:
            response.close()
        raw = bytes(buf)
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("failed to decode VLAHost quad image")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def split_vlahost_image(
    image: np.ndarray, mode: str = "hero3"
) -> dict[str, np.ndarray]:
    """Split the VLAHost mosaic using the same layout as lebot_client.py."""
    height, width = image.shape[:2]
    if mode == "hero3":
        top = int(round(height * 2.0 / 3.0))
        crops = {
            "top": image[:top, :],
            "wrist_L": image[top:, : width // 2],
            "wrist_R": image[top:, width // 2 :],
        }
    elif mode == "quad4":
        mid_h, mid_w = height // 2, width // 2
        crops = {
            "top": image[:mid_h, :mid_w],
            "wrist_L": image[:mid_h, mid_w:],
            "wrist_R": image[mid_h:, :mid_w],
        }
    else:
        raise ValueError(f"unknown VLAHost image_split_mode: {mode!r}")
    return {
        name: cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        for name, frame in crops.items()
    }


def _end_effector_position(value: Any) -> float:
    if isinstance(value, dict):
        value = value.get("position")
    if isinstance(value, list):
        value = value[0] if value else 0.0
    return float(value or 0.0)


def _eef_pose_to_xyz_rot6d(value: Any, name: str) -> np.ndarray:
    """Parse one VLAHost pose dictionary into dataset xyz+rot6d form."""
    if not isinstance(value, dict):
        raise ValueError(f"VLAHost /state did not return {name}")
    position = value.get("position")
    orientation = value.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        raise ValueError(f"VLAHost {name} must contain position and orientation")
    xyz = np.asarray([position.get(axis) for axis in "xyz"], dtype=np.float32)
    quat = np.asarray([orientation.get(axis) for axis in "xyzw"], dtype=np.float32)
    if not np.isfinite(xyz).all() or not np.isfinite(quat).all():
        raise ValueError(f"VLAHost {name} contains non-finite pose values")
    return np.concatenate([xyz, quaternion_xyzw_to_rot6d(quat)]).astype(np.float32)


def _eef_wire_action_to_rot6d(value: Any, name: str) -> np.ndarray:
    """Convert VLAHost ``[L xyz+quat+grip, R ...]`` to dataset EEF 20D."""
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (16,) or not np.isfinite(action).all():
        raise ValueError(f"VLAHost {name} must contain 16 finite EEF wire values")
    return np.concatenate(
        [
            action[:3],
            quaternion_xyzw_to_rot6d(action[3:7]),
            action[7:8],
            action[8:11],
            quaternion_xyzw_to_rot6d(action[11:15]),
            action[15:16],
        ]
    ).astype(np.float32)


@dataclass
class Zadai200RobotConfig:
    server_url: str = "http://127.0.0.1:8010"
    timeout: float = 2.0
    # Low-level command cadence used by either /action frames or /action_chunk.
    step_frequency: float = 30.0
    # ``joint`` preserves the original 16-D path. ``eef`` uses bilateral
    # [xyz, rot6d, gripper] absolute targets with a normalized 20-D RLT actor.
    control_space: str = "joint"
    action_mode: str = "absolute"
    action_transport: str = "action"
    image_split_mode: str = "hero3"
    image_width: int = 640
    image_height: int = 480
    max_num_steps: int = 300
    task_description: str = ""
    is_dummy: bool = False
    intervention_session: bool = False
    intervention_refresh_s: float = 0.5
    reset_wait_s: float = 0.0
    # ``None`` leaves the grippers unchanged. On the Zadai DM gripper input,
    # 0 is open and 1 is closed.
    reset_gripper_position: float | None = 0.0
    runtime_log_path: str | None = None
    # Per-low-level-step target change limits in physical units.  The first
    # 14 values are radians and the last two are normalized gripper values.
    # These are deliberately conservative relative to the 30 Hz dataset.
    enforce_action_delta_limit: bool = True
    # ``None`` means no limit; grippers are intentionally unrestricted.
    max_action_delta: list[float | None] = field(
        default_factory=lambda: [0.10] * 14 + [None, None]
    )
    # Bound the total motion of a submitted chunk relative to the observation
    # from which it was generated. This prevents per-step limits accumulating
    # into a large one-second move.
    enforce_chunk_action_delta_limit: bool = True
    max_chunk_action_delta: list[float | None] = field(
        default_factory=lambda: [0.50] * 14 + [None, None]
    )
    joint_lower: list[float] = field(default_factory=lambda: [-np.pi] * 14)
    joint_upper: list[float] = field(default_factory=lambda: [np.pi] * 14)
    eef_xyz_low: list[list[float]] = field(
        default_factory=lambda: DEFAULT_EEF_XYZ_LOW.tolist()
    )
    eef_xyz_high: list[list[float]] = field(
        default_factory=lambda: DEFAULT_EEF_XYZ_HIGH.tolist()
    )
    max_eef_translation_delta: float = 0.02
    max_eef_chunk_translation_delta: float = 0.12
    max_eef_rotation_delta_rad: float = 0.12
    max_eef_angular_velocity_rad_s: float = 1.5
    max_eef_chunk_rotation_delta_rad: float = 0.50

    def __post_init__(self):
        if self.action_mode != "absolute":
            raise ValueError("Zadai200Env supports absolute targets only")
        if self.control_space not in {"joint", "eef"}:
            raise ValueError("control_space must be 'joint' or 'eef'")
        if self.action_transport not in {"action", "chunk"}:
            raise ValueError("action_transport must be 'action' or 'chunk'")
        if len(self.joint_lower) != 14 or len(self.joint_upper) != 14:
            raise ValueError("Zadai joint bounds must contain 14 arm values")
        low = np.asarray(self.eef_xyz_low, dtype=np.float32)
        high = np.asarray(self.eef_xyz_high, dtype=np.float32)
        if low.shape != (2, 3) or high.shape != (2, 3) or np.any(low >= high):
            raise ValueError("eef_xyz_low/high must be valid [left, right] xyz boxes")
        if self.max_eef_translation_delta <= 0:
            raise ValueError("max_eef_translation_delta must be positive")
        if self.max_eef_chunk_translation_delta <= 0:
            raise ValueError("max_eef_chunk_translation_delta must be positive")
        if self.max_eef_rotation_delta_rad <= 0:
            raise ValueError("max_eef_rotation_delta_rad must be positive")
        if self.max_eef_angular_velocity_rad_s <= 0:
            raise ValueError("max_eef_angular_velocity_rad_s must be positive")
        if self.max_eef_chunk_rotation_delta_rad <= 0:
            raise ValueError("max_eef_chunk_rotation_delta_rad must be positive")
        if len(self.max_action_delta) != 16 or any(
            value is not None and float(value) <= 0 for value in self.max_action_delta
        ):
            raise ValueError("max_action_delta must contain 16 positive values or None")
        if len(self.max_chunk_action_delta) != 16 or any(
            value is not None and float(value) <= 0
            for value in self.max_chunk_action_delta
        ):
            raise ValueError(
                "max_chunk_action_delta must contain 16 positive values or None"
            )
        if self.step_frequency <= 0:
            raise ValueError("Zadai step_frequency must be positive")
        if self.reset_gripper_position is not None and not (
            0.0 <= float(self.reset_gripper_position) <= 1.0
        ):
            raise ValueError("reset_gripper_position must be in [0, 1] or None")
        if isinstance(self.is_dummy, str):
            self.is_dummy = self.is_dummy.strip().lower() in {"1", "true", "yes", "on"}


class Zadai200Env(gym.Env):
    """Zadai 200 environment with normalized RLT actions.

    The public action is ``[-1, 1]^16``; this class converts it to VLAHost's
    physical absolute joint/gripper values. In chunk transport mode, the whole
    policy output is submitted through ``POST /action_chunk`` so VLAHost can
    safely hand control back after an intervention. Observations and
    intervention labels are still sampled once per low-level step.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Any = None,
        hardware_info: Any = None,
        env_idx: int = 0,
        env_cfg: Any = None,
    ):
        del worker_info, hardware_info, env_idx, env_cfg
        self.config = Zadai200RobotConfig(**override_cfg)
        self._logger = get_logger()
        self._session = requests.Session()
        self._elapsed_steps = 0
        self._last_session_refresh = 0.0
        self._last_action_sent_at: float | None = None
        self._last_observed_physical: np.ndarray | None = None
        self._last_intervention_refresh_generation = 0
        self._submitted_chunk: list[dict[str, Any]] | None = None
        self._submitted_chunk_index = 0
        self._submitted_chunk_started_at: float | None = None
        self._runtime_log_path = self.config.runtime_log_path
        append_runtime_event(
            self._runtime_log_path,
            {
                "event": "env_init",
                "server_url": self.config.server_url,
                "step_frequency_hz": self.config.step_frequency,
                "action_mode": self.config.action_mode,
                "max_action_delta": self.config.max_action_delta,
            },
        )
        self._last_images: dict[str, np.ndarray] | None = None
        self._task_description = self.config.task_description

        self._action_dim = action_dim(self.config.control_space)
        if self.config.control_space == "joint":
            self._physical_action_low = np.asarray(
                self.config.joint_lower + [0.0, 0.0], dtype=np.float32
            )
            self._physical_action_high = np.asarray(
                self.config.joint_upper + [1.0, 1.0], dtype=np.float32
            )
            if not np.allclose(
                self._physical_action_low, ACTION_LOW
            ) or not np.allclose(self._physical_action_high, ACTION_HIGH):
                raise ValueError(
                    "Zadai custom joint bounds must also be wired into the RLT codec"
                )
        else:
            self._physical_action_low = np.full(20, -np.inf, dtype=np.float32)
            self._physical_action_high = np.full(20, np.inf, dtype=np.float32)
            for arm, offset in enumerate((0, 10)):
                self._physical_action_low[offset : offset + 3] = (
                    self.config.eef_xyz_low[arm]
                )
                self._physical_action_high[offset : offset + 3] = (
                    self.config.eef_xyz_high[arm]
                )
                self._physical_action_low[offset + 9] = 0.0
                self._physical_action_high[offset + 9] = 1.0
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(self._action_dim,), dtype=np.float32
        )
        if self.config.control_space == "joint":
            state_spaces = {
                "a_joint_position": gym.spaces.Box(
                    -np.inf, np.inf, shape=(14,), dtype=np.float32
                ),
                "b_gripper_position": gym.spaces.Box(
                    -np.inf, np.inf, shape=(2,), dtype=np.float32
                ),
            }
        else:
            # One key preserves the dataset's interleaved per-arm layout:
            # [L xyz, L rot6d, L grip, R xyz, R rot6d, R grip].
            state_spaces = {
                "a_eef_state": gym.spaces.Box(
                    -np.inf, np.inf, shape=(20,), dtype=np.float32
                )
            }
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_spaces),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            0,
                            255,
                            shape=(
                                self.config.image_height,
                                self.config.image_width,
                                3,
                            ),
                            dtype=np.uint8,
                        )
                        for name in ("top", "wrist_L", "wrist_R")
                    }
                ),
            }
        )

    @property
    def task_description(self):
        return self._task_description

    def _url(self, path: str) -> str:
        return f"{self.config.server_url.rstrip('/')}/{path.lstrip('/')}"

    def _post_session(self, enabled: bool) -> None:
        if not self.config.intervention_session or self.config.is_dummy:
            return
        # This endpoint is idempotent. A reset can leave the pooled HTTP
        # connection idle while go_home and reset_wait_s run, and Uvicorn may
        # close that keep-alive socket. Retry once so urllib3 opens a fresh
        # connection instead of terminating the whole training process.
        for attempt in range(2):
            try:
                response = self._session.post(
                    self._url("intervention/session"),
                    json={"enabled": enabled},
                    timeout=self.config.timeout,
                )
                break
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 1:
                    raise
                self._logger.warning(
                    "VLAHost intervention session request lost; retrying once "
                    "with a fresh HTTP connection"
                )
        response.raise_for_status()
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(f"VLAHost intervention session rejected: {result}")

    def _refresh_session(self) -> None:
        now = time.monotonic()
        if now - self._last_session_refresh >= self.config.intervention_refresh_s:
            self._post_session(True)
            self._last_session_refresh = now

    def _dummy_observation(self) -> dict[str, Any]:
        if self.config.control_space == "joint":
            robot_state = {
                "a_joint_position": np.zeros(14, dtype=np.float32),
                "b_gripper_position": np.zeros(2, dtype=np.float32),
            }
        else:
            identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
            robot_state = {
                "a_eef_state": np.concatenate(
                    [np.zeros(3), identity, [0.0], np.zeros(3), identity, [0.0]]
                ).astype(np.float32)
            }
        return {
            "state": robot_state,
            "frames": {
                name: np.zeros(
                    (self.config.image_height, self.config.image_width, 3),
                    dtype=np.uint8,
                )
                for name in ("top", "wrist_L", "wrist_R")
            },
        }

    def _reset_grippers(self) -> None:
        position = self.config.reset_gripper_position
        if position is None or self.config.is_dummy:
            return
        response = self._session.post(
            self._url("action"),
            json={
                "gripper_left": float(position),
                "gripper_right": float(position),
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success", False):
            raise RuntimeError(f"VLAHost gripper reset failed: {payload}")
        append_runtime_event(
            self._runtime_log_path,
            {
                "event": "gripper_reset",
                "gripper_left": float(position),
                "gripper_right": float(position),
            },
        )

    def _state(self) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self._session.get(self._url("state"), timeout=self.config.timeout)
        response.raise_for_status()
        state = response.json()
        if not isinstance(state, dict):
            raise ValueError("VLAHost /state must return a JSON object")
        joints = state.get("joint_states", {})
        positions = joints.get("positions", []) if isinstance(joints, dict) else []
        if self.config.control_space == "joint" and len(positions) != 14:
            raise ValueError(
                f"VLAHost returned {len(positions)} arm joints; expected 14"
            )

        ee_left = _end_effector_position(state.get("gripper_left"))
        ee_right = _end_effector_position(state.get("gripper_right"))
        quad = state.get("quad_image")
        if not isinstance(quad, dict):
            raise ValueError("VLAHost /state did not return quad_image")
        images = split_vlahost_image(
            _image_from_state(
                self._session, self.config.server_url, quad, self.config.timeout
            ),
            self.config.image_split_mode,
        )
        self._last_images = images
        grippers = np.asarray([ee_left, ee_right], dtype=np.float32)
        if self.config.control_space == "joint":
            self._last_observed_physical = np.concatenate(
                [np.asarray(positions, dtype=np.float32), grippers]
            )
            robot_state = {
                "a_joint_position": np.asarray(positions, dtype=np.float32),
                "b_gripper_position": grippers,
            }
        else:
            left_pose = _eef_pose_to_xyz_rot6d(state.get("eef_left"), "eef_left")
            right_pose = _eef_pose_to_xyz_rot6d(state.get("eef_right"), "eef_right")
            self._last_observed_physical = np.concatenate(
                [left_pose, grippers[:1], right_pose, grippers[1:]]
            ).astype(np.float32)
            robot_state = {"a_eef_state": self._last_observed_physical.copy()}
        raw_obs = {"state": robot_state, "frames": images}
        intervention = state.get("intervention")
        info: dict[str, Any] = {
            "stamp_ns": state.get("stamp"),
            "intervention": intervention if isinstance(intervention, dict) else {},
        }
        if isinstance(intervention, dict):
            self._last_intervention_refresh_generation = int(
                intervention.get("refresh_generation", 0)
            )
            info["intervene_flag"] = bool(intervention.get("intervene_flag", False))
            suffix = "_eef" if self.config.control_space == "eef" else ""
            info["policy_action"] = intervention.get(f"policy_action{suffix}")
            info["human_action"] = intervention.get(f"human_action{suffix}")
            info["executed_action"] = intervention.get(f"executed_action{suffix}")
            if self.config.control_space == "eef":
                for key in ("policy_action", "human_action", "executed_action"):
                    if info[key] is not None:
                        info[key] = _eef_wire_action_to_rot6d(info[key], key)
            if (
                self.config.control_space == "eef"
                and info["intervene_flag"]
                and info["executed_action"] is None
            ):
                # During a QP/VR intervention the measured EEF state is the
                # closest observable executed target and has the correct 20-D
                # Stage 2 semantics.
                info["executed_action"] = self._last_observed_physical.copy()
            if info["intervene_flag"] and info["executed_action"] is not None:
                info["intervene_action"] = np.asarray(
                    absolute_to_policy(
                        np.asarray(info["executed_action"], dtype=np.float32),
                        self.config.control_space,
                    ),
                    dtype=np.float32,
                )
        return raw_obs, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._elapsed_steps = 0
        self._submitted_chunk = None
        self._submitted_chunk_index = 0
        self._submitted_chunk_started_at = None
        if self.config.is_dummy:
            return self._dummy_observation(), {}
        result = self._session.post(self._url("go_home"), timeout=self.config.timeout)
        result.raise_for_status()
        payload = result.json()
        if not payload.get("success", False):
            raise RuntimeError(f"VLAHost go_home failed: {payload}")
        self._reset_grippers()
        if self.config.reset_wait_s > 0:
            time.sleep(self.config.reset_wait_s)
        self._post_session(True)
        self._last_session_refresh = time.monotonic()
        obs, info = self._state()
        append_runtime_event(
            self._runtime_log_path,
            {
                "event": "env_reset",
                "elapsed_steps": self._elapsed_steps,
                "observed_physical": self._last_observed_physical,
            },
        )
        return obs, info

    def _prepare_physical_action(
        self,
        action: np.ndarray,
        reference: np.ndarray | None,
        chunk_reference: np.ndarray | None = None,
    ) -> dict[str, Any]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self._action_dim,) or not np.isfinite(action).all():
            raise ValueError(
                "Zadai action must be finite with shape "
                f"({self._action_dim},), got {action.shape}"
            )
        requested_normalized = np.clip(
            action, self.action_space.low, self.action_space.high
        )
        requested_physical = policy_to_absolute(
            requested_normalized, self.config.control_space
        )
        physical_action = requested_physical.copy()
        safety_delta = np.zeros(self._action_dim, dtype=np.float32)
        if (
            self.config.control_space == "joint"
            and self.config.enforce_action_delta_limit
            and reference is not None
        ):
            max_delta = np.asarray(
                [
                    np.inf if value is None else float(value)
                    for value in self.config.max_action_delta
                ],
                dtype=np.float32,
            )
            physical_action = np.clip(
                physical_action, reference - max_delta, reference + max_delta
            )
            physical_action = np.clip(
                physical_action,
                self._physical_action_low,
                self._physical_action_high,
            )
        if (
            self.config.control_space == "joint"
            and self.config.enforce_chunk_action_delta_limit
            and chunk_reference is not None
        ):
            max_chunk_delta = np.asarray(
                [
                    np.inf if value is None else float(value)
                    for value in self.config.max_chunk_action_delta
                ],
                dtype=np.float32,
            )
            physical_action = np.clip(
                physical_action,
                chunk_reference - max_chunk_delta,
                chunk_reference + max_chunk_delta,
            )
            physical_action = np.clip(
                physical_action,
                self._physical_action_low,
                self._physical_action_high,
            )
        if self.config.control_space == "eef":
            physical_action = np.clip(
                physical_action,
                self._physical_action_low,
                self._physical_action_high,
            )
            for offset in (0, 10):
                if self.config.enforce_action_delta_limit and reference is not None:
                    physical_action[offset : offset + 3] = np.clip(
                        physical_action[offset : offset + 3],
                        reference[offset : offset + 3]
                        - self.config.max_eef_translation_delta,
                        reference[offset : offset + 3]
                        + self.config.max_eef_translation_delta,
                    )
                if (
                    self.config.enforce_chunk_action_delta_limit
                    and chunk_reference is not None
                ):
                    physical_action[offset : offset + 3] = np.clip(
                        physical_action[offset : offset + 3],
                        chunk_reference[offset : offset + 3]
                        - self.config.max_eef_chunk_translation_delta,
                        chunk_reference[offset : offset + 3]
                        + self.config.max_eef_chunk_translation_delta,
                    )
                rotation = slice(offset + 3, offset + 9)
                if self.config.enforce_action_delta_limit and reference is not None:
                    max_step_rotation = min(
                        self.config.max_eef_rotation_delta_rad,
                        self.config.max_eef_angular_velocity_rad_s
                        / self.config.step_frequency,
                    )
                    physical_action[rotation] = clamp_rot6d_angle(
                        physical_action[rotation],
                        reference[rotation],
                        max_step_rotation,
                    )
                if (
                    self.config.enforce_chunk_action_delta_limit
                    and chunk_reference is not None
                ):
                    physical_action[rotation] = clamp_rot6d_angle(
                        physical_action[rotation],
                        chunk_reference[rotation],
                        self.config.max_eef_chunk_rotation_delta_rad,
                    )
        safety_delta = requested_physical - physical_action
        return {
            "requested_normalized": requested_normalized.copy(),
            "requested_physical": requested_physical.copy(),
            "physical_action": physical_action.copy(),
            "safety_delta": safety_delta,
            "safety_clipped": bool(np.any(np.abs(safety_delta) > 1e-6)),
        }

    def _physical_action_payload(self, physical_action: np.ndarray) -> dict[str, Any]:
        if self.config.control_space == "joint":
            return {
                "jointcmd_left": physical_action[:7].tolist(),
                "jointcmd_right": physical_action[7:14].tolist(),
                "gripper_left": float(physical_action[14]),
                "gripper_right": float(physical_action[15]),
            }
        left_quat = rot6d_to_quaternion_xyzw(physical_action[3:9])
        right_quat = rot6d_to_quaternion_xyzw(physical_action[13:19])
        return {
            "eef_left": np.concatenate([physical_action[:3], left_quat]).tolist(),
            "eef_right": np.concatenate([physical_action[10:13], right_quat]).tolist(),
            "gripper_left": float(physical_action[9]),
            "gripper_right": float(physical_action[19]),
        }

    def begin_action_chunk(self, actions: np.ndarray) -> None:
        """Submit one intervention-aware VLAHost action chunk."""
        if self.config.action_transport != "chunk" or self.config.is_dummy:
            return
        actions = np.asarray(actions, dtype=np.float32)
        if (
            actions.ndim != 2
            or actions.shape[1] != self._action_dim
            or len(actions) == 0
        ):
            raise ValueError(
                "Zadai action chunk must have shape "
                f"[chunk_size, {self._action_dim}], got "
                f"{actions.shape}"
            )
        prepared = []
        reference = self._last_observed_physical
        chunk_reference = (
            None
            if self._last_observed_physical is None
            else self._last_observed_physical.copy()
        )
        for action in actions:
            step = self._prepare_physical_action(
                action,
                reference,
                chunk_reference=chunk_reference,
            )
            prepared.append(step)
            reference = step["physical_action"]
        response = self._session.post(
            self._url("action_chunk"),
            json={
                "source_hz": float(self.config.step_frequency),
                "observation_generation": int(
                    self._last_intervention_refresh_generation
                ),
                "actions": [
                    self._physical_action_payload(step["physical_action"])
                    for step in prepared
                ],
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(f"VLAHost rejected action chunk: {result}")
        self._submitted_chunk = prepared
        self._submitted_chunk_index = 0
        self._submitted_chunk_started_at = time.monotonic()
        append_runtime_event(
            self._runtime_log_path,
            {
                "event": "action_chunk_submit",
                "chunk_size": len(prepared),
                "chunk_id": result.get("chunk_id"),
                "execution": result.get("execution"),
                "observation_generation": self._last_intervention_refresh_generation,
            },
        )

    def step(self, action):
        prepared = self._prepare_physical_action(action, self._last_observed_physical)
        requested_normalized = prepared["requested_normalized"]
        requested_physical = prepared["requested_physical"]
        physical_action = prepared["physical_action"]
        if self.config.is_dummy:
            obs = self._dummy_observation()
            info = {}
        else:
            self._refresh_session()
            action_sent_at = time.monotonic()
            previous_action_sent_at = self._last_action_sent_at
            self._last_action_sent_at = action_sent_at
            if self.config.action_transport == "chunk":
                if (
                    self._submitted_chunk is None
                    or self._submitted_chunk_started_at is None
                ):
                    raise RuntimeError("Zadai step called before begin_action_chunk")
                if self._submitted_chunk_index >= len(self._submitted_chunk):
                    raise RuntimeError("Zadai action chunk was consumed more than once")
                prepared = self._submitted_chunk[self._submitted_chunk_index]
                requested_normalized = prepared["requested_normalized"]
                requested_physical = prepared["requested_physical"]
                physical_action = prepared["physical_action"]
                target_time = self._submitted_chunk_started_at + (
                    (self._submitted_chunk_index + 1) / self.config.step_frequency
                )
                time.sleep(max(0.0, target_time - time.monotonic()))
                self._submitted_chunk_index += 1
                if self._submitted_chunk_index >= len(self._submitted_chunk):
                    self._submitted_chunk = None
                    self._submitted_chunk_started_at = None
            else:
                response = self._session.post(
                    self._url("action"),
                    json=self._physical_action_payload(physical_action),
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                result = response.json()
                if not result.get("success", False):
                    raise RuntimeError(f"VLAHost rejected action: {result}")
            # Match the Franka env contract: wait for the remainder of the
            # control period, then read the observation that follows this
            # command.  This keeps each replay transition step-aligned.
            if self.config.action_transport != "chunk":
                elapsed = time.monotonic() - action_sent_at
                time.sleep(max(0.0, (1.0 / self.config.step_frequency) - elapsed))
            obs, info = self._state()
            state_read_at = time.monotonic()
            action_period = (
                None
                if previous_action_sent_at is None
                else action_sent_at - previous_action_sent_at
            )
            info["target_step_frequency_hz"] = float(self.config.step_frequency)
            info["action_to_state_s"] = float(state_read_at - action_sent_at)
            info["action_period_s"] = (
                None if action_period is None else float(action_period)
            )
            info["measured_action_frequency_hz"] = (
                None
                if action_period is None or action_period <= 0
                else float(1.0 / action_period)
            )
            info["safety_action_clipped"] = prepared["safety_clipped"]
            info["safety_action_delta"] = prepared["safety_delta"]
        info["sent_action"] = np.asarray(
            absolute_to_policy(physical_action, self.config.control_space),
            dtype=np.float32,
        )
        self._elapsed_steps += 1
        truncated = bool(
            self.config.max_num_steps > 0
            and self._elapsed_steps >= self.config.max_num_steps
        )
        append_runtime_event(
            self._runtime_log_path,
            {
                "event": "env_step",
                "elapsed_steps": self._elapsed_steps,
                "requested_action_normalized": requested_normalized,
                "requested_action_physical": requested_physical,
                "sent_action_physical": physical_action,
                "observed_physical": self._last_observed_physical,
                "reward": 0.0,
                "terminated": False,
                "truncated": truncated,
                "intervene_flag": info.get("intervene_flag", False),
                "intervention": info.get("intervention", {}),
                "safety_action_clipped": info.get("safety_action_clipped", False),
                "safety_action_delta": info.get(
                    "safety_action_delta",
                    np.zeros(self._action_dim, dtype=np.float32),
                ),
                "target_step_frequency_hz": info.get("target_step_frequency_hz"),
                "measured_action_frequency_hz": info.get(
                    "measured_action_frequency_hz"
                ),
                "action_to_state_s": info.get("action_to_state_s"),
            },
        )
        return obs, 0.0, False, truncated, info

    def close(self):
        if self.config.intervention_session and not self.config.is_dummy:
            try:
                self._post_session(False)
            except Exception as exc:  # best effort during process shutdown
                self._logger.warning(
                    "Failed to disable VLAHost intervention session: %s", exc
                )
        self._session.close()


def create_zadai_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Any,
) -> gym.Env:
    env = Zadai200Env(override_cfg, worker_info, hardware_info, env_idx, env_cfg)
    if env_cfg.get("keyboard_reward_wrapper", None) == "rlt_policy_switch":
        env = KeyboardRLTPolicySwitchWrapper(env)
    return env
