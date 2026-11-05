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

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

from rlinf.algorithms.rlt.expert import predict_expert_actions
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.utils.runtime_log import append_runtime_event


@dataclass(kw_only=True)
class RLTRouteContext:
    env_obs: dict[str, Any]
    rlt_obs: dict[str, torch.Tensor]
    student_actions: torch.Tensor
    result: dict[str, Any]
    mode: Literal["train", "eval"]
    rlt_switch_flags: torch.Tensor | None = None
    intervene_requested: torch.Tensor | None = None
    expert_model: Any | None = None
    version: int = 0
    default_actor_switch: bool = False


@dataclass(kw_only=True)
class RLTRouteOutput:
    actions: torch.Tensor
    result: dict[str, Any]


def _last_info_bool(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    default: bool,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), bool(default), dtype=torch.bool, device=device)
    value = torch.as_tensor(value, device=device)
    if value.numel() == 1:
        return torch.full(
            (batch_size,),
            bool(value.reshape(-1)[0].item()),
            dtype=torch.bool,
            device=device,
        )
    return value.reshape(batch_size, -1).to(torch.bool)[:, -1]


def _flatten_action_chunk(actions: torch.Tensor) -> torch.Tensor:
    if actions.dim() <= 2:
        return actions
    return actions.reshape(actions.shape[0], -1)


def _base_ref_actions(
    ref_chunk: torch.Tensor,
    *,
    chunk_len: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if ref_chunk.dim() == 2:
        ref_chunk = ref_chunk.reshape(ref_chunk.shape[0], -1, action_dim)
    return ref_chunk[:, :chunk_len, :action_dim].to(device=device, dtype=dtype)


def reference_tracking_actions(
    current_actions: torch.Tensor,
    ref_actions: torch.Tensor,
    residuals: torch.Tensor,
    max_step_delta: float | list[float] | tuple[float, ...],
) -> torch.Tensor:
    """Convert residuals into recursively reference-tracked absolute actions.

    The first step starts from the measured/current target. Each following step
    starts from the action produced by the previous iteration, so a residual is
    never applied to a stale absolute joint target.

    Args:
        current_actions: Current absolute actions, shaped ``[B, action_dim]``.
        ref_actions: Absolute reference chunk, shaped ``[B, T, action_dim]``.
        residuals: Residual policy output, shaped like ``ref_actions``.
        max_step_delta: Per-dimension maximum physical step in the same domain.

    Returns:
        Recursively tracked absolute actions with shape ``[B, T, action_dim]``.
    """
    if current_actions.ndim != 2 or ref_actions.ndim != 3:
        raise ValueError("current_actions must be [B,D] and chunks must be [B,T,D]")
    if ref_actions.shape != residuals.shape:
        raise ValueError("ref_actions and residuals must have the same shape")
    if current_actions.shape[0] != ref_actions.shape[0]:
        raise ValueError("current_actions and action chunks must have matching batch")
    if current_actions.shape[-1] != ref_actions.shape[-1]:
        raise ValueError("current_actions and action chunks must have matching dims")

    max_delta = torch.as_tensor(
        max_step_delta, device=ref_actions.device, dtype=ref_actions.dtype
    )
    if max_delta.numel() == 1:
        max_delta = max_delta.expand(ref_actions.shape[-1])
    if max_delta.numel() != ref_actions.shape[-1] or (max_delta < 0).any():
        raise ValueError("max_step_delta must be non-negative and match action_dim")

    executed = []
    q_exec = current_actions.to(device=ref_actions.device, dtype=ref_actions.dtype)
    for step in range(ref_actions.shape[1]):
        desired_delta = ref_actions[:, step] - q_exec + residuals[:, step]
        q_exec = q_exec + desired_delta.clamp(-max_delta, max_delta)
        executed.append(q_exec)
    return torch.stack(executed, dim=1)


def apply_reference_tracking(
    current_actions: torch.Tensor,
    ref_actions: torch.Tensor,
    actor_actions: torch.Tensor,
    *,
    mode: str,
    max_step_delta: float | list[float] | tuple[float, ...],
    residual_scale: float | list[float] | tuple[float, ...] = 1.0,
) -> torch.Tensor:
    """Apply tracking to either legacy absolute or residual actor outputs.

    ``absolute_compat`` converts legacy absolute actor targets to residuals,
    allowing old joint checkpoints to be used with the recursive controller.
    ``residual`` treats actor outputs as bounded residuals and scales them
    before execution.
    """
    if mode == "absolute_compat":
        residuals = actor_actions - ref_actions
    elif mode == "residual":
        scale = torch.as_tensor(
            residual_scale, device=actor_actions.device, dtype=actor_actions.dtype
        )
        if scale.numel() == 1:
            scale = scale.expand(actor_actions.shape[-1])
        if scale.numel() != actor_actions.shape[-1] or (scale < 0).any():
            raise ValueError("residual_scale must be non-negative and match action_dim")
        residuals = actor_actions * scale
    else:
        raise ValueError(
            "reference tracking mode must be 'absolute_compat' or 'residual'"
        )
    return reference_tracking_actions(
        current_actions, ref_actions, residuals, max_step_delta
    )


def smooth_reference_residual_actions(
    actions: torch.Tensor,
    ref_actions: torch.Tensor,
    *,
    max_abs_residual: float | list[float] | tuple[float, ...],
    max_residual_step: float | list[float] | tuple[float, ...],
    filter_kernel: list[float] | tuple[float, ...],
    boundary_blend_steps: int = 0,
) -> torch.Tensor:
    """Smooth an absolute action chunk as a bounded reference residual.

    The complete chunk is available before execution, so a centered FIR filter
    can remove frame-to-frame residual oscillation without introducing the lag
    of a causal EMA. A cosine bridge optionally preserves the original handoff
    action and fades its offset out over the leading chunk steps.

    A zero residual limit pins that action dimension to the reference. Zadai
    uses this for both grippers while smoothing the 14 arm joints.
    """
    if actions.ndim != 3 or actions.shape != ref_actions.shape:
        raise ValueError("actions and ref_actions must have matching [B,T,D] shape")
    action_dim = actions.shape[-1]

    def _limit_tensor(value, name: str) -> torch.Tensor:
        limit = torch.as_tensor(value, device=actions.device, dtype=actions.dtype)
        if limit.numel() == 1:
            limit = limit.expand(action_dim)
        if limit.numel() != action_dim or (limit < 0).any():
            raise ValueError(f"{name} must be non-negative and match action_dim")
        return limit

    amplitude = _limit_tensor(max_abs_residual, "max_abs_residual")
    residual_step = _limit_tensor(max_residual_step, "max_residual_step")
    kernel = torch.as_tensor(
        filter_kernel, device=actions.device, dtype=actions.dtype
    ).flatten()
    if kernel.numel() == 0 or kernel.numel() % 2 == 0:
        raise ValueError("filter_kernel must contain a non-empty odd number of values")
    if (kernel < 0).any() or kernel.sum() <= 0:
        raise ValueError("filter_kernel must be non-negative with a positive sum")
    if boundary_blend_steps < 0:
        raise ValueError("boundary_blend_steps must be non-negative")
    kernel = kernel / kernel.sum()

    residual = (actions - ref_actions).clamp(-amplitude, amplitude)
    radius = kernel.numel() // 2
    padded = F.pad(
        residual.transpose(1, 2),
        (radius, radius),
        mode="replicate",
    ).transpose(1, 2)
    filtered = sum(
        kernel[index] * padded[:, index : index + actions.shape[1]]
        for index in range(kernel.numel())
    )

    limited = [filtered[:, 0]]
    for step in range(1, actions.shape[1]):
        delta = (filtered[:, step] - limited[-1]).clamp(-residual_step, residual_step)
        limited.append(limited[-1] + delta)
    smoothed_residual = torch.stack(limited, dim=1)
    smoothed_actions = ref_actions + smoothed_residual

    if boundary_blend_steps > 0 and actions.shape[1] > 1:
        blend_steps = min(boundary_blend_steps, actions.shape[1] - 1)
        phase = torch.arange(
            blend_steps + 1, device=actions.device, dtype=actions.dtype
        ) / float(blend_steps)
        weights = 0.5 * (1.0 + torch.cos(torch.pi * phase))
        # Dimensions with a zero amplitude limit stay exactly on the reference.
        active_dims = (amplitude > 0).to(dtype=actions.dtype)
        handoff_offset = (actions[:, :1] - smoothed_actions[:, :1]) * active_dims.view(
            1, 1, -1
        )
        prefix = smoothed_actions[:, : blend_steps + 1] + (
            weights.view(1, -1, 1) * handoff_offset
        )
        smoothed_actions = torch.cat(
            (prefix, smoothed_actions[:, blend_steps + 1 :]), dim=1
        )

    bounded_actions = smoothed_actions.clamp(-1.0, 1.0)
    pinned_dims = amplitude == 0
    return torch.where(pinned_dims.view(1, 1, -1), ref_actions, bounded_actions)


def apply_reference_residual_smoothing(
    actions: torch.Tensor,
    ref_actions: torch.Tensor,
    smoothing_cfg: Any | None,
) -> torch.Tensor:
    """Apply configured residual smoothing, or return actions unchanged."""
    if not smoothing_cfg or not bool(smoothing_cfg.get("enable", False)):
        return actions
    return smooth_reference_residual_actions(
        actions,
        ref_actions,
        max_abs_residual=smoothing_cfg.get("max_abs_residual", 0.015),
        max_residual_step=smoothing_cfg.get("max_residual_step", 0.003),
        filter_kernel=smoothing_cfg.get("filter_kernel", [1.0, 2.0, 3.0, 2.0, 1.0]),
        boundary_blend_steps=int(smoothing_cfg.get("boundary_blend_steps", 10)),
    )


def apply_handoff_blend(
    reference_actions: torch.Tensor,
    actor_actions: torch.Tensor,
    current_actions: torch.Tensor,
    enter_mask: torch.Tensor,
    exit_mask: torch.Tensor,
    *,
    enter_steps: int,
    exit_steps: int,
    max_step_delta: float | list[float] | tuple[float, ...],
    pinned_dims: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend only on OpenPI/RLT edges, anchored at measured current joints."""
    if not enter_mask.any() and not exit_mask.any():
        return actor_actions
    _, chunk_len, action_dim = actor_actions.shape
    current = current_actions[:, None, :].to(
        device=actor_actions.device, dtype=actor_actions.dtype
    )
    if pinned_dims is not None:
        pinned = pinned_dims.to(device=actor_actions.device, dtype=torch.bool)
        current = torch.where(pinned.view(1, 1, -1), reference_actions[:, :1], current)
    max_delta = torch.as_tensor(
        max_step_delta, device=actor_actions.device, dtype=actor_actions.dtype
    )
    if max_delta.numel() == 1:
        max_delta = max_delta.expand(action_dim)
    if max_delta.numel() != action_dim or (max_delta < 0).any():
        raise ValueError("handoff max_step_delta must match action_dim")

    result = torch.where(enter_mask[:, None, None], actor_actions, reference_actions)
    active = enter_mask | exit_mask
    max_steps = max(1, int(max(enter_steps, exit_steps)))
    previous = current[:, 0]
    # Include one extra sample at the end of the blend so the transition from
    # the blended prefix to the normal chunk is also step-limited.
    for step in range(min(chunk_len, max_steps + 1)):
        if step == 0:
            target = current[:, 0]
        else:
            enter_alpha = min(1.0, step / max(1, enter_steps - 1))
            exit_alpha = min(1.0, step / max(1, exit_steps - 1))
            enter_target = reference_actions[:, step] + enter_alpha * (
                actor_actions[:, step] - reference_actions[:, step]
            )
            exit_target = current[:, 0] + exit_alpha * (
                reference_actions[:, step] - current[:, 0]
            )
            target = torch.where(enter_mask[:, None], enter_target, exit_target)
        target = previous + (target - previous).clamp(-max_delta, max_delta)
        target = torch.where(active[:, None], target, result[:, step])
        result[:, step] = target
        previous = target
    return result.clamp(-1.0, 1.0)


def project_actions_to_reference(
    actions: torch.Tensor,
    ref_actions: torch.Tensor,
    max_delta: float | list[float] | tuple[float, ...] | None,
    constrained_steps: int | None = None,
    action_space: str | None = None,
) -> torch.Tensor:
    """Project full policy actions into a box around reference actions.

    Args:
        actions: Full action candidates shaped ``[..., action_dim]``.
        ref_actions: Reference actions with the same shape as ``actions``.
        max_delta: Non-negative normalized delta per action dimension, or one
            scalar shared by all dimensions. ``None`` disables projection.
        constrained_steps: Number of leading chunk steps to project. ``None``
            projects the complete chunk.

    Returns:
        Actions bounded around the reference and inside ``[-1, 1]``.
    """
    if max_delta is None:
        projected = actions
    else:
        projected = None
    if constrained_steps is not None and constrained_steps < 0:
        raise ValueError("RLT constrained_steps must be non-negative or None")
    if max_delta is None:
        delta = None
    else:
        delta = torch.as_tensor(max_delta, device=actions.device, dtype=actions.dtype)
    if delta is not None and delta.numel() == 1:
        delta = delta.expand(actions.shape[-1])
    if delta is not None and delta.numel() != actions.shape[-1]:
        raise ValueError(
            "RLT actor reference limit must be scalar or match action_dim, got "
            f"{delta.numel()} values for action_dim={actions.shape[-1]}."
        )
    if delta is not None and (delta < 0).any():
        raise ValueError("RLT actor reference limits must be non-negative")
    steps = (
        actions.shape[-2]
        if constrained_steps is None
        else min(constrained_steps, actions.shape[-2])
    )
    if steps == 0:
        projected = actions
    elif delta is not None:
        projected_prefix = torch.maximum(
            torch.minimum(actions[..., :steps, :], ref_actions[..., :steps, :] + delta),
            ref_actions[..., :steps, :] - delta,
        ).clamp(-1.0, 1.0)
        projected = (
            projected_prefix
            if steps == actions.shape[-2]
            else torch.cat((projected_prefix, actions[..., steps:, :]), dim=-2)
        )
    if action_space == "eef":
        if projected.shape[-1] != 20:
            raise ValueError("EEF reference projection requires action_dim=20")
        from rlinf.envs.realworld.zadai.action_codec import canonicalize_rot6d

        projected = projected.clone()
        projected[..., 3:9] = canonicalize_rot6d(projected[..., 3:9])
        projected[..., 13:19] = canonicalize_rot6d(projected[..., 13:19])
    return projected


class RLTRoute(ABC):
    @abstractmethod
    def route(self, ctx: RLTRouteContext) -> RLTRouteOutput:
        """Route student actions among actor, reference, and optional expert."""


class RealworldRLTRoute(RLTRoute):
    """Actor/ref routing for realworld RLT (keyboard or env switch flags)."""

    def __init__(
        self,
        *,
        use_schedule: bool,
        warmup_updates: int,
        actor_reference_max_delta: float | list[float] | None = None,
        actor_reference_constrained_steps: int | None = None,
        actor_action_space: str | None = None,
        reference_tracking_mode: str | None = None,
        reference_tracking_max_step_delta: float | list[float] | None = None,
        reference_tracking_residual_scale: float | list[float] = 1.0,
        reference_residual_smoothing: Any | None = None,
        handoff_enable: bool = True,
        handoff_enter_steps: int = 10,
        handoff_exit_steps: int = 10,
        handoff_max_step_delta: float | list[float] | None = None,
    ):
        self.use_schedule = use_schedule
        self.warmup_updates = warmup_updates
        self.actor_reference_max_delta = actor_reference_max_delta
        self.actor_reference_constrained_steps = actor_reference_constrained_steps
        self.actor_action_space = actor_action_space
        self.reference_tracking_mode = reference_tracking_mode
        self.reference_tracking_max_step_delta = reference_tracking_max_step_delta
        self.reference_tracking_residual_scale = reference_tracking_residual_scale
        self.reference_residual_smoothing = reference_residual_smoothing
        self.handoff_enable = bool(handoff_enable)
        self.handoff_enter_steps = max(1, int(handoff_enter_steps))
        self.handoff_exit_steps = max(1, int(handoff_exit_steps))
        self.handoff_max_step_delta = handoff_max_step_delta
        self._route_sequence = 0
        self._previous_routed_last: torch.Tensor | None = None
        self._previous_actor_switch: torch.Tensor | None = None

    @staticmethod
    def _route_log_tensor(value: torch.Tensor | None):
        if value is None or not torch.is_tensor(value):
            return None
        return value.detach().float().cpu().tolist()

    def _ready_for_online(self, version: int) -> bool:
        return not self.use_schedule or int(version) >= self.warmup_updates

    def route(self, ctx: RLTRouteContext) -> RLTRouteOutput:
        self._route_sequence += 1
        actions = ctx.student_actions
        result = ctx.result
        critical_phase = _last_info_bool(
            ctx.rlt_switch_flags,
            batch_size=actions.shape[0],
            device=actions.device,
            default=ctx.default_actor_switch,
        )
        learner_ready = torch.full_like(
            critical_phase,
            self._ready_for_online(ctx.version),
        )
        actor_switch = critical_phase & learner_ready
        previous_actor_switch = self._previous_actor_switch
        if previous_actor_switch is None or previous_actor_switch.shape != actor_switch.shape:
            previous_actor_switch = torch.zeros_like(actor_switch)
        enter_handoff = actor_switch & ~previous_actor_switch
        exit_handoff = ~actor_switch & previous_actor_switch
        ref_actions = result["forward_inputs"]["ref_chunk"].to(
            device=actions.device, dtype=actions.dtype
        )
        constrained_actor_actions = project_actions_to_reference(
            actions,
            ref_actions[:, : actions.shape[1], : actions.shape[2]],
            self.actor_reference_max_delta,
            self.actor_reference_constrained_steps,
            self.actor_action_space,
        )
        if self.reference_tracking_mode is not None:
            if self.actor_action_space != "joint":
                raise ValueError("reference tracking is currently supported for joint")
            if self.reference_tracking_max_step_delta is None:
                raise ValueError("reference tracking requires max_step_delta")
            from rlinf.envs.realworld.zadai.action_codec import absolute_to_policy

            current = absolute_to_policy(ctx.rlt_obs["proprio"], "joint")
            constrained_actor_actions = apply_reference_tracking(
                current,
                ref_actions[:, : actions.shape[1], : actions.shape[2]],
                actions,
                mode=self.reference_tracking_mode,
                max_step_delta=self.reference_tracking_max_step_delta,
                residual_scale=self.reference_tracking_residual_scale,
            )
        constrained_actor_actions = apply_reference_residual_smoothing(
            constrained_actor_actions,
            ref_actions[:, : actions.shape[1], : actions.shape[2]],
            self.reference_residual_smoothing,
        )
        # Tracking/smoothing can move the constrained action again; reapply
        # the explicit reference bound last.
        constrained_actor_actions = project_actions_to_reference(
            constrained_actor_actions,
            ref_actions[:, : actions.shape[1], : actions.shape[2]],
            self.actor_reference_max_delta,
            self.actor_reference_constrained_steps,
            self.actor_action_space,
        )
        if self.handoff_enable and self.actor_action_space == "joint":
            current = ctx.rlt_obs.get("proprio")
            if current is not None and self.handoff_max_step_delta is not None:
                from rlinf.envs.realworld.zadai.action_codec import absolute_to_policy

                current = absolute_to_policy(current, "joint")
                pinned_dims = None
                if self.actor_reference_max_delta is not None:
                    pinned_dims = (
                        torch.as_tensor(
                            self.actor_reference_max_delta,
                            device=actions.device,
                            dtype=torch.bool,
                        )
                        == 0
                    )
                constrained_actor_actions = apply_handoff_blend(
                    ref_actions[:, : actions.shape[1], : actions.shape[2]],
                    constrained_actor_actions,
                    current,
                    enter_handoff,
                    exit_handoff,
                    enter_steps=self.handoff_enter_steps,
                    exit_steps=self.handoff_exit_steps,
                    max_step_delta=self.handoff_max_step_delta,
                    pinned_dims=pinned_dims,
                )
        routed_actions = torch.where(
            actor_switch[:, None, None],
            constrained_actor_actions,
            ref_actions[:, : actions.shape[1], : actions.shape[2]],
        ).contiguous()
        result["forward_inputs"]["action"] = routed_actions.reshape(
            routed_actions.shape[0], -1
        ).contiguous()
        result["forward_inputs"]["record_transition"] = critical_phase[:, None]
        result["forward_inputs"]["actor_switch"] = actor_switch[:, None]
        # Keep a compact, human-readable handoff trace in the same JSONL file
        # as the environment diagnostics.  Full chunks remain in raw archive;
        # this log focuses on the values needed to diagnose switching jumps.
        route_log_path = os.environ.get("ZADAI_RLT_ROUTE_LOG") or os.environ.get(
            "ZADAI_RLT_RUNTIME_LOG"
        )
        if route_log_path:
            ref_first = ref_actions[:, 0]
            actor_first = constrained_actor_actions[:, 0]
            routed_first = routed_actions[:, 0]
            current = ctx.rlt_obs.get("proprio")
            previous_last = self._previous_routed_last
            if previous_last is not None and previous_last.shape != routed_first.shape:
                previous_last = None
            append_runtime_event(
                route_log_path,
                {
                    "event": "rlt_route_action",
                    "route_sequence": self._route_sequence,
                    "version": int(ctx.version),
                    "actor_switch": actor_switch,
                    "critical_phase": critical_phase,
                    "previous_actor_switch": previous_actor_switch,
                    "enter_handoff": enter_handoff,
                    "exit_handoff": exit_handoff,
                    "handoff_enter_steps": self.handoff_enter_steps,
                    "handoff_exit_steps": self.handoff_exit_steps,
                    "handoff_max_step_delta": self.handoff_max_step_delta,
                    "current_proprio_normalized": self._route_log_tensor(current),
                    "actor_raw_first": self._route_log_tensor(actions[:, 0]),
                    "actor_constrained_first": self._route_log_tensor(actor_first),
                    "reference_first": self._route_log_tensor(ref_first),
                    "routed_first": self._route_log_tensor(routed_first),
                    "routed_last": self._route_log_tensor(routed_actions[:, -1]),
                    "previous_routed_last": self._route_log_tensor(previous_last),
                    "previous_to_routed_first_max": (
                        torch.amax(torch.abs(routed_first - previous_last), dim=-1)
                        if previous_last is not None
                        else None
                    ),
                    "actor_minus_reference_first_max": torch.max(
                        torch.abs(actor_first - ref_first), dim=-1
                    ).values,
                    "routed_minus_reference_first_max": torch.max(
                        torch.abs(routed_first - ref_first), dim=-1
                    ).values,
                    "routed_chunk_max_abs": torch.amax(
                        torch.abs(routed_actions), dim=(-2, -1)
                    ),
                    "switch_flags": self._route_log_tensor(ctx.rlt_switch_flags),
                },
            )
        self._previous_routed_last = routed_actions[:, -1].detach()
        self._previous_actor_switch = actor_switch.detach().clone()
        return RLTRouteOutput(actions=routed_actions, result=result)


class SimulatorRLTRoute(RLTRoute):
    """Actor/ref/expert routing for ManiSkill RLT with schedule warmup."""

    def __init__(self, *, use_schedule: bool, warmup_updates: int):
        self.use_schedule = use_schedule
        self.warmup_updates = warmup_updates

    def _ready_for_online(self, version: int) -> bool:
        return not self.use_schedule or int(version) >= self.warmup_updates

    def route(self, ctx: RLTRouteContext) -> RLTRouteOutput:
        actions = ctx.student_actions
        result = ctx.result
        batch_size, chunk_len, action_dim = actions.shape
        ready_for_online = self._ready_for_online(ctx.version)

        critical_phase = _last_info_bool(
            ctx.rlt_switch_flags,
            batch_size=batch_size,
            device=actions.device,
            default=False,
        )
        actor_switch = critical_phase
        if self.use_schedule:
            actor_switch = actor_switch & torch.full(
                (batch_size,),
                ready_for_online,
                dtype=torch.bool,
                device=actions.device,
            )

        requested_expert_takeover = _last_info_bool(
            ctx.intervene_requested,
            batch_size=batch_size,
            device=actions.device,
            default=False,
        )
        expert_takeover = (
            requested_expert_takeover
            & ready_for_online
            & (ctx.mode == "train")
            & (ctx.expert_model is not None)
        )

        base_actions = _base_ref_actions(
            ctx.rlt_obs["ref_chunk"],
            chunk_len=chunk_len,
            action_dim=action_dim,
            device=actions.device,
            dtype=actions.dtype,
        )
        routed_actions = torch.where(
            actor_switch[:, None, None],
            actions,
            base_actions,
        ).contiguous()

        intervene_flags = torch.zeros(
            (batch_size, chunk_len),
            dtype=torch.bool,
            device=actions.device,
        )
        if expert_takeover.any():
            if ctx.expert_model is None:
                raise RuntimeError(
                    "ManiSkill RLT expert takeover was requested, but expert_model "
                    "is not configured."
                )
            expert_actions = predict_expert_actions(
                ctx.expert_model,
                ctx.env_obs,
                chunk_len=chunk_len,
                action_dim=action_dim,
                device=actions.device,
                dtype=actions.dtype,
            )
            routed_actions = torch.where(
                expert_takeover[:, None, None],
                expert_actions,
                routed_actions,
            ).contiguous()
            intervene_flags[expert_takeover] = True

        forward_inputs = result["forward_inputs"]
        forward_inputs["action"] = _flatten_action_chunk(routed_actions).detach()
        forward_inputs["record_transition"] = critical_phase[:, None]
        forward_inputs["actor_switch"] = (actor_switch & ~expert_takeover)[:, None]
        forward_inputs["intervention_requested"] = requested_expert_takeover[:, None]
        result["intervene_flags"] = intervene_flags
        return RLTRouteOutput(actions=routed_actions, result=result)


def build_rlt_route(cfg: Any) -> RLTRoute:
    schedule_cfg = cfg.algorithm.get("rlt_schedule", {}) or {}
    route_kwargs = {
        "use_schedule": bool(schedule_cfg.get("enable", False)),
        "warmup_updates": int(schedule_cfg.get("warmup_post_collect_updates", 0)),
    }
    if use_simulator_transition_replay(cfg):
        return SimulatorRLTRoute(**route_kwargs)
    constraint_cfg = cfg.algorithm.get("actor_action_constraint", {}) or {}
    route_kwargs["actor_reference_max_delta"] = (
        constraint_cfg.get("max_reference_delta")
        if bool(constraint_cfg.get("enable", False))
        else None
    )
    route_kwargs["actor_reference_constrained_steps"] = constraint_cfg.get(
        "constrained_steps"
    )
    route_kwargs["actor_action_space"] = constraint_cfg.get("action_space")
    route_kwargs["reference_tracking_mode"] = (
        constraint_cfg.get("tracking_mode", None)
        if bool(constraint_cfg.get("enable", False))
        else None
    )
    route_kwargs["reference_tracking_max_step_delta"] = constraint_cfg.get(
        "tracking_max_step_delta"
    )
    route_kwargs["reference_tracking_residual_scale"] = constraint_cfg.get(
        "residual_scale", 1.0
    )
    route_kwargs["reference_residual_smoothing"] = constraint_cfg.get(
        "residual_smoothing", None
    )
    handoff_cfg = constraint_cfg.get("handoff", {}) or {}
    route_kwargs["handoff_enable"] = bool(handoff_cfg.get("enable", True))
    route_kwargs["handoff_enter_steps"] = int(handoff_cfg.get("enter_steps", 10))
    route_kwargs["handoff_exit_steps"] = int(handoff_cfg.get("exit_steps", 10))
    route_kwargs["handoff_max_step_delta"] = handoff_cfg.get(
        "max_step_delta", None
    )
    return RealworldRLTRoute(**route_kwargs)
