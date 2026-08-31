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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch

from rlinf.algorithms.rlt.expert import predict_expert_actions
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay


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
    ):
        self.use_schedule = use_schedule
        self.warmup_updates = warmup_updates
        self.actor_reference_max_delta = actor_reference_max_delta
        self.actor_reference_constrained_steps = actor_reference_constrained_steps
        self.actor_action_space = actor_action_space

    def _ready_for_online(self, version: int) -> bool:
        return not self.use_schedule or int(version) >= self.warmup_updates

    def route(self, ctx: RLTRouteContext) -> RLTRouteOutput:
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
    return RealworldRLTRoute(**route_kwargs)
