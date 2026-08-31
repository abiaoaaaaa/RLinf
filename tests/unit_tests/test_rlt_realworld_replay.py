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

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.route import (
    RealworldRLTRoute,
    RLTRouteContext,
    project_actions_to_reference,
)
from rlinf.algorithms.rlt.transition import update_rlt_transitions
from rlinf.data.embodied_io_struct import EmbodiedRolloutResult, Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.workers.actor import fsdp_rlt_ac_policy_worker as rlt_worker


def _route_result(ref_chunk: torch.Tensor) -> dict:
    return {"forward_inputs": {"ref_chunk": ref_chunk.clone()}}


def test_realworld_route_records_critical_phase_before_actor_is_ready():
    route = RealworldRLTRoute(use_schedule=True, warmup_updates=3)
    student = torch.zeros((2, 2, 1))
    reference = torch.ones_like(student)
    switch_flags = torch.tensor([[True], [False]])

    warmup = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs={},
            student_actions=student,
            result=_route_result(reference),
            mode="train",
            rlt_switch_flags=switch_flags,
            version=2,
        )
    )
    assert torch.equal(warmup.actions, reference)
    assert torch.equal(
        warmup.result["forward_inputs"]["record_transition"], switch_flags
    )
    assert not warmup.result["forward_inputs"]["actor_switch"].any()

    online = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs={},
            student_actions=student,
            result=_route_result(reference),
            mode="train",
            rlt_switch_flags=switch_flags,
            version=3,
        )
    )
    assert torch.equal(online.actions[0], student[0])
    assert torch.equal(online.actions[1], reference[1])
    assert torch.equal(online.result["forward_inputs"]["actor_switch"], switch_flags)


def test_realworld_route_limits_online_actor_around_reference():
    route = RealworldRLTRoute(
        use_schedule=False,
        warmup_updates=0,
        actor_reference_max_delta=[0.02, 0.0],
        actor_reference_constrained_steps=1,
    )
    reference = torch.tensor([[[0.5, -0.5], [0.4, -0.4]]])
    student = torch.tensor([[[-1.0, 1.0], [1.0, -1.0]]])

    output = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs={},
            student_actions=student,
            result=_route_result(reference),
            mode="train",
            rlt_switch_flags=torch.tensor([[True]]),
        )
    )

    expected = torch.tensor([[[0.48, -0.5], [1.0, -1.0]]])
    torch.testing.assert_close(output.actions, expected)
    torch.testing.assert_close(
        output.result["forward_inputs"]["action"], expected.reshape(1, -1)
    )


def test_eef_reference_projection_returns_valid_rot6d():
    actions = torch.zeros((1, 2, 20))
    reference = torch.zeros_like(actions)
    reference[..., 3] = 1.0
    reference[..., 7] = 1.0
    reference[..., 13] = 1.0
    reference[..., 17] = 1.0

    projected = project_actions_to_reference(
        actions, reference, 0.1, action_space="eef"
    )

    for start in (3, 13):
        first = projected[..., start : start + 3]
        second = projected[..., start + 3 : start + 6]
        torch.testing.assert_close(
            torch.linalg.vector_norm(first, dim=-1), torch.ones((1, 2))
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(second, dim=-1), torch.ones((1, 2))
        )
        torch.testing.assert_close((first * second).sum(dim=-1), torch.zeros((1, 2)))


def test_rollout_replay_action_can_be_replaced_by_sent_action():
    rollout = EmbodiedRolloutResult(max_episode_length=1)
    rollout.actions.append(torch.zeros((1, 4)))
    rollout.intervene_flags.append(torch.zeros((1, 4), dtype=torch.bool))
    rollout.forward_inputs.append({"action": torch.zeros((1, 4))})
    executed = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])

    rollout.replace_last_actions(executed)

    expected = executed.reshape(1, 4)
    torch.testing.assert_close(rollout.actions[-1], expected)
    torch.testing.assert_close(rollout.forward_inputs[-1]["action"], expected)


def test_human_action_overrides_sent_action_only_at_intervention_steps():
    rollout = EmbodiedRolloutResult(max_episode_length=1)
    rollout.actions.append(torch.zeros((1, 4)))
    rollout.intervene_flags.append(torch.zeros((1, 4), dtype=torch.bool))
    rollout.forward_inputs.append({"action": torch.zeros((1, 4))})
    sent = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
    human = torch.tensor([[[-0.1, -0.2], [-0.3, -0.4]]]).reshape(1, 4)
    flags = torch.tensor([[False, True]])

    rollout.replace_last_actions(sent)
    rollout.update_last_actions(human, flags)

    expected = torch.tensor([[0.1, 0.2, -0.3, -0.4]])
    torch.testing.assert_close(rollout.actions[-1], expected)
    torch.testing.assert_close(rollout.forward_inputs[-1]["action"], expected)


class _ReplayBuffer:
    _flatten_trajectory = TrajectoryReplayBuffer._flatten_trajectory

    def __init__(self):
        self.added: list[Trajectory] = []

    def add_trajectories(self, trajectories: list[Trajectory]):
        self.added.extend(trajectories)


class _ReplayHarness(rlt_worker.RLTACReplayMixin):
    def __init__(self):
        self.replay_buffer = _ReplayBuffer()
        self.demo_buffer = None
        self.cfg = SimpleNamespace(env=SimpleNamespace(train={"auto_reset": False}))


def test_realworld_replay_ingest_keeps_only_recorded_transitions(monkeypatch):
    monkeypatch.setattr(
        rlt_worker,
        "collect_trajectory_replay_metrics",
        lambda trajectories, reducer=None: {},
    )
    trajectory = Trajectory(
        actions=torch.tensor([[[0.0]], [[1.0]], [[2.0]]]),
        intervene_flags=torch.zeros((3, 1, 1), dtype=torch.bool),
        rewards=torch.zeros((3, 1, 1)),
        forward_inputs={
            "record_transition": torch.tensor([[[False]], [[True]], [[False]]])
        },
        curr_obs={
            "z_rl": torch.tensor([[[0.0]], [[1.0]], [[2.0]]]),
            "proprio": torch.tensor([[[0.0]], [[1.0]], [[2.0]]]),
            "ref_chunk": torch.tensor([[[0.0]], [[1.0]], [[2.0]]]),
        },
        next_obs={
            "z_rl": torch.tensor([[[1.0]], [[2.0]], [[3.0]]]),
            "proprio": torch.tensor([[[1.0]], [[2.0]], [[3.0]]]),
            "ref_chunk": torch.tensor([[[1.0]], [[2.0]], [[3.0]]]),
        },
    )
    harness = _ReplayHarness()

    added, completed = harness._ingest_rollout_trajectories([trajectory])

    assert (added, completed) == (1, 0)
    assert len(harness.replay_buffer.added) == 1
    assert harness.replay_buffer.added[0].actions.item() == 1.0
    assert harness.replay_buffer.added[0].next_obs["z_rl"].item() == 2.0


class _TransitionCollector:
    def __init__(self):
        self.transitions = []

    def append_transitions(self, curr_obs, next_obs):
        self.transitions.append((curr_obs, next_obs))


def test_final_bootstrap_row_is_not_recorded_as_executed_transition():
    current_obs = {
        "z_rl": torch.tensor([[1.0]]),
        "proprio": torch.tensor([[2.0]]),
        "ref_chunk": torch.tensor([[[3.0]]]),
    }
    bootstrap_result = SimpleNamespace(
        forward_inputs={
            "record_transition": torch.tensor([[True]]),
            "rlt_transition_z_rl": torch.tensor([[4.0]]),
            "rlt_transition_proprio": torch.tensor([[5.0]]),
            "rlt_transition_ref_chunk": torch.tensor([[[6.0]]]),
        }
    )
    collector = _TransitionCollector()
    pending_obs = [current_obs]

    update_rlt_transitions(
        0,
        pending_obs,
        [collector],
        bootstrap_result,
        cache_current=False,
    )

    assert len(collector.transitions) == 1
    assert pending_obs == [None]
    assert not bootstrap_result.forward_inputs["record_transition"].any()


def test_runner_step_schedule_adds_update_epoch_without_new_transitions():
    harness = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "algorithm": {
                    "update_epoch": 8,
                    "replay_buffer": {"min_buffer_size": 100},
                }
            }
        ),
        rlt_schedule_cfg={
            "warmup_min_size": 250,
            "warmup_post_collect_updates": 100,
            "train_every_runner_step": True,
            "train_every_transitions": 0,
            "train_every_episodes": 0,
            "max_updates_per_train_step": 20,
        },
        update_step=100,
        _warmup_ready_total_transitions=0,
        _warmup_ready_total_episodes=0,
        pending_update_budget=0,
        _global_rlt_counters=lambda: {
            "min_replay_size": 362,
            "min_demo_size": 0,
            "total_transitions_added": 82,
            "total_episodes_added": 0,
            "transitions_since_train": 0,
            "episodes_since_train": 0,
        },
    )

    updates, metrics = rlt_worker.RLTACFSDPPolicy._rlt_updates_to_run(harness)

    assert updates == 8
    assert metrics["rlt/desired_total_updates"] == 108
    assert metrics["rlt/train_every_runner_step"] == 1
