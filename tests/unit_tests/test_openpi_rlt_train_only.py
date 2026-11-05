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

import pytest
import torch
from torch import nn

from rlinf.models.embodiment.modules.rlt_token_transformer import (
    freeze_non_rlt_parameters,
)
from rlinf.models.embodiment.openpi.rtc_guidance import (
    RTCGuidanceContext,
    build_rtc_target_and_mask,
    select_rtc_action_window,
)


class _TinyRLTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.paligemma_with_expert = nn.Linear(2, 2)
        self.rlt_module = nn.Linear(2, 2)


def test_freeze_for_rlt_stage1_only_leaves_compressor_trainable():
    model = _TinyRLTModel()

    trainable_params = freeze_non_rlt_parameters(model, model.rlt_module)

    assert trainable_params == sum(param.numel() for param in model.rlt_module.parameters())
    assert all(param.requires_grad for param in model.rlt_module.parameters())
    assert all(not param.requires_grad for param in model.backbone.parameters())
    assert all(not param.requires_grad for param in model.paligemma_with_expert.parameters())

    model.rlt_module(torch.ones(1, 2)).sum().backward()
    assert all(param.grad is not None for param in model.rlt_module.parameters())
    assert all(param.grad is None for param in model.backbone.parameters())


def test_rtc_skips_stale_prefix_and_keeps_full_execution_chunk():
    full_plan = torch.arange(50, dtype=torch.float32).reshape(1, 50, 1)

    selected = select_rtc_action_window(full_plan, start=10, length=30)

    torch.testing.assert_close(selected, full_plan[:, 10:40])
    with pytest.raises(ValueError, match="must fit the model horizon"):
        select_rtc_action_window(full_plan, start=25, length=30)


def test_rtc_overlap_target_uses_unexecuted_full_plan_tail():
    previous = torch.arange(50, dtype=torch.float32).reshape(1, 50, 1)
    context = RTCGuidanceContext(
        prev_model_actions=previous,
        executed_horizon=30,
        delay_steps=10,
    )

    target, mask = build_rtc_target_and_mask(
        context.get_prev_remaining(),
        horizon=50,
        action_dim=1,
        delay_steps=context.delay_steps,
        device=previous.device,
        dtype=previous.dtype,
    )

    torch.testing.assert_close(target[:, :20], previous[:, 30:])
    assert mask[:, :10].eq(1).all()
    assert mask[:, 10:20].gt(0).all()
    assert mask[:, 20:].eq(0).all()
