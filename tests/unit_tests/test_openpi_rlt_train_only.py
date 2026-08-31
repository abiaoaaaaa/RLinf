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

import torch
from torch import nn

from rlinf.models.embodiment.modules.rlt_token_transformer import (
    freeze_non_rlt_parameters,
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
