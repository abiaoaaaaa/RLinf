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
"""OpenPI data config for the Zadai 20-D bilateral EEF rot6d dataset."""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import zadai_eef_rot6d_policy
from rlinf.models.embodiment.openpi.transforms import (
    DUAL_ARM_ROT6D_LAYOUT,
    RigidBodyAbsoluteActions,
    RigidBodyDeltaActions,
)

DEFAULT_PROMPT = "Insert the tail of the yellow cable tie into its head to fasten it."


@dataclasses.dataclass(frozen=True)
class LeRobotZadaiEefRot6dDataConfig(DataConfigFactory):
    """Repack Zadai cameras and learn body-frame SE(3) action deltas."""

    default_prompt: str = DEFAULT_PROMPT
    output_action_dim: int = 20
    extra_delta_transform: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/top": "observation.images.top",
                        "observation/wrist_L": "observation.images.wrist_L",
                        "observation/wrist_R": "observation.images.wrist_R",
                        "observation/state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                zadai_eef_rot6d_policy.ZadaiEefRot6dInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    default_prompt=self.default_prompt,
                )
            ],
            outputs=[
                zadai_eef_rot6d_policy.ZadaiEefRot6dOutputs(
                    output_action_dim=self.output_action_dim
                )
            ],
        )
        if self.extra_delta_transform:
            data_transforms = data_transforms.push(
                inputs=[RigidBodyDeltaActions(DUAL_ARM_ROT6D_LAYOUT)],
                outputs=[RigidBodyAbsoluteActions(DUAL_ARM_ROT6D_LAYOUT)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory(default_prompt=self.default_prompt)(
                model_config
            ),
            action_sequence_keys=("action",),
        )
