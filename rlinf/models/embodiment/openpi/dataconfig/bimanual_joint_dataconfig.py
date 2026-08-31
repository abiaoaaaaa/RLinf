# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LeRobot data config for 16-D dual-arm absolute joint targets."""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import bimanual_joint_policy

_DEFAULT_PROMPT = "Insert the tail of the yellow cable tie into its head to fasten it."


@dataclasses.dataclass(frozen=True)
class LeRobotBimanualJointDataConfig(DataConfigFactory):
    default_prompt: str = _DEFAULT_PROMPT
    output_action_dim: int = 16

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
                bimanual_joint_policy.BimanualJointInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    default_prompt=self.default_prompt,
                )
            ],
            outputs=[
                bimanual_joint_policy.BimanualJointOutputs(
                    output_action_dim=self.output_action_dim
                )
            ],
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
