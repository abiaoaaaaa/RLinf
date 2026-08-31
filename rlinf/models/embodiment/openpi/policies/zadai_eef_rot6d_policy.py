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
"""OpenPI transforms for the Zadai bilateral absolute-EEF dataset."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

EEF_ACTION_DIM = 20


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ZadaiEefRot6dInputs(transforms.DataTransformFn):
    """Map the 20-D EEF state/action and three Zadai cameras to OpenPI."""

    action_dim: int
    model_type: _model.ModelType
    default_prompt: str

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"])
        if state.shape[-1] != EEF_ACTION_DIM:
            raise ValueError(f"expected 20-D Zadai EEF state, got shape {state.shape}")
        inputs = {
            "state": transforms.pad_to_dim(state, self.action_dim),
            "image": {
                "base_0_rgb": _parse_image(data["observation/top"]),
                "left_wrist_0_rgb": _parse_image(data["observation/wrist_L"]),
                "right_wrist_0_rgb": _parse_image(data["observation/wrist_R"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "prompt": data.get("prompt", self.default_prompt),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != EEF_ACTION_DIM:
                raise ValueError(
                    f"expected 20-D Zadai EEF actions, got shape {actions.shape}"
                )
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        return inputs


@dataclasses.dataclass(frozen=True)
class ZadaiEefRot6dOutputs(transforms.DataTransformFn):
    """Crop padded OpenPI output back to the 20-D EEF robot action."""

    output_action_dim: int = EEF_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}
