# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""OpenPI transforms for the Zadai dual-arm absolute-joint policy."""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BimanualJointInputs(transforms.DataTransformFn):
    """Map raw 16-D dual-arm observations/actions into OpenPI fields."""

    action_dim: int
    model_type: _model.ModelType
    default_prompt: str

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": np.asarray(data["observation/state"]),
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
            inputs["actions"] = np.asarray(data["actions"])
        return inputs


@dataclasses.dataclass(frozen=True)
class BimanualJointOutputs(transforms.DataTransformFn):
    """Crop padded OpenPI actions back to the 16-D robot interface."""

    output_action_dim: int = 16

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}
