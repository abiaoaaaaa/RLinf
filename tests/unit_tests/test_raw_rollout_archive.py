# Copyright 2025 The RLinf Authors.
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

import json

import cv2
import numpy as np
import torch

from rlinf.data.embodied_io_struct import EnvOutput, RolloutResult
from rlinf.data.raw_rollout_archive import RawRolloutArchive
from rlinf.workers.env.env_worker import EnvWorker


def _obs(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "states": torch.arange(16, dtype=torch.float32).reshape(1, 16),
        "main_images": torch.from_numpy(
            rng.integers(0, 256, size=(1, 12, 16, 3), dtype=np.uint8)
        ),
        "extra_view_images": torch.from_numpy(
            rng.integers(0, 256, size=(1, 2, 12, 16, 3), dtype=np.uint8)
        ),
        "task_descriptions": ["test task"],
    }


def test_raw_rollout_archive_writes_complete_chunk(tmp_path):
    archive = RawRolloutArchive(
        tmp_path,
        manifest={"action_dim": 16},
        jpeg_quality=90,
        queue_size=2,
    )
    archive.submit(
        {
            "runner_step": 7,
            "rollout_epoch": 1,
            "chunk_step": 3,
            "stage_id": 0,
            "observations": {"current": _obs(1), "next": _obs(2)},
            "actions": {
                "pi05_model_output": torch.zeros(1, 50, 32),
                "pi05_reference": torch.ones(1, 30, 16),
                "rlt_actor_raw": torch.full((1, 480), 2.0),
                "routed": torch.full((1, 30, 16), 3.0),
                "executed": torch.full((1, 30, 16), 4.0),
                "intervention": torch.full((1, 30, 16), 5.0),
            },
            "flags": {
                "rl_phase_requested": torch.tensor([[True]]),
                "rlt_actor_active": torch.tensor([[True]]),
                "intervention": torch.tensor([[False, True]]),
            },
            "outcome": {"rewards": torch.tensor([[1.0]])},
            "rtc": {"rtc_delay_steps": torch.tensor([[10]])},
            "versions": torch.tensor([[12]]),
        }
    )
    archive.close()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["format_version"] == 1
    index = json.loads((tmp_path / "index.jsonl").read_text().strip())
    assert index["runner_step"] == 7
    chunk = torch.load(tmp_path / index["file"], weights_only=False)
    assert chunk["actions"]["pi05_model_output"].shape == (1, 50, 32)
    assert chunk["actions"]["executed"][0, 0, 0].item() == 4.0
    assert chunk["flags"]["rl_phase_requested"].item()
    encoded = chunk["observations"]["current"]["main_images"]
    assert encoded["leading_shape"] == (1,)
    decoded = cv2.imdecode(
        np.frombuffer(encoded["data"][0], dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded.shape == (12, 16, 3)


def test_raw_rollout_archive_appends_after_restart(tmp_path):
    record = {
        "observations": {"current": _obs(1), "next": _obs(2)},
        "actions": {},
        "flags": {},
        "outcome": {},
        "rtc": {},
    }
    first = RawRolloutArchive(tmp_path, manifest={})
    first.submit(record)
    first.close()
    second = RawRolloutArchive(tmp_path, manifest={})
    second.submit(record)
    second.close()
    assert (tmp_path / "chunks/chunk_000000000.pt").is_file()
    assert (tmp_path / "chunks/chunk_000000001.pt").is_file()


def test_env_worker_labels_raw_rollout_actions_and_flags():
    class Collector:
        def __init__(self):
            self.record = None

        def submit(self, record):
            self.record = record

    worker = EnvWorker.__new__(EnvWorker)
    worker.raw_rollout_archive = Collector()
    worker.raw_rollout_runner_step = 4
    forward = {
        "pi05_model_output": torch.zeros(1, 50, 32),
        "ref_chunk": torch.ones(1, 30, 16),
        "model_action": torch.full((1, 480), 2.0),
        "record_transition": torch.tensor([[True]]),
        "actor_switch": torch.tensor([[False]]),
    }
    rollout_result = RolloutResult(
        actions=torch.full((1, 30, 16), 3.0), forward_inputs=forward
    )
    next_output = EnvOutput(
        obs=_obs(2),
        executed_actions=torch.full((1, 30, 16), 4.0),
        intervene_actions=torch.full((1, 30, 16), 5.0),
        intervene_flags=torch.tensor([[False, True]]),
    )

    worker._archive_raw_rollout_chunk(
        policy_input_obs=_obs(0),
        chunk_start_obs=_obs(1),
        next_output=next_output,
        rollout_result=rollout_result,
        rewards=torch.tensor([[1.0]]),
        rollout_epoch=2,
        chunk_step=3,
        stage_id=0,
    )

    record = worker.raw_rollout_archive.record
    assert record["runner_step"] == 4
    assert record["actions"]["pi05_reference"].shape == (1, 30, 16)
    assert record["actions"]["rlt_actor_raw"][0, 0].item() == 2.0
    assert record["actions"]["intervention"][0, 0, 0].item() == 5.0
    assert record["flags"]["rl_phase_requested"].item()
    assert not record["flags"]["rlt_actor_active"].item()
    assert "pi05_model_output" not in rollout_result.forward_inputs
