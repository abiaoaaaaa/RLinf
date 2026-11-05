# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryCache, TrajectoryReplayBuffer


def test_trajectory_cache_ignores_later_optional_metadata():
    cache = TrajectoryCache(max_size=2)
    cache.put(
        0,
        {
            "rewards": torch.zeros(1, 1),
            "forward_inputs": {"action": torch.zeros(1, 2)},
        },
    )
    cache.put(
        1,
        {
            "rewards": torch.ones(1, 1),
            "forward_inputs": {
                "action": torch.ones(1, 2),
                "rtc_chunk_id": torch.ones(1, 1),
            },
        },
    )

    cached = cache.get(1)
    assert torch.equal(cached["rewards"], torch.ones(1, 1))
    assert torch.equal(cached["forward_inputs"]["action"], torch.ones(1, 2))
    assert "rtc_chunk_id" not in cached["forward_inputs"]


def _buffer(path):
    return TrajectoryReplayBuffer(
        enable_cache=True,
        cache_size=8,
        sample_window_size=8,
        auto_save=True,
        auto_save_path=str(path),
    )


def test_auto_resume_loads_and_continues_persistent_buffer(tmp_path):
    first = _buffer(tmp_path)
    first.add_trajectories(
        [
            Trajectory(
                model_weights_id="stage2",
                actions=torch.zeros((1, 1, 2)),
                rewards=torch.ones((1, 1, 1)),
            )
        ]
    )
    first.close()

    resumed = _buffer(tmp_path)
    assert resumed.auto_resume()
    assert resumed.get_stats()["total_samples"] == 1

    resumed.add_trajectories(
        [
            Trajectory(
                model_weights_id="stage2",
                actions=torch.ones((1, 1, 2)),
                rewards=torch.ones((1, 1, 1)),
            )
        ]
    )
    resumed.close()

    final = _buffer(tmp_path)
    assert final.auto_resume()
    assert final.get_stats() == {
        "num_trajectories": 2,
        "total_samples": 2,
        "cache_size": 2,
    }
    final.close()


def test_auto_resume_starts_empty_without_metadata(tmp_path):
    buffer = _buffer(tmp_path)
    assert not buffer.auto_resume()
    assert buffer.get_stats()["total_samples"] == 0
    buffer.close()
