# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import json

import numpy as np

from rlinf.utils.runtime_log import append_runtime_event


def test_runtime_log_serializes_multielement_numpy_arrays(tmp_path):
    path = tmp_path / "runtime.jsonl"

    append_runtime_event(path, {"event": "chunk", "values": np.arange(3)})

    row = json.loads(path.read_text())
    assert row["event"] == "chunk"
    assert row["values"] == [0, 1, 2]
