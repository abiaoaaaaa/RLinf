# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Small process-safe JSONL logger for real-world runtime diagnostics."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def append_runtime_event(path: str | os.PathLike[str] | None, event: dict[str, Any]) -> None:
    """Append one timestamped JSON event; failures never stop robot control."""
    if not path:
        return
    try:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"wall_time": time.time(), **event}
        line = json.dumps(payload, default=_json_default, separators=(",", ":")) + "\n"
        with target.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line)
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        # Diagnostics must not turn a transient disk/path problem into a
        # robot-control failure.
        return
