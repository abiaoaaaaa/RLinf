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

"""Model-independent, asynchronous archives for real-world rollout chunks."""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


_IMAGE_KEYS = {"main_images", "wrist_images", "extra_view_images"}


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone().contiguous()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_clone(item) for item in value]
    return value


def _encode_jpeg(image: torch.Tensor | np.ndarray, quality: int) -> bytes:
    array = image.detach().cpu().numpy() if torch.is_tensor(image) else image
    array = np.asarray(array)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected HWC image, got shape {array.shape}")
    if array.shape[-1] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    elif array.shape[-1] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
    ok, encoded = cv2.imencode(
        ".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode rollout image")
    return encoded.tobytes()


def _encode_image_batch(value: Any, quality: int) -> dict[str, Any]:
    array = value.detach().cpu() if torch.is_tensor(value) else np.asarray(value)
    if array.ndim < 4:
        raise ValueError(f"Expected batched HWC images, got shape {array.shape}")
    leading_shape = tuple(int(dim) for dim in array.shape[:-3])
    flat = array.reshape(-1, *array.shape[-3:])
    return {
        "codec": "jpeg",
        "leading_shape": leading_shape,
        "image_shape": tuple(int(dim) for dim in array.shape[-3:]),
        "data": [_encode_jpeg(image, quality) for image in flat],
    }


def _encode_observation(obs: dict[str, Any], quality: int) -> dict[str, Any]:
    encoded = {}
    for key, value in obs.items():
        if value is None:
            encoded[key] = None
        elif key in _IMAGE_KEYS:
            encoded[key] = _encode_image_batch(value, quality)
        else:
            encoded[key] = value
    return encoded


class RawRolloutArchive:
    """Write every rollout chunk without blocking robot control on JPEG or I/O."""

    _STOP = object()

    def __init__(
        self,
        path: str | Path,
        *,
        manifest: dict[str, Any],
        jpeg_quality: int = 90,
        queue_size: int = 64,
    ) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        self.path = Path(path)
        self.chunk_dir = self.path / "chunks"
        self.index_path = self.path / "index.jsonl"
        self.jpeg_quality = jpeg_quality
        self.path.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(manifest)
        self._next_sequence = self._find_next_sequence()
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="raw-rollout-archive", daemon=True
        )
        self._thread.start()
        atexit.register(self.close)

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        path = self.path / "manifest.json"
        if path.exists():
            return
        payload = {
            "format": "rlinf_raw_rollout_chunk",
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image_codec": "jpeg",
            **manifest,
        }
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp_path, path)

    def _find_next_sequence(self) -> int:
        existing = [
            int(path.stem.split("_")[-1])
            for path in self.chunk_dir.glob("chunk_*.pt")
        ]
        return max(existing, default=-1) + 1

    def submit(self, record: dict[str, Any]) -> None:
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("RawRolloutArchive is closed")
        # Clone before returning to the environment loop; tensors in EnvOutput
        # and RolloutResult may be reused by the next chunk.
        self._queue.put(_cpu_clone(record))

    def flush(self) -> None:
        self._queue.join()
        self._raise_if_failed()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._queue.put(self._STOP)
        self._thread.join()
        atexit.unregister(self.close)
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Raw rollout archive writer failed") from self._error

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                self._write_one(item)
            except BaseException as exc:  # surfaced on submit/flush
                self._error = exc
            finally:
                self._queue.task_done()

    def _write_one(self, record: dict[str, Any]) -> None:
        sequence = self._next_sequence
        self._next_sequence += 1
        record["sequence_id"] = sequence
        record["archived_at"] = datetime.now(timezone.utc).isoformat()
        record["observations"] = {
            key: _encode_observation(obs, self.jpeg_quality)
            for key, obs in record["observations"].items()
        }
        filename = f"chunk_{sequence:09d}.pt"
        target = self.chunk_dir / filename
        tmp_target = self.chunk_dir / f".{filename}.tmp"
        torch.save(record, tmp_target)
        os.replace(tmp_target, target)
        index_entry = {
            "sequence_id": sequence,
            "file": f"chunks/{filename}",
            "runner_step": record.get("runner_step"),
            "rollout_epoch": record.get("rollout_epoch"),
            "chunk_step": record.get("chunk_step"),
            "stage_id": record.get("stage_id"),
        }
        with self.index_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
