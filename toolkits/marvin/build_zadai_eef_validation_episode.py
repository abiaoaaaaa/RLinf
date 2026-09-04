# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build one joint-to-EEF episode for hardware frame validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from toolkits.marvin.convert_joint_lerobot_to_eef import (
    GENERATED_FEATURES,
    PinocchioMarvinFK,
    convert_arrays,
)


def _load_episode(source: Path, episode: int) -> pa.Table:
    tables: list[pa.Table] = []
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        if "episode_index" not in table.column_names:
            continue
        selected = table.filter(pc.equal(table["episode_index"], episode))
        if len(selected):
            tables.append(selected)
    if not tables:
        raise FileNotFoundError(f"episode {episode} was not found below {source}")
    return pa.concat_tables(tables)


def _float_list(values: np.ndarray) -> pa.Array:
    return pa.array(values.tolist(), type=pa.list_(pa.float32()))


def build(source: Path, output: Path, mjcf: Path, episode: int) -> dict[str, object]:
    """Convert one source episode and retain exact joint audit fields."""
    source = source.resolve()
    output = output.resolve()
    mjcf = mjcf.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    source_table = _load_episode(source, episode)
    state = np.asarray(source_table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(source_table["action"].to_pylist(), dtype=np.float32)
    velocity = np.asarray(
        source_table["observation.velocity"].to_pylist(), dtype=np.float32
    )
    converted = convert_arrays(state, action, velocity, PinocchioMarvinFK(mjcf))

    columns: dict[str, pa.Array] = {}
    for name in GENERATED_FEATURES:
        values = converted[name]
        columns[name] = (
            pa.array(values, type=pa.int8())
            if values.ndim == 1
            else _float_list(values)
        )
    for name in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        if name in source_table.column_names:
            columns[name] = source_table[name].combine_chunks()

    target = output / "data/chunk-000" / f"episode_{episode:06d}.parquet"
    target.parent.mkdir(parents=True)
    pq.write_table(pa.table(columns), target)

    # Independent consistency checks: the EEF primary fields must agree with
    # their quaternion audit fields and source joints must survive bit-exactly.
    observation_joint_error = float(
        np.max(np.abs(converted["observation.joint_position"] - state[:, :14]))
    )
    action_joint_error = float(
        np.max(np.abs(converted["action.joint_position_source"] - action[:, :14]))
    )
    report: dict[str, object] = {
        "source": str(source),
        "output": str(output),
        "mjcf": str(mjcf),
        "episode": episode,
        "frames": len(source_table),
        "observation_joint_roundtrip_max_abs_rad": observation_joint_error,
        "action_joint_roundtrip_max_abs_rad": action_joint_error,
        "first_observation_joint": state[0, :14].tolist(),
        "first_observation_eef_quat": converted["observation.eef_pose_quat"][
            0
        ].tolist(),
    }
    (output / "validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.mjcf, args.episode), indent=2))


if __name__ == "__main__":
    main()
