#!/usr/bin/env bash

# Prepare and train the 335-episode Zadai Pi0.5 RLT Stage1 workflow without
# writing to the OpenPI checkout or transferring datasets between machines.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-prepare}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv_openpi/bin/python}"
ARTIFACT_ROOT="${ZADAI_335_ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/zadai_335}"
DATA_OUTPUT="${ARTIFACT_ROOT}/zadai_335_rlinf_v21"
MODEL_OUTPUT="${ARTIFACT_ROOT}/zadai_335_pi05_pytorch_bfloat16"

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Required environment variable is not set: ${name}" >&2
        exit 2
    fi
}

require_inside_repo() {
    local resolved
    resolved="$(${PYTHON_BIN} -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1")"
    case "${resolved}" in
        "${REPO_ROOT}"/*) ;;
        *)
            echo "Refusing output outside ${REPO_ROOT}: ${resolved}" >&2
            exit 2
            ;;
    esac
}

prepare_data() {
    require_env ZADAI_335_SOURCE
    if [[ -d "${DATA_OUTPUT}" ]]; then
        echo "Dataset output already exists; keeping it: ${DATA_OUTPUT}"
        return
    fi
    mkdir -p "${ARTIFACT_ROOT}"
    "${PYTHON_BIN}" "${REPO_ROOT}/toolkits/lerobot/convert_zadai_v3_to_rlinf_v21.py" \
        "${ZADAI_335_SOURCE}" "${DATA_OUTPUT}" --video-mode hardlink
}

prepare_model() {
    require_env ZADAI_335_JAX_CKPT
    require_env ZADAI_335_NORM_STATS
    if [[ -s "${MODEL_OUTPUT}/model.safetensors" ]]; then
        echo "Model output already exists; keeping it: ${MODEL_OUTPUT}"
        return
    fi
    mkdir -p "${ARTIFACT_ROOT}"
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    export PYTHONUNBUFFERED=1
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" \
        -m rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python \
        --checkpoint-dir "${ZADAI_335_JAX_CKPT}" \
        --output-path "${MODEL_OUTPUT}" \
        --precision bfloat16 \
        --pi05 \
        --action-horizon 50 \
        --norm-stats-path "${ZADAI_335_NORM_STATS}" \
        --asset-id zadai_335
}

verify() {
    "${PYTHON_BIN}" - "${DATA_OUTPUT}" "${MODEL_OUTPUT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

data_root, model_root = map(Path, sys.argv[1:])
info = json.loads((data_root / "meta/info.json").read_text())
assert info["codebase_version"] == "v2.1", info["codebase_version"]
assert info["total_episodes"] == 335, info["total_episodes"]
assert info["total_frames"] == 424905, info["total_frames"]
first_data = next((data_root / "data").rglob("*.parquet"))
assert pq.read_schema(first_data).metadata is None, first_data
config = json.loads((model_root / "config.json").read_text())
assert config["action_dim"] == 32, config["action_dim"]
assert config["action_horizon"] == 50, config["action_horizon"]
weights = model_root / "model.safetensors"
stats = model_root / "assets/zadai_335/norm_stats.json"
assert weights.stat().st_size > 6_500_000_000, weights.stat().st_size
assert stats.is_file(), stats
print(f"dataset={data_root} episodes=335 frames=424905")
print(f"model={weights} bytes={weights.stat().st_size}")
print(f"norm_stats_sha256={hashlib.sha256(stats.read_bytes()).hexdigest()}")
PY
}

launch_training() {
    verify
    export ZADAI_335_DATA="${DATA_OUTPUT}"
    export ZADAI_335_PI05_CKPT="${MODEL_OUTPUT}"
    export ZADAI_335_NORM_STATS="${MODEL_OUTPUT}/assets/zadai_335/norm_stats.json"
    export EMBODIED_PATH="${REPO_ROOT}/examples/sft"
    export REPO_PATH="${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    export MUJOCO_GL="egl"
    export PYOPENGL_PLATFORM="egl"
    local log_dir
    log_dir="${REPO_ROOT}/logs/$(date +'%Y%m%d-%H:%M:%S')-zadai_335_rlt_stage1"
    mkdir -p "${log_dir}"
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" examples/sft/train_vla_sft.py \
        --config-path "${REPO_ROOT}/examples/sft/config" \
        --config-name zadai_335_rlt_stage1_sft_openpi_pi05 \
        runner.logger.log_path="${log_dir}" "$@" 2>&1 \
        | tee -a "${log_dir}/run_sft.log"
}

smoke() {
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    launch_training runner.max_steps=1 runner.save_interval=-1 \
        actor.micro_batch_size=1 actor.global_batch_size=1 \
        cluster.component_placement.actor=0-0
}

train() {
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    launch_training
}

require_inside_repo "${ARTIFACT_ROOT}"
case "${ACTION}" in
    data) prepare_data ;;
    model) prepare_model ;;
    prepare) prepare_data; prepare_model; verify ;;
    verify) verify ;;
    smoke) smoke ;;
    train) train ;;
    *)
        echo "Usage: $0 {data|model|prepare|verify|smoke|train}" >&2
        exit 2
        ;;
esac
