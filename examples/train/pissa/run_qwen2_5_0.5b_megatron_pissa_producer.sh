#!/usr/bin/env bash

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
LORA_RANK="${LORA_RANK:-32}"
PISSA_INIT_DIR="${PISSA_INIT_DIR:-$HOME/pissa/qwen2_5_0.5b_r${LORA_RANK}}"
DEFAULT_BACKEND_CONFIG='{"strategy":"megatron","trainer.placement.colocate_all":false,"trainer.placement.policy_num_gpus_per_node":2,"trainer.policy.megatron_config.tensor_model_parallel_size":2,"trainer.policy.megatron_config.pipeline_model_parallel_size":1}'
BACKEND_CONFIG="${BACKEND_CONFIG:-$DEFAULT_BACKEND_CONFIG}"

uv run --isolated --extra megatron -m skyrl.train.entrypoints.pissa_init \
  --base-model "$BASE_MODEL" \
  --rank "$LORA_RANK" \
  --output-dir "$PISSA_INIT_DIR" \
  --backend-config "$BACKEND_CONFIG"
