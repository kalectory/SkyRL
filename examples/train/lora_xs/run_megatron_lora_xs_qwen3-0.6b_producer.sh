#!/usr/bin/env bash

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
LORA_RANK="${LORA_RANK:-32}"
LORA_XS_INIT_METHOD="${LORA_XS_INIT_METHOD:-lora_xs}"
LORA_XS_INIT_DIR="${LORA_XS_INIT_DIR:-$HOME/lora_xs/qwen3_0.6b_${LORA_XS_INIT_METHOD}_r${LORA_RANK}}"
DEFAULT_CONFIG_OVERRIDES='{"trainer.placement.policy_num_gpus_per_node":8,"trainer.policy.megatron_config.tensor_model_parallel_size":1,"trainer.policy.megatron_config.pipeline_model_parallel_size":1,"trainer.policy.megatron_config.context_parallel_size":1}'
CONFIG_OVERRIDES="${CONFIG_OVERRIDES:-$DEFAULT_CONFIG_OVERRIDES}"

uv run --isolated --extra megatron -m skyrl.train.entrypoints.lora_xs_init \
  --base-model "$BASE_MODEL" \
  --rank "$LORA_RANK" \
  --init-method "$LORA_XS_INIT_METHOD" \
  --output-dir "$LORA_XS_INIT_DIR" \
  --config-overrides "$CONFIG_OVERRIDES"
