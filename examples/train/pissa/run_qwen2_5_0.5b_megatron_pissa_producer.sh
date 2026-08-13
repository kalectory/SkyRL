#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_BACKEND_CONFIG='{"strategy":"megatron","trainer.placement.colocate_all":false,"trainer.placement.policy_num_gpus_per_node":2,"trainer.policy.megatron_config.tensor_model_parallel_size":2,"trainer.policy.megatron_config.pipeline_model_parallel_size":1,"trainer.policy.megatron_config.lora_config.merge_lora":true,"trainer.policy.model.lora.rank":32,"trainer.policy.model.lora.alpha":32,"trainer.policy.model.lora.init_method":"pissa","trainer.policy.model.lora.export_residual_base":true,"trainer.micro_train_batch_size_per_gpu":1,"trainer.micro_forward_batch_size_per_gpu":1,"generator.inference_engine.num_engines":2,"generator.inference_engine.tensor_parallel_size":1,"generator.inference_engine.backend":"vllm","generator.inference_engine.run_engines_locally":true,"generator.inference_engine.weight_sync_backend":"nccl"}'
BACKEND_CONFIG="${BACKEND_CONFIG:-$DEFAULT_BACKEND_CONFIG}"

uv run --extra tinker --extra megatron -m skyrl.tinker.api \
  --base-model "$MODEL_NAME" \
  --backend megatron \
  --port 8000 \
  --backend-config "$BACKEND_CONFIG" \
  "$@"
