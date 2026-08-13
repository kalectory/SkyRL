#!/usr/bin/env bash

set -euo pipefail
set -x

if [[ -z "${PISSA_INIT_DIR:-}" ]]; then
  echo "Set PISSA_INIT_DIR to the PiSSA producer output directory." >&2
  exit 1
fi

mapfile -t PISSA_VALUES < <(
  uv run python -c \
    'import json, sys; m = json.load(open(sys.argv[1])); print(m["source_model"]); print(m["rank"])' \
    "$PISSA_INIT_DIR/pissa_init.json"
)
SOURCE_MODEL="${PISSA_VALUES[0]}"
LORA_RANK="${PISSA_VALUES[1]}"
RESIDUAL_MODEL="$PISSA_INIT_DIR/residual_base"
RESUME_PATH="$PISSA_INIT_DIR/global_step_0"

echo "PiSSA source model: $SOURCE_MODEL"
echo "PiSSA residual model: $RESIDUAL_MODEL"
echo "PiSSA initialization checkpoint: $RESUME_PATH"
echo "PiSSA LoRA rank: $LORA_RANK"

DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
NUM_GPUS="${NUM_GPUS:-2}"
LOGGER="${LOGGER:-wandb}"

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="$RESIDUAL_MODEL" \
  trainer.ref.model.path="$SOURCE_MODEL" \
  trainer.resume_mode=from_path \
  trainer.resume_path="$RESUME_PATH" \
  trainer.strategy=megatron \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  trainer.policy.megatron_config.tensor_model_parallel_size=2 \
  trainer.ref.megatron_config.tensor_model_parallel_size=2 \
  trainer.policy.megatron_config.lora_config.merge_lora=false \
  trainer.policy.model.lora.rank="$LORA_RANK" \
  trainer.policy.model.lora.alpha="$LORA_RANK" \
  trainer.policy.model.lora.target_modules=all-linear \
  generator.inference_engine.num_engines="$NUM_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  trainer.algorithm.use_kl_loss=true \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.policy.optimizer_config.lr=1.0e-5 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  generator.n_samples_per_prompt=5 \
  environment.env_class=gsm8k \
  trainer.logger="$LOGGER" \
  trainer.project_name=gsm8k_megatron_pissa \
  trainer.run_name=qwen2_5_0.5b_megatron_pissa \
  "$@"
