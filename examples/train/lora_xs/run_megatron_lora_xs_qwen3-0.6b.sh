#!/usr/bin/env bash

set -euo pipefail
set -x

# Colocated GRPO for Qwen3-0.6B on GSM8K with Megatron and LoRA-XS.

LORA_RANK="${LORA_RANK:-32}"
LORA_XS_INIT_DIR="${LORA_XS_INIT_DIR:-$HOME/lora_xs/qwen3_0.6b_lora_xs_r${LORA_RANK}}"
SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-0.6B}"
POLICY_MODEL="${POLICY_MODEL:-$SOURCE_MODEL}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
NUM_GPUS="${NUM_GPUS:-8}"
LOGGER="${LOGGER:-wandb}"
LORA_ALPHA="${LORA_ALPHA:-$LORA_RANK}"

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="$POLICY_MODEL" \
  trainer.ref.model.path="$SOURCE_MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  generator.inference_engine.num_engines="$NUM_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.policy.megatron_config.tensor_model_parallel_size=1 \
  trainer.policy.megatron_config.pipeline_model_parallel_size=1 \
  trainer.policy.megatron_config.context_parallel_size=1 \
  trainer.policy.megatron_config.lora_config.lora_type=lora_xs \
  trainer.ref.megatron_config.tensor_model_parallel_size=1 \
  trainer.ref.megatron_config.pipeline_model_parallel_size=1 \
  trainer.ref.megatron_config.context_parallel_size=1 \
  trainer.policy.model.lora.rank="$LORA_RANK" \
  trainer.policy.model.lora.alpha="$LORA_ALPHA" \
  trainer.policy.model.lora.target_modules=all-linear \
  trainer.gradient_checkpointing=true \
  trainer.remove_microbatch_padding=true \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-5 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  environment.env_class=gsm8k \
  trainer.logger="$LOGGER" \
  trainer.project_name=gsm8k_megatron \
  trainer.run_name="gsm8k_megatron_qwen3_0.6b_lora_xs_r${LORA_RANK}_a${LORA_ALPHA}" \
  trainer.resume_mode=from_path \
  trainer.resume_path="$LORA_XS_INIT_DIR/global_step_0" \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_0.6b_lora_xs_ckpt" \
  "$@"
