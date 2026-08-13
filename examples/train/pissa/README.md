# PiSSA with Megatron

PiSSA initialization is an offline step. It writes a Hugging Face residual model and a standard SkyRL step-zero checkpoint containing the initialized adapter and optimizer state.

Current support is experimental and limited to dense Megatron models using standard LoRA adapters. Grouped expert adapters are rejected.

Prepare GSM8K, then produce the initialization artifacts:

```bash
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"
PISSA_INIT_DIR="$HOME/pissa/qwen3_0.6b_r32" \
  bash examples/train/pissa/run_megatron_pissa_qwen3-0.6b_producer.sh
```

The producer creates:

```text
residual_base/
global_step_0/
```

Start training from those artifacts:

```bash
PISSA_INIT_DIR="$HOME/pissa/qwen3_0.6b_r32" \
  bash examples/train/pissa/run_megatron_pissa_qwen3-0.6b.sh
```

The policy loads the residual model and resumes the initialized adapter from `global_step_0`; the frozen KL reference loads the original source model. Additional arguments appended to the launcher override its defaults.

The base model, LoRA rank, and target modules must match between the producer and training launcher.

PiSSA uses `alpha=rank`, `target_modules=all-linear`, and no exclusions. Apart from loading the PiSSA artifacts, the training example matches `examples/train/megatron/run_megatron_lora_qwen3-0.6b.sh`.
