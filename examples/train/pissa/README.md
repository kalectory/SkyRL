# PiSSA with Megatron

PiSSA initialization is an offline step. It writes a Hugging Face residual model and a standard SkyRL step-zero checkpoint containing the initialized adapter and optimizer state.

Prepare GSM8K, then produce the initialization artifacts:

```bash
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"
PISSA_INIT_DIR="$HOME/pissa/qwen2_5_0.5b_r32" \
  bash examples/train/pissa/run_qwen2_5_0.5b_megatron_pissa_producer.sh
```

The producer creates:

```text
pissa_init.json
residual_base/
global_step_0/
```

Start training from those artifacts:

```bash
PISSA_INIT_DIR="$HOME/pissa/qwen2_5_0.5b_r32" \
  bash examples/train/pissa/run_qwen2_5_0.5b_megatron_pissa.sh
```

The launcher reads the source model and LoRA rank from `pissa_init.json`, prints the resolved values, and then appends any additional arguments. User arguments therefore override its defaults. The policy loads the residual model and resumes the initialized adapter from `global_step_0`; the frozen KL reference loads the original source model.

PiSSA uses `alpha=rank`, `target_modules=all-linear`, and no exclusions. Training may use either Megatron weight-sync mode; the example defaults to `merge_lora=false`.
