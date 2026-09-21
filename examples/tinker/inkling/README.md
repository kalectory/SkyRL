# Inkling-Small with Tinker and FSDP

This text-only attention-LoRA recipe uses Inkling-Small (276B total / 12B active
parameters) on four eight-GPU H200 nodes: 24 FSDP training GPUs and a separate
TP8 inference engine. It uses a BF16 frozen base, FP32 router/convolution
computation, rank-32 adapters, and a maximum context of 32,768 tokens. Packed
sequences, sequence parallelism, and nonzero attention dropout are unsupported.

Use an existing 32-GPU Ray cluster with the same SkyRL environment on every node.
Set `INKLING_MULTIMEM_AR=0` in the head and worker environments before starting
Ray; the model's multimem all-reduce path does not support LoRA. Mount writable
storage at the same path on every node and the API host. Replace the example
`/shared/skyrl/inkling` paths below and in `backend_config.json` with that mount.

The JSON pins training and inference to model revision
`8cc5877b44d343f88b92086aa1fb72897950f06a`. Pre-cache this revision on each node.
The API tokenizer currently resolves the repository's default revision, so its
local cache must also resolve that same snapshot before enabling offline mode.

From the repository root, start the API on the Ray head:

```bash
RAY_ADDRESS=auto INKLING_MULTIMEM_AR=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --isolated --extra tinker --extra fsdp -m skyrl.tinker.api \
  --backend fsdp \
  --base-model thinkingmachines/Inkling-Small \
  --backend-config "$(cat examples/tinker/inkling/backend_config.json)" \
  --checkpoints-base /shared/skyrl/inkling/checkpoints
```

Create the Tinker training client with LoRA rank 32. SkyRL installs the model's
compact attention and precision patches automatically, including the vLLM FP32
router projection; no Docker-side source patch is required.

The preceding implementation completed native training, sampling, checkpoint
restore, and exact-context capacity checks on this topology. Those results
predate this source refactor and the rebase onto upstream `309de3f4`; GPU
validation of the refactored code and updated upstream weight-sync path remains
pending.
