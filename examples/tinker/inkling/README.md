# Inkling-Small with Tinker and Megatron

This text-only attention-LoRA recipe uses Inkling-Small (276B total / 12B active
parameters) through Megatron-Bridge's native Inkling model. The candidate topology
uses two eight-GPU H200 nodes: an EP8 learner and a separate TP8 inference engine.
Full-model memory, parity, and 32,768-token capacity validation are pending.

Use an existing 16-GPU Ray cluster with the same SkyRL environment on every node.
Set `INKLING_MULTIMEM_AR=0` in the head and worker environments before starting
Ray; the model's multimem all-reduce path does not support LoRA. Mount writable
storage at the same path on every node and the API host. Replace the example
`/shared/skyrl/inkling` paths below and in `backend_config.json` with that mount.

Download the immutable model revision into shared storage. Passing the same
local snapshot to the API, learner, and sampler also pins their tokenizers:

```bash
INKLING_MODEL=$(uv run --isolated --extra megatron --extra tinker hf download \
  thinkingmachines/Inkling-Small \
  --revision 8cc5877b44d343f88b92086aa1fb72897950f06a \
  --cache-dir /shared/skyrl/huggingface)
```

From the repository root, start the API on the Ray head:

```bash
RAY_ADDRESS=auto INKLING_MULTIMEM_AR=0 \
uv run --isolated --extra tinker --extra megatron -m skyrl.tinker.api \
  --backend megatron \
  --base-model "$INKLING_MODEL" \
  --backend-config "$(cat examples/tinker/inkling/backend_config.json)" \
  --checkpoints-base /shared/skyrl/inkling/checkpoints
```

Create the Tinker training client with LoRA rank 32. The attention targets use
Megatron's projection names; Bridge exports the published Inkling adapter names
for vLLM. The recipe uses SkyRL's existing adapter-only weight synchronization
and checkpoint paths. Packed sequences and context parallelism are unsupported.

The lockfile pins [Megatron-Bridge's native Inkling contribution](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6170)
and its matching Megatron-Core revision. Earlier FSDP experiment results do not
qualify this native Megatron implementation. The Bridge contribution passes
tiny-model TP1/EP1, TP2/EP1, and TP1/EP2 conversion, forward/backward, LoRA export,
and checkpoint-restore tests; full-model qualification remains pending.
