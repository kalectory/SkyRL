"""PiSSA initialization for Megatron LoRA adapters.

PiSSA (Meng et al., 2024, https://arxiv.org/abs/2404.02948) replaces LoRA's
"noise & zero" adapter init with the principal singular components of the base
weight. For each adapted linear with base weight ``W`` (out, in):

    W = U S Vᵀ                       (economy SVD, singular values descending)
    linear_out (B) = Uᵣ Sᵣ^½          (out, r)
    linear_in  (A) = Sᵣ^½ Vᵣᵀ         (r, in)
    W_res          = W − Uᵣ Sᵣ Vᵣᵀ    (frozen base, replaces W)

so that at init ``W_res + scale·B·A == W`` exactly (the adapter starts on the
principal subspace; the frozen residual carries the rest).

megatron-bridge's ``ParallelLinearAdapter.forward`` computes
``(alpha/dim)·linear_out(linear_in(x))``, i.e. the effective delta weight is
``scale·B·A`` with ``scale = alpha/dim``. We fold ``1/√scale`` into both
factors so ``scale·B·A`` reproduces the principal component for any alpha.

This module overwrites the adapter/base tensors *after* the megatron-bridge LoRA
transform has run (inside the worker's lora pre-wrap hook, by which point the
pretrained HF weights are already loaded). The pristine snapshot taken later
by ``AdapterStore.register_pristine`` therefore captures the PiSSA state.
"""

import math

import torch


def pissa_decompose(weight: torch.Tensor, rank: int, scale: float):
    """Decompose a base weight into PiSSA adapter factors + frozen residual.

    Pure tensor math (no distributed / megatron deps) so it is unit-testable on
    CPU. Operates on the full, unsharded weight in ``(out_features, in_features)``
    layout (Megatron/torch ``nn.Linear`` convention, ``y = x Wᵀ``).

    Args:
      weight: full base weight, shape (out, in). Upcast to float32 internally.
      rank: LoRA rank r (number of principal components to keep).
      scale: the adapter's ``alpha/dim`` forward scaling factor.

    Returns:
      (linear_in, linear_out, residual) all float32:
        linear_in:  (r, in)   — adapter A
        linear_out: (out, r)  — adapter B
        residual:   (out, in) — frozen base, == weight − scale·linear_out·linear_in
    """
    out_features, in_features = weight.shape
    max_rank = min(out_features, in_features)
    if rank > max_rank:
        raise ValueError(f"PiSSA rank {rank} exceeds min(out, in) = {max_rank} for weight {tuple(weight.shape)}")

    w = weight.float()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)  # u:(out,k) s:(k) vh:(k,in)
    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]

    sqrt_s = s_r.sqrt()
    inv = 1.0 / math.sqrt(scale)
    linear_out = (u_r * sqrt_s.unsqueeze(0)) * inv  # (out, r)
    linear_in = (sqrt_s.unsqueeze(1) * vh_r) * inv  # (r, in)

    principal = (u_r * s_r.unsqueeze(0)) @ vh_r  # (out, in) == scale·linear_out·linear_in
    residual = w - principal
    return linear_in, linear_out, residual


def _all_gather(local: torch.Tensor, dim: int, tp_size: int, tp_group) -> torch.Tensor:
    """Gather an evenly-sharded tensor across the TP group and concat along ``dim``."""
    local = local.contiguous()
    parts = [torch.empty_like(local) for _ in range(tp_size)]
    torch.distributed.all_gather(parts, local, group=tp_group)
    return torch.cat(parts, dim=dim)


def _shard(full: torch.Tensor, dim: int, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Return this TP rank's contiguous shard of ``full`` along ``dim``."""
    if tp_size == 1:
        return full
    size = full.shape[dim]
    if size % tp_size != 0:
        raise ValueError(f"PiSSA: dim {dim} size {size} not divisible by tp_size {tp_size}")
    chunk = size // tp_size
    return full.narrow(dim, tp_rank * chunk, chunk).contiguous()


@torch.no_grad()
def _init_one_adapter(base_linear, adapter, tp_size: int, tp_rank: int, tp_group) -> None:
    base_weight = base_linear.weight
    if base_weight.is_meta:
        raise RuntimeError(
            "PiSSA: base weight is on the meta device at adapter-init time; pretrained weights "
            "must be loaded first. Disable meta-device init (init_model_with_meta_device) for PiSSA."
        )

    input_is_parallel = adapter.input_is_parallel  # True => RowParallel base (linear_proj/fc2)
    rank = adapter.dim
    scale = adapter.alpha / adapter.dim

    # 1. Reconstruct the full base weight W (out, in) in fp32.
    #    ColumnParallel base shards out (dim 0); RowParallel base shards in (dim 1).
    base_shard_dim = 1 if input_is_parallel else 0
    if tp_size == 1:
        full_w = base_weight.data.float()
    else:
        full_w = _all_gather(base_weight.data.float(), base_shard_dim, tp_size, tp_group)

    # 2. PiSSA decomposition on the full weight.
    linear_in_full, linear_out_full, residual_full = pissa_decompose(full_w, rank, scale)

    # 3. Write back, re-sharding to each tensor's TP layout.
    #    - base weight: same sharding as it was read.
    #    - linear_out is always ColumnParallel (out/TP, dim) -> shard out (dim 0).
    #    - linear_in: ColumnParallel (dim/TP, in) -> shard dim 0 when base is column-parallel;
    #                 RowParallel  (dim, in/TP)   -> shard dim 1 when base is row-parallel.
    lin_in_shard_dim = 1 if input_is_parallel else 0
    base_dtype = base_weight.dtype
    lora_dtype = adapter.linear_in.weight.dtype

    base_weight.data.copy_(_shard(residual_full, base_shard_dim, tp_rank, tp_size).to(base_dtype))
    adapter.linear_out.weight.data.copy_(_shard(linear_out_full, 0, tp_rank, tp_size).to(lora_dtype))
    adapter.linear_in.weight.data.copy_(_shard(linear_in_full, lin_in_shard_dim, tp_rank, tp_size).to(lora_dtype))


@torch.no_grad()
def apply_pissa_init(model_chunks) -> None:
    """Overwrite megatron-bridge LoRA adapters in-place with PiSSA initialization.

    Walks every ``LoRALinear`` in the (already weight-loaded, LoRA-transformed)
    model and replaces the adapter's A/B and the frozen base weight with the
    PiSSA factors. Grouped MoE expert adapters are skipped (their sharded layout
    differs); a count of skipped modules is logged.
    """
    # Lazy import: megatron.bridge is only importable inside the GPU worker, and
    # keeping these out of module scope lets pissa_decompose be unit-tested on CPU.
    import megatron.core.parallel_state as mpu
    from loguru import logger
    from megatron.bridge.peft.lora_layers import LoRALinear
    from megatron.bridge.peft.utils import ParallelLinearAdapter

    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_group = mpu.get_tensor_model_parallel_group()

    chunks = model_chunks if isinstance(model_chunks, (list, tuple)) else [model_chunks]
    initialized = 0
    skipped = 0
    for chunk in chunks:
        for module in chunk.modules():
            if not isinstance(module, LoRALinear):
                continue
            adapter = module.adapter
            if not isinstance(adapter, ParallelLinearAdapter):
                skipped += 1
                continue
            _init_one_adapter(module.to_wrap, adapter, tp_size, tp_rank, tp_group)
            initialized += 1

    logger.info(f"PiSSA: initialized {initialized} LoRA adapter(s) (skipped {skipped} non-parallel/expert adapter(s))")
