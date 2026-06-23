"""Runtime PiSSA initialization for Megatron LoRA adapters.

Applies PiSSA (the pure math lives in ``skyrl.utils.pissa``) to megatron-bridge
LoRA adapters *after* the LoRA transform, inside the worker's lora pre-wrap hook
(by which point the pretrained HF weights are loaded). For each adapted linear,
the base weight is gathered across tensor-parallel ranks, decomposed, and the
residual + principal A/B written back in the sharded layout. The pristine
snapshot taken later by ``AdapterStore.register_pristine`` captures this state.

megatron-bridge's ``ParallelLinearAdapter.forward`` computes
``(alpha/dim)·linear_out(linear_in(x))``, so the effective delta weight is
``scale·B·A`` with ``scale = alpha/dim``; ``pissa_decompose`` folds ``1/√scale``
into both factors so ``scale·B·A`` reproduces the principal component for any alpha.

For the multi-tenant (``merge_lora=false``) serving path, prefer materializing a
residual base model offline via ``skyrl.utils.pissa.materialize_pissa`` instead.
"""

import torch

from skyrl.utils.pissa import PissaConfig, pissa_decompose


def bridge_a_init_method(init_method: str) -> str:
    """Map ``init_method`` to a megatron-bridge-valid ``lora_A_init_method``.

    PiSSA overwrites A/B post-transform and the bridge's ``_get_init_fn`` rejects
    "pissa", so PiSSA methods resolve to a valid placeholder; all others pass through.
    """
    return "kaiming" if PissaConfig.from_init_method(init_method) is not None else init_method


def register_pissa_pre_wrap_hook(provider, init_method: str) -> None:
    """Register a pre-wrap hook that applies PiSSA init when ``init_method`` is PiSSA.

    Registered after the LoRA transform hook, so it runs once the adapters exist
    and the pretrained weights are loaded; a no-op for non-PiSSA init methods.
    """
    config = PissaConfig.from_init_method(init_method)
    if config is None:
        return

    def pissa_pre_wrap_hook(model):
        apply_pissa_init(model, config)
        return model

    provider.register_pre_wrap_hook(pissa_pre_wrap_hook)


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
def _init_one_adapter(base_linear, adapter, tp_size: int, tp_rank: int, tp_group, niter: int) -> None:
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
    linear_in_full, linear_out_full, residual_full = pissa_decompose(full_w, rank, scale, niter)

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
def apply_pissa_init(model_chunks, config: PissaConfig) -> None:
    """Overwrite megatron-bridge LoRA adapters in-place with PiSSA initialization.

    Walks every ``LoRALinear`` in the (already weight-loaded, LoRA-transformed)
    model and replaces the adapter's A/B and the frozen base weight with the
    PiSSA factors. Grouped MoE expert adapters are skipped (their sharded layout
    differs); a count of skipped modules is logged.

    Args:
      model_chunks: the LoRA-transformed model (single module or VPP chunk list).
      config: parsed PiSSA options (fast-SVD niter, etc.).
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
            _init_one_adapter(module.to_wrap, adapter, tp_size, tp_rank, tp_group, config.niter)
            initialized += 1

    logger.info(
        f"PiSSA(niter={config.niter}): initialized {initialized} LoRA adapter(s) "
        f"(skipped {skipped} non-parallel/expert adapter(s))"
    )
