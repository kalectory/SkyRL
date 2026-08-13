"""Tensor-parallel PiSSA initialization for Megatron LoRA adapters."""

import torch

from skyrl.utils.pissa import pissa_decompose


def pissa_pre_wrap_hook():
    """Build a pre-wrap hook that applies PiSSA after the LoRA transform."""

    def hook(model):
        apply_pissa_init(model)
        return model

    return hook


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

    input_is_parallel = adapter.input_is_parallel
    rank = adapter.dim
    scale = adapter.alpha / adapter.dim

    base_shard_dim = 1 if input_is_parallel else 0
    if tp_size == 1:
        full_w = base_weight.data.float()
    else:
        full_w = _all_gather(base_weight.data.float(), base_shard_dim, tp_size, tp_group)

    linear_in_full, linear_out_full, residual_full = pissa_decompose(full_w, rank, scale)

    lin_in_shard_dim = 1 if input_is_parallel else 0
    base_dtype = base_weight.dtype
    lora_dtype = adapter.linear_in.weight.dtype

    base_weight.data.copy_(_shard(residual_full, base_shard_dim, tp_rank, tp_size).to(base_dtype))
    adapter.linear_out.weight.data.copy_(_shard(linear_out_full, 0, tp_rank, tp_size).to(lora_dtype))
    adapter.linear_in.weight.data.copy_(_shard(linear_in_full, lin_in_shard_dim, tp_rank, tp_size).to(lora_dtype))


@torch.no_grad()
def apply_pissa_init(model_chunks) -> None:
    """Overwrite supported Megatron LoRA adapters with PiSSA factors."""
    import megatron.core.parallel_state as mpu
    from loguru import logger
    from megatron.bridge.peft.lora_layers import LoRALinear
    from megatron.bridge.peft.utils import ParallelLinearAdapter

    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_group = mpu.get_tensor_model_parallel_group()

    chunks = model_chunks if isinstance(model_chunks, (list, tuple)) else [model_chunks]
    adapters = []
    unsupported = []
    for chunk in chunks:
        for module in chunk.modules():
            if not isinstance(module, LoRALinear):
                continue
            if isinstance(module.adapter, ParallelLinearAdapter):
                adapters.append((module.to_wrap, module.adapter))
            else:
                unsupported.append(type(module.adapter).__name__)

    if unsupported:
        adapter_types = ", ".join(sorted(set(unsupported)))
        raise ValueError(f"PiSSA does not support adapter types: {adapter_types}")
    if not adapters:
        raise ValueError("PiSSA found no supported LoRA adapters")

    for base_linear, adapter in adapters:
        _init_one_adapter(base_linear, adapter, tp_size, tp_rank, tp_group)

    logger.info(f"PiSSA: initialized {len(adapters)} LoRA adapter(s)")
