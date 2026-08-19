"""Megatron worker for offline LoRA-XS initialization."""

import ray
import torch
from loguru import logger
from megatron.bridge.peft.lora_layers import LoRALinear
from megatron.bridge.peft.utils import ParallelLinearAdapter
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)

from skyrl.backends.skyrl_train.workers.megatron.lora_xs import (
    LORA_XS_INIT_STD,
    LoRAXSCore,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.backends.skyrl_train.workers.megatron.pissa_init_worker import (
    _all_gather,
    _shard,
)
from skyrl.train.config.config import get_config_as_dict


def lora_xs_factors(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the principal right vectors and singular-value-scaled left vectors."""
    max_rank = min(weight.shape)
    if rank > max_rank:
        raise ValueError(f"LoRA-XS rank {rank} exceeds maximum rank {max_rank} for weight shape {tuple(weight.shape)}")

    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return s[:rank].unsqueeze(1) * vh[:rank, :], u[:, :rank]


def _synchronized_lora_xs_factors(weight: torch.Tensor, rank: int, group) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.distributed.get_rank(group) == 0:
        linear_in, linear_out = lora_xs_factors(weight, rank)
    else:
        linear_in = torch.empty((rank, weight.shape[1]), dtype=torch.float32, device=weight.device)
        linear_out = torch.empty((weight.shape[0], rank), dtype=torch.float32, device=weight.device)

    torch.distributed.broadcast(linear_in, group=group, group_src=0)
    torch.distributed.broadcast(linear_out, group=group, group_src=0)
    return linear_in, linear_out


@torch.no_grad()
def _init_one_lora_xs_adapter(
    base_linear,
    adapter,
    tp_size: int,
    tp_rank: int,
    tp_group,
    use_residual_base: bool,
) -> None:
    base_weight = base_linear.weight
    if base_weight.is_meta:
        raise RuntimeError("LoRA-XS requires pretrained weights before adapter initialization")

    base_shard_dim = 1 if adapter.input_is_parallel else 0
    if tp_size == 1:
        full_weight = base_weight.detach().float()
        linear_in, linear_out = lora_xs_factors(full_weight, adapter.dim)
    else:
        full_weight = _all_gather(base_weight.detach().float(), base_shard_dim, tp_size, tp_group)
        linear_in, linear_out = _synchronized_lora_xs_factors(full_weight, adapter.dim, tp_group)

    linear_in_shard_dim = 1 if adapter.input_is_parallel else 0
    dtype = adapter.linear_in.weight.dtype
    adapter.linear_in.weight.copy_(_shard(linear_in, linear_in_shard_dim, tp_rank, tp_size).to(dtype))
    adapter.linear_out.weight.copy_(_shard(linear_out, 0, tp_rank, tp_size).to(dtype))
    adapter.activation.weight.normal_(std=LORA_XS_INIT_STD)
    if use_residual_base:
        adapter.activation.weight.add_(torch.eye(adapter.dim, device=base_weight.device, dtype=dtype))
        scale = adapter.alpha / adapter.dim
        residual = full_weight - scale * (linear_out @ linear_in)
        base_weight.copy_(_shard(residual, base_shard_dim, tp_rank, tp_size).to(base_weight.dtype))


def apply_lora_xs_init(model_chunks, use_residual_base: bool = False) -> None:
    """Initialize dense LoRA-XS adapters from their base weights."""
    import megatron.core.parallel_state as mpu

    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_group = mpu.get_tensor_model_parallel_group()
    chunks = model_chunks if isinstance(model_chunks, (list, tuple)) else [model_chunks]
    adapters = []
    for chunk in chunks:
        for module in chunk.modules():
            if isinstance(module, LoRALinear) and isinstance(module.adapter, ParallelLinearAdapter):
                if not isinstance(module.adapter.activation, LoRAXSCore):
                    raise TypeError("LoRA-XS found a non-LoRA-XS parallel adapter")
                adapters.append((module.to_wrap, module.adapter))

    if not adapters:
        raise ValueError("LoRA-XS found no supported adapters")
    for base_linear, adapter in adapters:
        _init_one_lora_xs_adapter(
            base_linear,
            adapter,
            tp_size,
            tp_rank,
            tp_group,
            use_residual_base,
        )
    logger.info(f"LoRA-XS: initialized {len(adapters)} adapter(s)")


class LoRAXSInitWorkerBase(MegatronPolicyWorkerBase):
    use_residual_base = False

    def make_megatron_module(
        self,
        wrap_with_ddp=True,
        ddp_config=None,
        lora_config=None,
        lora_type="lora_xs",
        bf16=True,
    ):
        if lora_type != "lora_xs":
            raise ValueError("LoRA-XS initialization requires lora_type='lora_xs'")
        self.configure_lora(lora_config, lora_type)

        def lora_pre_wrap_hook(model):
            lora_model = self.lora_cls(model, training=True)
            self.lora_cls.set_params_to_save(lora_model)
            return lora_model

        def lora_xs_pre_wrap_hook(model):
            apply_lora_xs_init(model, self.use_residual_base)
            return model

        self.provider.register_pre_wrap_hook(lora_pre_wrap_hook)
        self.provider.register_pre_wrap_hook(lora_xs_pre_wrap_hook)

        resolved_ddp_config = DistributedDataParallelConfig()
        if wrap_with_ddp:
            resolved_ddp_config.use_distributed_optimizer = True
        if ddp_config is not None:
            for key, value in get_config_as_dict(ddp_config).items():
                setattr(resolved_ddp_config, key, value)
        return self.provider.provide_distributed_model(
            ddp_config=resolved_ddp_config,
            wrap_with_ddp=wrap_with_ddp,
            bf16=bf16,
        )

    def save_residual_base(self, export_dir: str, tokenizer) -> None:
        from skyrl.backends.skyrl_train.workers.megatron.pissa_init_worker import (
            zeroed_adapters,
        )

        with zeroed_adapters(self.model.actor_module):
            self.strategy.save_hf_model(
                self.bridge,
                self.model,
                export_dir,
                tokenizer=tokenizer,
            )


class PiSSAXSInitWorkerBase(LoRAXSInitWorkerBase):
    """Megatron worker for residual-base PiSSA-XS initialization."""

    use_residual_base = True


LoRAXSInitWorker = ray.remote(num_gpus=1)(LoRAXSInitWorkerBase)
PiSSAXSInitWorker = ray.remote(num_gpus=1)(PiSSAXSInitWorkerBase)
