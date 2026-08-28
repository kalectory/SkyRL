"""LoRA-XS adapters for Megatron training and PEFT export."""

import contextlib

import torch
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.lora_layers import LoRALinear, TEFusedLoRALinear
from megatron.bridge.peft.utils import ParallelLinearAdapter
from torch import nn

LORA_XS_INIT_STD = 1e-5


class LoRAXSCore(nn.Linear):
    """Trainable square projection between frozen LoRA factors."""


def configure_lora_xs_adapter(adapter: ParallelLinearAdapter) -> None:
    """Freeze LoRA factors and insert a noise-initialized square core."""
    if adapter.is_expert:
        raise ValueError("LoRA-XS does not support expert adapters")
    if not isinstance(adapter.activation, nn.Identity):
        raise TypeError("LoRA-XS requires identity LoRA activation")

    adapter.linear_in.weight.requires_grad_(False)
    adapter.linear_out.weight.requires_grad_(False)
    core = LoRAXSCore(
        adapter.dim,
        adapter.dim,
        bias=False,
        device=adapter.linear_in.weight.device,
        dtype=adapter.linear_in.weight.dtype,
    )
    nn.init.normal_(core.weight, std=LORA_XS_INIT_STD)
    adapter.activation = core


class LoRAXS(LoRA):
    """LoRA with frozen SVD factors and a trainable square core."""

    def export_context(self, model_chunks):
        """Materialize standard LoRA weights while exporting."""
        return materialized_lora_xs_adapters(model_chunks)

    def transform(self, module: nn.Module, name=None, prefix=None) -> nn.Module:
        if isinstance(module, LoRALinear):
            return module

        transformed = super().transform(module, name, prefix)
        if transformed is module:
            return module
        if isinstance(transformed, TEFusedLoRALinear):
            transformed = LoRALinear(transformed.to_wrap, transformed.adapter)
        if not isinstance(transformed, LoRALinear) or not isinstance(transformed.adapter, ParallelLinearAdapter):
            raise TypeError("LoRA-XS supports only Megatron parallel linear adapters")

        configure_lora_xs_adapter(transformed.adapter)
        return transformed


def _get_lora_xs_adapters(model_chunks) -> list[ParallelLinearAdapter]:
    chunks = model_chunks if isinstance(model_chunks, (list, tuple)) else [model_chunks]
    return [
        module.adapter
        for chunk in chunks
        for module in chunk.modules()
        if isinstance(module, LoRALinear)
        and isinstance(module.adapter, ParallelLinearAdapter)
        and isinstance(module.adapter.activation, LoRAXSCore)
    ]


@contextlib.contextmanager
def materialized_lora_xs_adapters(model_chunks):
    """Temporarily expose LoRA-XS adapters as standard two-factor LoRA."""
    saved = []
    try:
        with torch.no_grad():
            for adapter in _get_lora_xs_adapters(model_chunks):
                core = adapter.activation
                linear_out = adapter.linear_out.weight
                saved.append((adapter, core, linear_out.detach().clone()))
                linear_out.copy_(linear_out.float() @ core.weight.float())
                adapter.activation = nn.Identity()
        yield
    finally:
        with torch.no_grad():
            for adapter, core, linear_out in saved:
                adapter.linear_out.weight.copy_(linear_out)
                adapter.activation = core
