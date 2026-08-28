"""Load a precomputed PEFT adapter into a dense TP=1 Megatron LoRA model."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import torch

PRETRAINED_LORA_PREFIX = "pretrained:"
_LAYER_PATTERN = re.compile(
    r"(?:^|\.)decoder\.layers\.(?P<layer>\d+)\."
    r"(?P<block>self_attention|mlp)\."
    r"(?P<projection>linear_qkv|linear_proj|linear_fc1|linear_fc2)$"
)
_PEFT_PREFIX = "base_model.model."


@dataclasses.dataclass(frozen=True)
class PretrainedLoraConfig:
    """Path to PEFT factors that accompany the model's residual base."""

    adapter_path: str

    @classmethod
    def from_init_method(cls, init_method: str) -> PretrainedLoraConfig | None:
        """Parse ``pretrained:<adapter_model.safetensors>`` or return ``None``."""
        if not init_method.startswith(PRETRAINED_LORA_PREFIX):
            return None
        adapter_path = init_method.removeprefix(PRETRAINED_LORA_PREFIX)
        if not adapter_path:
            raise ValueError(f"{PRETRAINED_LORA_PREFIX} requires an adapter path")
        return cls(adapter_path=adapter_path)


def _normalize_adapter_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for name, tensor in state.items():
        normalized_name = name.removeprefix(_PEFT_PREFIX)
        if normalized_name in normalized:
            raise ValueError(f"duplicate normalized adapter key: {normalized_name}")
        normalized[normalized_name] = tensor
    return normalized


def _get_factor(state: dict[str, torch.Tensor], base_name: str, factor: str) -> torch.Tensor:
    name = f"{base_name}.lora_{factor}.weight"
    try:
        return state[name]
    except KeyError as exc:
        raise ValueError(f"pretrained LoRA adapter is missing {name}") from exc


def _get_projection_factors(
    state: dict[str, torch.Tensor],
    base_names: tuple[str, ...],
) -> tuple[torch.Tensor, list[torch.Tensor], set[str]]:
    linear_ins = [_get_factor(state, base_name, "A") for base_name in base_names]
    for other in linear_ins[1:]:
        if not torch.equal(linear_ins[0], other):
            raise ValueError(f"fused pretrained LoRA projections do not share A: {base_names}")
    linear_outs = [_get_factor(state, base_name, "B") for base_name in base_names]
    used = {f"{base_name}.lora_{factor}.weight" for base_name in base_names for factor in ("A", "B")}
    return linear_ins[0], linear_outs, used


def _get_hf_base_names(layer: int, block: str, projection: str) -> tuple[str, ...]:
    if block == "self_attention" and projection == "linear_qkv":
        prefix = f"model.layers.{layer}.self_attn"
        return tuple(f"{prefix}.{name}" for name in ("q_proj", "k_proj", "v_proj"))
    if block == "self_attention" and projection == "linear_proj":
        return (f"model.layers.{layer}.self_attn.o_proj",)
    if block == "mlp" and projection == "linear_fc1":
        prefix = f"model.layers.{layer}.mlp"
        return tuple(f"{prefix}.{name}" for name in ("gate_proj", "up_proj"))
    if block == "mlp" and projection == "linear_fc2":
        return (f"model.layers.{layer}.mlp.down_proj",)
    raise ValueError(f"unsupported Megatron LoRA projection: {block}.{projection}")


def _copy_factor(target: torch.Tensor, source: torch.Tensor, name: str) -> None:
    if target.shape != source.shape:
        raise ValueError(
            f"pretrained LoRA factor {name} has shape {tuple(source.shape)}, expected {tuple(target.shape)}"
        )
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def _merge_qkv_output_factors(config, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Interleave PEFT Q/K/V output factors in Megatron's fused-QKV order."""
    rank = q.shape[1]
    if any(factor.ndim != 2 or factor.shape[1] != rank for factor in (q, k, v)):
        raise ValueError("pretrained Q/K/V LoRA B factors must be rank-2 with a shared rank")

    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = config.kv_channels or (config.hidden_size // head_num)
    q_head_size = head_size * 2 if getattr(config, "attention_output_gate", False) else head_size
    q = q.view(head_num, q_head_size, rank)
    k = k.view(num_query_groups, head_size, rank)
    v = v.view(num_query_groups, head_size, rank)
    if getattr(config, "attention_output_gate", False):
        q, output_gate = torch.chunk(q, 2, dim=1)

    groups = []
    for index in range(num_query_groups):
        q_group = q[index * heads_per_group : (index + 1) * heads_per_group]
        if getattr(config, "attention_output_gate", False):
            output_gate_group = output_gate[index * heads_per_group : (index + 1) * heads_per_group]
            groups.extend((q_group, output_gate_group, k[index : index + 1], v[index : index + 1]))
        else:
            groups.extend((q_group, k[index : index + 1], v[index : index + 1]))
    return torch.cat(groups, dim=0).reshape(-1, rank)


@torch.no_grad()
def apply_pretrained_lora_init(model_chunks, config: PretrainedLoraConfig) -> None:
    """Overwrite fresh dense Megatron LoRA factors from a PEFT safetensors file."""
    import megatron.core.parallel_state as mpu
    from loguru import logger
    from megatron.bridge.peft.lora_layers import LoRALinear
    from megatron.bridge.peft.utils import ParallelLinearAdapter
    from safetensors.torch import load_file

    tp_size = mpu.get_tensor_model_parallel_world_size()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    if tp_size != 1 or pp_size != 1:
        raise ValueError(
            f"pretrained LoRA initialization currently requires TP=1 and PP=1, got TP={tp_size}, PP={pp_size}"
        )

    adapter_path = Path(config.adapter_path)
    if not adapter_path.is_file():
        raise FileNotFoundError(f"pretrained LoRA adapter does not exist: {adapter_path}")
    state = _normalize_adapter_state(load_file(adapter_path, device="cpu"))
    used_keys: set[str] = set()
    initialized = 0

    chunks = model_chunks if isinstance(model_chunks, (list, tuple)) else [model_chunks]
    for chunk in chunks:
        for module_name, module in chunk.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            adapter = module.adapter
            if not isinstance(adapter, ParallelLinearAdapter):
                raise ValueError(f"pretrained LoRA does not support adapter type {type(adapter).__name__}")
            match = _LAYER_PATTERN.search(module_name)
            if match is None:
                raise ValueError(f"pretrained LoRA cannot map Megatron module {module_name}")

            layer = int(match.group("layer"))
            block = match.group("block")
            projection = match.group("projection")
            base_names = _get_hf_base_names(layer, block, projection)
            linear_in, linear_outs, keys = _get_projection_factors(state, base_names)
            used_keys.update(keys)

            if projection == "linear_qkv":
                linear_out = _merge_qkv_output_factors(module.to_wrap.config, *linear_outs)
            elif projection == "linear_fc1":
                linear_out = torch.cat(linear_outs, dim=0)
            else:
                linear_out = linear_outs[0]

            _copy_factor(adapter.linear_in.weight, linear_in, f"{module_name}.linear_in")
            _copy_factor(adapter.linear_out.weight, linear_out, f"{module_name}.linear_out")
            initialized += 1

    unused_keys = set(state) - used_keys
    if unused_keys:
        raise ValueError(f"pretrained LoRA adapter has unmapped factors: {sorted(unused_keys)}")
    if initialized == 0:
        raise ValueError("pretrained LoRA did not find any Megatron adapter modules")
    logger.info(f"Loaded {initialized} pretrained LoRA adapter(s) from {adapter_path}")


def build_pretrained_lora_pre_wrap_hook(config: PretrainedLoraConfig):
    """Build the post-LoRA-transform hook that loads precomputed factors."""

    def hook(model):
        apply_pretrained_lora_init(model, config)
        return model

    return hook
