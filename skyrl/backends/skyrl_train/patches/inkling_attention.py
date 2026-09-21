"""Memory-efficient native Inkling relative attention for unpacked training."""

from types import MethodType

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.flex_attention import compile_friendly_flex_attention
from transformers.models.inkling.modeling_inkling import InklingRelativeLogits


def _project_compact_bias(module, relative_states, query_positions, key_positions):
    if query_positions.shape != key_positions.shape:
        raise ValueError("Inkling compact attention requires full sequences without a KV cache")
    return (relative_states @ module.proj).transpose(1, 2)


def _compact_flex_attention(module, query, key, value, attention_mask, scaling, position_bias, dropout=0.0, **kwargs):
    if dropout != 0.0:
        raise ValueError("Inkling compact attention requires zero attention dropout")
    if not isinstance(attention_mask, BlockMask):
        raise TypeError("Inkling compact attention requires the native flex BlockMask")
    extent = position_bias.shape[-1]

    def score_mod(score, batch, head, query_index, key_index):
        distance = query_index - key_index
        bias = position_bias[batch, head, query_index, distance.clamp(0, extent - 1)]
        return score + torch.where((distance >= 0) & (distance < extent), bias, 0.0)

    attention = flex_attention if query.device.type == "cpu" else compile_friendly_flex_attention
    options = (
        {} if query.device.type == "cpu" else {"training": module.training, "kernel_options": {"BACKEND": "TRITON"}}
    )
    output = attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=attention_mask,
        scale=scaling,
        enable_gqa=True,
        **options,
    )
    return output.transpose(1, 2).contiguous(), None


def install_inkling_flex_attention(model):
    if model.config.model_type != "inkling_mm_model":
        raise ValueError("inkling_flex_attention requires the native multimodal Inkling model")
    name = "skyrl_inkling_flex_attention"
    AttentionInterface.register(name, _compact_flex_attention)
    AttentionMaskInterface.register(name, AttentionMaskInterface()["flex_attention"])
    text_model = model.model.language_model
    for module in text_model.modules():
        if isinstance(module, InklingRelativeLogits):
            module.forward = MethodType(_project_compact_bias, module)
    # Keep the native attention forward, including FP32 tau scaling and sconv.
    # Only the distance-bank expansion moves inside the flex score modifier.
    text_model.set_attn_implementation(name)
    text_model.config.use_cache = False
