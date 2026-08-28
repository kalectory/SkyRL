"""CPU tests for precomputed Megatron LoRA initialization helpers."""

import pytest
import torch

from skyrl.backends.skyrl_train.workers.megatron.pretrained_lora_init import (
    PretrainedLoraConfig,
    _get_hf_base_names,
    _get_projection_factors,
    _normalize_adapter_state,
)


def test_pretrained_lora_config_parses_path():
    assert PretrainedLoraConfig.from_init_method("kaiming") is None
    assert PretrainedLoraConfig.from_init_method("pretrained:/artifact/adapter.safetensors") == (
        PretrainedLoraConfig(adapter_path="/artifact/adapter.safetensors")
    )
    with pytest.raises(ValueError, match="requires an adapter path"):
        PretrainedLoraConfig.from_init_method("pretrained:")


def test_projection_factors_require_shared_fused_a():
    base_names = _get_hf_base_names(3, "self_attention", "linear_qkv")
    state = {}
    for index, base_name in enumerate(base_names):
        state[f"{base_name}.lora_A.weight"] = torch.ones(2, 4)
        state[f"{base_name}.lora_B.weight"] = torch.full((4, 2), index)

    linear_in, linear_outs, used = _get_projection_factors(state, base_names)

    assert torch.equal(linear_in, torch.ones(2, 4))
    assert [tensor[0, 0].item() for tensor in linear_outs] == [0, 1, 2]
    assert used == set(state)

    state[f"{base_names[-1]}.lora_A.weight"][0, 0] = 2
    with pytest.raises(ValueError, match="do not share A"):
        _get_projection_factors(state, base_names)


def test_normalize_adapter_state_strips_peft_prefix():
    tensor = torch.ones(1)
    state = _normalize_adapter_state({"base_model.model.model.layers.0.foo.lora_A.weight": tensor})

    assert state == {"model.layers.0.foo.lora_A.weight": tensor}
