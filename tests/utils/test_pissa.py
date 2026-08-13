"""Tests for PiSSA decomposition and tensor-parallel initialization."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.workers.megatron import pissa_init
from skyrl.utils.pissa import pissa_decompose


def _principal(weight, rank):
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return (u[:, :rank] * s[:rank].unsqueeze(0)) @ vh[:rank, :]


@pytest.mark.parametrize("shape", [(64, 48), (48, 64), (32, 32)])
def test_pissa_decomposition_uses_principal_components(shape):
    torch.manual_seed(0)
    w = torch.randn(*shape)
    rank = 4
    scale = 0.5
    linear_in, linear_out, residual = pissa_decompose(w, rank, scale)

    principal = scale * (linear_out @ linear_in)
    torch.testing.assert_close(principal, _principal(w, rank), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(residual + principal, w, atol=1e-6, rtol=1e-6)


def test_rank_too_large_raises():
    w = torch.randn(8, 12)
    with pytest.raises(ValueError):
        pissa_decompose(w, 9, 1.0)  # 9 > min(8, 12)


@pytest.mark.parametrize(
    ("init_method", "export_residual_base", "merge_lora", "rank", "lora_type"),
    [
        ("pissa", False, True, 32, "lora"),
        ("pissa", True, False, 32, "lora"),
        ("kaiming", True, True, 32, "lora"),
        ("pissa_niter_4", True, True, 32, "lora"),
        ("pissa", True, True, 0, "lora"),
        ("pissa", True, True, 32, "canonical_lora"),
    ],
)
def test_invalid_pissa_producer_config_raises(init_method, export_residual_base, merge_lora, rank, lora_type):
    with pytest.raises(ValueError):
        pissa_init.validate_pissa_producer_config(
            init_method,
            export_residual_base,
            merge_lora,
            rank,
            lora_type,
        )


@pytest.mark.parametrize(
    ("init_method", "export_residual_base", "merge_lora", "rank", "lora_type", "expected"),
    [
        ("kaiming", False, False, 32, "canonical_lora", False),
        ("pissa", True, True, 32, "lora", True),
    ],
)
def test_valid_pissa_producer_config_is_identified(
    init_method,
    export_residual_base,
    merge_lora,
    rank,
    lora_type,
    expected,
):
    assert (
        pissa_init.validate_pissa_producer_config(
            init_method,
            export_residual_base,
            merge_lora,
            rank,
            lora_type,
        )
        is expected
    )


@pytest.mark.parametrize("export_raises", [False, True])
def test_residual_export_restores_adapter_weights(monkeypatch, export_raises):
    class ParallelLinearAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_out = torch.nn.Linear(3, 4, bias=False)

    class LoRALinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = ParallelLinearAdapter()

    lora_layers = ModuleType("megatron.bridge.peft.lora_layers")
    lora_layers.LoRALinear = LoRALinear
    peft_utils = ModuleType("megatron.bridge.peft.utils")
    peft_utils.ParallelLinearAdapter = ParallelLinearAdapter
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft.lora_layers", lora_layers)
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft.utils", peft_utils)

    model = torch.nn.Sequential(LoRALinear())
    weight = model[0].adapter.linear_out.weight
    original = weight.detach().clone()
    original_device = weight.device

    if export_raises:
        with pytest.raises(RuntimeError):
            with pissa_init.zeroed_adapters(model):
                torch.testing.assert_close(weight, torch.zeros_like(weight))
                assert weight.device == original_device
                raise RuntimeError("export failed")
    else:
        with pissa_init.zeroed_adapters(model):
            torch.testing.assert_close(weight, torch.zeros_like(weight))
            assert weight.device == original_device

    torch.testing.assert_close(weight, original)


@pytest.mark.parametrize("input_is_parallel", [False, True], ids=["column_parallel", "row_parallel"])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_pissa_initialization_reshards_parallel_adapters(monkeypatch, input_is_parallel, tp_rank):
    torch.manual_seed(6)
    full_weight = torch.randn(12, 8)
    rank = 4
    tp_size = 2
    base_shard_dim = 1 if input_is_parallel else 0
    linear_in_shard_dim = 1 if input_is_parallel else 0
    base_weight = full_weight.chunk(tp_size, dim=base_shard_dim)[tp_rank].clone()
    linear_in_shape = [rank, full_weight.shape[1]]
    linear_in_shape[linear_in_shard_dim] //= tp_size

    base_linear = SimpleNamespace(weight=torch.nn.Parameter(base_weight))
    adapter = SimpleNamespace(
        input_is_parallel=input_is_parallel,
        dim=rank,
        alpha=rank,
        linear_in=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(linear_in_shape))),
        linear_out=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(full_weight.shape[0] // tp_size, rank))),
    )
    monkeypatch.setattr(pissa_init, "_all_gather", lambda *args: full_weight)

    pissa_init._init_one_adapter(base_linear, adapter, tp_size, tp_rank, None)

    linear_in, linear_out, residual = pissa_decompose(full_weight, rank, scale=1.0)
    torch.testing.assert_close(base_linear.weight, residual.chunk(tp_size, dim=base_shard_dim)[tp_rank])
    torch.testing.assert_close(adapter.linear_in.weight, linear_in.chunk(tp_size, dim=linear_in_shard_dim)[tp_rank])
    torch.testing.assert_close(adapter.linear_out.weight, linear_out.chunk(tp_size, dim=0)[tp_rank])
