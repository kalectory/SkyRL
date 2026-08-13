"""Tests for PiSSA decomposition and tensor-parallel initialization."""

import json
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.workers.megatron import pissa_init
from skyrl.train.entrypoints.pissa_init import _write_manifest

pytestmark = pytest.mark.megatron


def _principal(weight, rank):
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return (u[:, :rank] * s[:rank].unsqueeze(0)) @ vh[:rank, :]


@pytest.mark.parametrize("shape", [(64, 48), (48, 64), (32, 32)])
def test_pissa_decomposition_uses_principal_components(shape):
    torch.manual_seed(0)
    w = torch.randn(*shape)
    rank = 4
    scale = 0.5
    linear_in, linear_out, residual = pissa_init.pissa_decompose(w, rank, scale)

    principal = scale * (linear_out @ linear_in)
    torch.testing.assert_close(principal, _principal(w, rank), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(residual + principal, w, atol=1e-6, rtol=1e-6)


def test_rank_too_large_raises():
    w = torch.randn(8, 12)
    with pytest.raises(ValueError):
        pissa_init.pissa_decompose(w, 9, 1.0)  # 9 > min(8, 12)


@pytest.mark.parametrize(
    ("rank", "lora_type"),
    [
        (0, "lora"),
        (32, "canonical_lora"),
    ],
)
def test_invalid_pissa_config_raises(rank, lora_type):
    with pytest.raises(ValueError):
        pissa_init.validate_pissa_config(rank, lora_type)


def test_valid_pissa_config_is_accepted():
    pissa_init.validate_pissa_config(32, "lora")


def test_pissa_manifest_records_training_inputs(tmp_path):
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            policy=SimpleNamespace(
                model=SimpleNamespace(
                    lora=SimpleNamespace(target_modules="all-linear", exclude_modules=None),
                )
            )
        )
    )

    _write_manifest(tmp_path, "Qwen/test-model", 32, cfg)

    manifest = json.loads((tmp_path / "pissa_init.json").read_text())
    assert manifest == {
        "schema_version": 1,
        "source_model": "Qwen/test-model",
        "policy_model_path": "residual_base",
        "resume_path": "global_step_0",
        "reference_model": "Qwen/test-model",
        "rank": 32,
        "alpha": 32,
        "target_modules": "all-linear",
        "exclude_modules": None,
    }


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

    monkeypatch.setattr(pissa_init, "LoRALinear", LoRALinear)
    monkeypatch.setattr(pissa_init, "ParallelLinearAdapter", ParallelLinearAdapter)

    model = torch.nn.Sequential(LoRALinear())
    weight = model[0].adapter.linear_out.weight
    original = weight.detach().clone()
    original_device = weight.device

    if export_raises:
        with pytest.raises(RuntimeError), pissa_init.zeroed_adapters(model):
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

    linear_in, linear_out, residual = pissa_init.pissa_decompose(full_weight, rank, scale=1.0)
    torch.testing.assert_close(base_linear.weight, residual.chunk(tp_size, dim=base_shard_dim)[tp_rank])
    torch.testing.assert_close(adapter.linear_in.weight, linear_in.chunk(tp_size, dim=linear_in_shard_dim)[tp_rank])
    torch.testing.assert_close(adapter.linear_out.weight, linear_out.chunk(tp_size, dim=0)[tp_rank])
