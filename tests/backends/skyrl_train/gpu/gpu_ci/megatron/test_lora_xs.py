"""Tests for LoRA-XS initialization and standard-LoRA export."""

from types import SimpleNamespace

import pytest
import torch
from megatron.bridge.peft.lora_layers import LoRALinear
from torch import nn

from skyrl.backends.skyrl_train.workers.megatron import lora_xs, lora_xs_init_worker

pytestmark = pytest.mark.megatron


class _Adapter(nn.Module):
    def __init__(self, in_features=8, out_features=12, rank=4):
        super().__init__()
        self.dim = rank
        self.alpha = rank
        self.is_expert = False
        self.input_is_parallel = False
        self.linear_in = nn.Linear(in_features, rank, bias=False)
        self.linear_out = nn.Linear(rank, out_features, bias=False)
        self.activation = nn.Identity()

    def forward(self, x):
        return self.linear_out(self.activation(self.linear_in(x)))


def _principal(weight, rank):
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return (u[:, :rank] * s[:rank].unsqueeze(0)) @ vh[:rank, :]


def test_lora_xs_factors_use_principal_components():
    torch.manual_seed(0)
    weight = torch.randn(12, 8)
    linear_in, linear_out = lora_xs_init_worker.lora_xs_factors(weight, rank=4)

    torch.testing.assert_close(linear_out @ linear_in, _principal(weight, 4))
    assert linear_in.is_contiguous()
    assert linear_out.is_contiguous()


def test_lora_xs_uses_unfused_forward_for_square_core(monkeypatch):
    adapter = _Adapter()
    fused = lora_xs.TEFusedLoRALinear(nn.Identity(), adapter)
    monkeypatch.setattr(lora_xs.LoRA, "transform", lambda self, module, name, prefix: fused)
    monkeypatch.setattr(lora_xs, "ParallelLinearAdapter", _Adapter)

    transformed = lora_xs.LoRAXS().transform(nn.Identity())

    assert type(transformed) is LoRALinear
    assert isinstance(transformed.adapter.activation, lora_xs.LoRAXSCore)


def test_lora_xs_trains_only_square_core():
    torch.manual_seed(1)
    adapter = _Adapter()
    lora_xs.configure_lora_xs_adapter(adapter)
    nn.init.normal_(adapter.linear_in.weight)
    nn.init.normal_(adapter.linear_out.weight)

    output = adapter(torch.randn(3, 8)).sum()
    output.backward()

    assert adapter.linear_in.weight.grad is None
    assert adapter.linear_out.weight.grad is None
    assert adapter.activation.weight.grad.count_nonzero() > 0
    assert sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad) == adapter.dim**2
    assert output != 0
    assert adapter.activation.weight.abs().max() < 1e-3


@pytest.mark.parametrize("export_raises", [False, True])
def test_lora_xs_export_materializes_and_restores_standard_lora(monkeypatch, export_raises):
    torch.manual_seed(2)
    adapter = _Adapter()
    lora_xs.configure_lora_xs_adapter(adapter)
    nn.init.normal_(adapter.linear_out.weight)
    nn.init.normal_(adapter.activation.weight)
    wrapped = LoRALinear(nn.Identity(), adapter)
    original_out = adapter.linear_out.weight.detach().clone()
    original_core = adapter.activation
    expected_out = original_out @ original_core.weight
    monkeypatch.setattr(lora_xs, "ParallelLinearAdapter", _Adapter)

    if export_raises:
        with pytest.raises(RuntimeError), lora_xs.LoRAXS().export_context(wrapped):
            torch.testing.assert_close(adapter.linear_out.weight, expected_out)
            assert isinstance(adapter.activation, nn.Identity)
            raise RuntimeError("export failed")
    else:
        with lora_xs.LoRAXS().export_context(wrapped):
            torch.testing.assert_close(adapter.linear_out.weight, expected_out)
            assert isinstance(adapter.activation, nn.Identity)

    torch.testing.assert_close(adapter.linear_out.weight, original_out)
    assert adapter.activation is original_core


@pytest.mark.parametrize("input_is_parallel", [False, True], ids=["column_parallel", "row_parallel"])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_lora_xs_initialization_reshards_frozen_factors(
    monkeypatch,
    input_is_parallel,
    tp_rank,
):
    torch.manual_seed(3)
    full_weight = torch.randn(12, 8)
    rank = 4
    tp_size = 2
    base_shard_dim = 1 if input_is_parallel else 0
    linear_in_shard_dim = 1 if input_is_parallel else 0
    base_weight = full_weight.chunk(tp_size, dim=base_shard_dim)[tp_rank].clone()
    linear_in_shape = [rank, full_weight.shape[1]]
    linear_in_shape[linear_in_shard_dim] //= tp_size
    core = lora_xs.LoRAXSCore(rank, rank, bias=False)
    nn.init.normal_(core.weight)
    adapter = SimpleNamespace(
        input_is_parallel=input_is_parallel,
        dim=rank,
        alpha=rank,
        linear_in=SimpleNamespace(weight=nn.Parameter(torch.empty(linear_in_shape), requires_grad=False)),
        linear_out=SimpleNamespace(
            weight=nn.Parameter(torch.empty(full_weight.shape[0] // tp_size, rank), requires_grad=False)
        ),
        activation=core,
    )
    base_linear = SimpleNamespace(weight=nn.Parameter(base_weight))
    monkeypatch.setattr(lora_xs_init_worker, "_all_gather", lambda *args: full_weight)
    monkeypatch.setattr(
        lora_xs_init_worker,
        "_synchronized_lora_xs_factors",
        lambda weight, rank, group: lora_xs_init_worker.lora_xs_factors(weight, rank),
    )

    lora_xs_init_worker._init_one_lora_xs_adapter(
        base_linear,
        adapter,
        tp_size,
        tp_rank,
        None,
    )

    linear_in, linear_out = lora_xs_init_worker.lora_xs_factors(full_weight, rank)
    torch.testing.assert_close(base_linear.weight, full_weight.chunk(tp_size, dim=base_shard_dim)[tp_rank])
    torch.testing.assert_close(
        adapter.linear_in.weight,
        linear_in.chunk(tp_size, dim=linear_in_shard_dim)[tp_rank],
    )
    torch.testing.assert_close(adapter.linear_out.weight, linear_out.chunk(tp_size, dim=0)[tp_rank])
    expected_core = torch.zeros(rank, rank)
    torch.testing.assert_close(adapter.activation.weight, expected_core, atol=1e-4, rtol=0)
    assert not torch.equal(adapter.activation.weight, expected_core)
