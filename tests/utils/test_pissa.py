"""Tests for PiSSA decomposition and offline materialization."""

import math

import pytest
import torch

from skyrl.utils.pissa import PissaConfig, materialize_pissa, pissa_decompose


def _principal(weight, rank):
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return (u[:, :rank] * s[:rank].unsqueeze(0)) @ vh[:rank, :]


@pytest.mark.parametrize("shape", [(64, 48), (48, 64), (32, 32)])
@pytest.mark.parametrize("rank", [4, 16])
def test_residual_plus_adapter_reconstructs_weight(shape, rank):
    torch.manual_seed(0)
    w = torch.randn(*shape)
    scale = 1.0  # alpha == rank
    linear_in, linear_out, residual = pissa_decompose(w, rank, scale)

    assert linear_in.shape == (rank, shape[1])
    assert linear_out.shape == (shape[0], rank)
    assert residual.shape == tuple(shape)

    merged = residual + scale * (linear_out @ linear_in)
    assert torch.allclose(merged, w, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("rank", [1, 8, 30])
def test_adapter_equals_principal_components(rank):
    torch.manual_seed(1)
    w = torch.randn(40, 30)
    scale = 1.0
    linear_in, linear_out, residual = pissa_decompose(w, rank, scale)

    principal = scale * (linear_out @ linear_in)
    assert torch.allclose(principal, _principal(w, rank), atol=1e-4, rtol=1e-4)
    assert torch.allclose(residual, w - _principal(w, rank), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("alpha,rank", [(8, 16), (32, 16), (16, 16)])
def test_scale_invariance(alpha, rank):
    torch.manual_seed(2)
    w = torch.randn(50, 50)
    scale = alpha / rank
    linear_in, linear_out, residual = pissa_decompose(w, rank, scale)

    merged = residual + scale * (linear_out @ linear_in)
    assert torch.allclose(merged, w, atol=1e-4, rtol=1e-4)
    assert torch.allclose(scale * (linear_out @ linear_in), _principal(w, rank), atol=1e-4, rtol=1e-4)


def test_full_rank_residual_is_zero():
    torch.manual_seed(3)
    w = torch.randn(20, 20)
    _, _, residual = pissa_decompose(w, 20, 1.0)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-4)


def test_rank_too_large_raises():
    w = torch.randn(8, 12)
    with pytest.raises(ValueError):
        pissa_decompose(w, 9, 1.0)  # 9 > min(8, 12)


def test_factor_symmetry():
    torch.manual_seed(4)
    w = torch.randn(32, 24)
    rank = 6
    linear_in, linear_out, _ = pissa_decompose(w, rank, 1.0)
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    expected = s[:rank].sqrt()
    assert torch.allclose(linear_out.norm(dim=0), expected, atol=1e-4)
    assert torch.allclose(linear_in.norm(dim=1), expected, atol=1e-4)
    assert math.isclose(linear_out.norm().item(), linear_in.norm().item(), rel_tol=1e-4)


@pytest.mark.parametrize("method,niter", [("pissa", 0), ("pissa_niter_4", 4), ("pissa_niter_16", 16)])
def test_pissa_config_parses_method(method, niter):
    assert PissaConfig.from_init_method(method) == PissaConfig(niter=niter)


@pytest.mark.parametrize("method", ["kaiming", "xavier", "normal", "zero"])
def test_non_pissa_init_method_parses_to_none(method):
    assert PissaConfig.from_init_method(method) is None


def test_niter_reconstructs_weight():
    torch.manual_seed(5)
    out, in_, rank = 64, 48, 8
    base = (torch.randn(out, rank) * torch.arange(rank, 0, -1).float()) @ torch.randn(rank, in_)
    w = base + 0.01 * torch.randn(out, in_)
    linear_in, linear_out, residual = pissa_decompose(w, rank, 1.0, niter=8)
    assert linear_in.shape == (rank, in_) and linear_out.shape == (out, rank)
    merged = residual + 1.0 * (linear_out @ linear_in)
    assert torch.allclose(merged, w, atol=1e-4, rtol=1e-4)


def test_materialize_pissa_is_identity_at_init(tmp_path):
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    config = transformers.LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    src = str(tmp_path / "src")
    model.save_pretrained(src)

    ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        ref = model(ids).logits

    base_dir, adapter_dir = materialize_pissa(src, rank=8, out_dir=str(tmp_path / "out"), dtype="float32")

    base = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float32)
    reconstructed = PeftModel.from_pretrained(base, adapter_dir).eval()
    with torch.no_grad():
        out = reconstructed(ids).logits
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
