"""CPU unit tests for PiSSA decomposition math (no Megatron/GPU needed)."""

import math

import pytest
import torch

from skyrl.backends.skyrl_train.workers.megatron.pissa_init import (
    PissaConfig,
    bridge_a_init_method,
    pissa_decompose,
)


def _principal(weight, rank):
    """Reference top-r reconstruction via SVD."""
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
    # Residual is exactly the non-principal part.
    assert torch.allclose(residual, w - _principal(w, rank), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("alpha,rank", [(8, 16), (32, 16), (16, 16)])
def test_scale_invariance(alpha, rank):
    """The merged delta scale*B@A must reproduce the principal component for any alpha/rank."""
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
    """A and B should carry equal singular-value mass (Sᵣ^½ split), per the PiSSA init."""
    torch.manual_seed(4)
    w = torch.randn(32, 24)
    rank = 6
    linear_in, linear_out, _ = pissa_decompose(w, rank, 1.0)
    # Column norms of B (out, r) and row norms of A (r, in) both equal sqrt(singular value).
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    expected = s[:rank].sqrt()
    assert torch.allclose(linear_out.norm(dim=0), expected, atol=1e-4)
    assert torch.allclose(linear_in.norm(dim=1), expected, atol=1e-4)
    assert math.isclose(linear_out.norm().item(), linear_in.norm().item(), rel_tol=1e-4)


@pytest.mark.parametrize("method,niter", [("pissa", 0), ("pissa_niter_4", 4), ("pissa_niter_16", 16)])
def test_pissa_config_parses_method(method, niter):
    cfg = PissaConfig.from_init_method(method)
    assert cfg == PissaConfig(niter=niter)


@pytest.mark.parametrize("method", ["kaiming", "xavier", "normal", "zero"])
def test_non_pissa_init_method_parses_to_none(method):
    assert PissaConfig.from_init_method(method) is None
    # Non-PiSSA methods pass through to the bridge unchanged; PiSSA resolves to a placeholder.
    assert bridge_a_init_method(method) == method


@pytest.mark.parametrize("method", ["pissa", "pissa_niter_4"])
def test_bridge_a_init_method_placeholder_for_pissa(method):
    assert bridge_a_init_method(method) == "kaiming"


def test_niter_reconstructs_weight_for_dominant_spectrum():
    """Fast randomized SVD (niter>0) still yields W_res + scale·B·A == W on a low-rank-ish matrix."""
    torch.manual_seed(5)
    # Construct a matrix whose top-r subspace dominates so randomized SVD is accurate.
    out, in_, rank = 64, 48, 8
    base = (torch.randn(out, rank) * torch.arange(rank, 0, -1).float()) @ torch.randn(rank, in_)
    w = base + 0.01 * torch.randn(out, in_)
    linear_in, linear_out, residual = pissa_decompose(w, rank, 1.0, niter=8)
    assert linear_in.shape == (rank, in_) and linear_out.shape == (out, rank)
    merged = residual + 1.0 * (linear_out @ linear_in)
    assert torch.allclose(merged, w, atol=1e-4, rtol=1e-4)
