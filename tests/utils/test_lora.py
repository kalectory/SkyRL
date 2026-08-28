"""CPU tests for LoRA instrumentation math."""

import pytest
import torch

from skyrl.utils.lora import compute_effective_lora_delta_squared


@pytest.mark.parametrize("shape,rank,scale", [((17, 11), 3, 1.0), ((11, 17), 5, 0.25)])
@pytest.mark.parametrize("zero_reference_out", [False, True])
def test_effective_lora_delta_matches_materialized_weight(shape, rank, scale, zero_reference_out):
    torch.manual_seed(0)
    out_features, in_features = shape
    a0 = torch.randn(rank, in_features)
    b0 = torch.randn(out_features, rank)
    if zero_reference_out:
        b0.zero_()
    a = a0 + 0.01 * torch.randn_like(a0)
    b = b0 + 0.01 * torch.randn_like(b0)

    actual = compute_effective_lora_delta_squared(a, b, a0, b0, scale)
    expected = (scale * (b.double() @ a.double() - b0.double() @ a0.double())).square().sum()

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_effective_lora_delta_is_zero_at_reference():
    torch.manual_seed(1)
    a = torch.randn(4, 9)
    b = torch.randn(7, 4)

    actual = compute_effective_lora_delta_squared(a, b, a.clone(), b.clone(), 2.0)

    assert actual.item() == 0
