"""Pure tensor utilities for LoRA instrumentation."""

import torch


def compute_effective_lora_delta_squared(
    linear_in: torch.Tensor,
    linear_out: torch.Tensor,
    reference_linear_in: torch.Tensor,
    reference_linear_out: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Return ``||scale * (B A - B0 A0)||_F^2`` without materializing the weight.

    Expanding around the reference factors avoids subtracting two large, nearly
    equal products. The remaining products are rank-by-rank Gram matrices, so
    instrumentation cost scales with the LoRA rank rather than the dense weight.
    """
    a = linear_in.double()
    b = linear_out.double()
    a0 = reference_linear_in.double()
    b0 = reference_linear_out.double()
    da = a - a0
    db = b - b0

    db_a_squared = torch.trace((db.mT @ db) @ (a @ a.mT))
    b0_da_squared = torch.trace((b0.mT @ b0) @ (da @ da.mT))
    cross = 2 * torch.trace((db.mT @ b0) @ (da @ a.mT))
    return ((scale**2) * (db_a_squared + b0_da_squared + cross)).clamp_min(0)
