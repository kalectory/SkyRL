"""PiSSA decomposition."""

import math

import torch


def pissa_decompose(weight: torch.Tensor, rank: int, scale: float):
    """Return PiSSA A, B, and residual factors for a full ``(out, in)`` weight."""
    out_features, in_features = weight.shape
    max_rank = min(out_features, in_features)
    if rank > max_rank:
        raise ValueError(f"PiSSA rank {rank} exceeds min(out, in) = {max_rank} for weight {tuple(weight.shape)}")

    w = weight.float()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]

    sqrt_s = s_r.sqrt()
    inv = 1.0 / math.sqrt(scale)
    linear_out = (u_r * sqrt_s.unsqueeze(0)) * inv  # (out, r)
    linear_in = (sqrt_s.unsqueeze(1) * vh_r) * inv  # (r, in)

    principal = (u_r * s_r.unsqueeze(0)) @ vh_r
    residual = w - principal
    return linear_in, linear_out, residual
