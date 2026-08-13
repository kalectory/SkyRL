"""PiSSA decomposition and offline residual-base materialization."""

import dataclasses
import logging
import math
from typing import Optional

import torch

logger = logging.getLogger(__name__)

PISSA_PREFIX = "pissa"


@dataclasses.dataclass(frozen=True)
class PissaConfig:
    """Parsed PiSSA initialization options."""

    niter: int = 0  # 0 = exact SVD; >0 = torch.svd_lowrank subspace iterations

    @classmethod
    def from_init_method(cls, init_method: str) -> Optional["PissaConfig"]:
        """Parse PEFT-compatible PiSSA initialization names."""
        if init_method == PISSA_PREFIX:
            return cls(niter=0)
        if init_method.startswith(f"{PISSA_PREFIX}_niter_"):
            return cls(niter=int(init_method.rsplit("_", 1)[1]))
        return None


def pissa_decompose(weight: torch.Tensor, rank: int, scale: float, niter: int = 0):
    """Return PiSSA A, B, and residual factors for a full ``(out, in)`` weight."""
    out_features, in_features = weight.shape
    max_rank = min(out_features, in_features)
    if rank > max_rank:
        raise ValueError(f"PiSSA rank {rank} exceeds min(out, in) = {max_rank} for weight {tuple(weight.shape)}")

    w = weight.float()
    if niter > 0:
        u_r, s_r, v_r = torch.svd_lowrank(w, q=rank, niter=niter)  # u:(out,r) s:(r) v:(in,r)
        vh_r = v_r.mH
    else:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)  # u:(out,k) s:(k) vh:(k,in)
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


def materialize_pissa(
    model_path: str,
    rank: int,
    out_dir: str,
    alpha: Optional[int] = None,
    target_modules: str = "all-linear",
    niter: Optional[int] = None,
    dtype: str = "bfloat16",
) -> tuple[str, str]:
    """Write a residual base and principal adapter as HF/PEFT artifacts."""
    import os

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    init_lora_weights = PISSA_PREFIX if niter is None else f"{PISSA_PREFIX}_niter_{niter}"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=getattr(torch, dtype))
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha if alpha is not None else rank,  # PiSSA recipe: alpha == rank
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=init_lora_weights,
    )
    peft_model = get_peft_model(model, lora_config)
    # Reload the materialized factors without running PiSSA again.
    peft_model.peft_config["default"].init_lora_weights = True

    adapter_dir = os.path.join(out_dir, "adapter")
    residual_base_dir = os.path.join(out_dir, "residual_base")
    peft_model.save_pretrained(adapter_dir)
    residual_model = peft_model.unload()
    residual_model.save_pretrained(residual_base_dir)
    try:
        AutoTokenizer.from_pretrained(model_path).save_pretrained(residual_base_dir)
    except (OSError, ValueError) as exc:
        logger.warning(
            "PiSSA: could not copy tokenizer from %s (%s); residual base written without it.", model_path, exc
        )
    return residual_base_dir, adapter_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Materialize a PiSSA residual base model + principal adapter.")
    parser.add_argument("--model", required=True, help="HF model id or local path to decompose.")
    parser.add_argument("--rank", type=int, required=True, help="LoRA rank r.")
    parser.add_argument("--out", required=True, help="Output dir (writes residual_base/ and adapter/).")
    parser.add_argument("--alpha", type=int, default=None, help="LoRA alpha (defaults to rank, the PiSSA recipe).")
    parser.add_argument("--target-modules", default="all-linear", help="PEFT target_modules (default all-linear).")
    parser.add_argument("--niter", type=int, default=None, help="Fast-SVD iterations; omit for exact SVD.")
    parser.add_argument("--dtype", default="bfloat16", help="torch dtype for the saved weights.")
    args = parser.parse_args()

    base_dir, adapter_dir = materialize_pissa(
        args.model,
        args.rank,
        args.out,
        alpha=args.alpha,
        target_modules=args.target_modules,
        niter=args.niter,
        dtype=args.dtype,
    )
    print(f"residual base: {base_dir}\nadapter:       {adapter_dir}")


if __name__ == "__main__":
    main()
