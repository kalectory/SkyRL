"""PiSSA: principal-SVD LoRA initialization — pure math + offline materialization.

PiSSA (Meng et al., 2024, https://arxiv.org/abs/2404.02948) seeds LoRA A/B from
the base weight's principal singular components and freezes the residual
``W_res = W - A·B`` in place of ``W``, so the adapter starts on the principal
subspace and ``W_res + A·B == W`` at init.

This module has no GPU / distributed / megatron dependencies so it is usable on
CPU and offline. Two consumers build on it:
  - Runtime Megatron application:
    ``skyrl/backends/skyrl_train/workers/megatron/pissa_init.py`` (TP-aware, in-worker).
  - Offline materialization (``materialize_pissa`` here): write a residual base
    model + principal adapter as standard HF/PEFT artifacts, so a multi-tenant
    LoRA deployment serves PiSSA as plain LoRA over a frozen base (no runtime
    base mutation, no weight-sync changes).
"""

import dataclasses
import logging
import math
from typing import Optional

import torch

logger = logging.getLogger(__name__)

PISSA_PREFIX = "pissa"


@dataclasses.dataclass(frozen=True)
class PissaConfig:
    """Parsed PiSSA initialization options.

    Carries everything the init needs as typed fields; the ``init_method`` string
    is parsed once (``from_init_method``) and the object is passed around thereafter.
    """

    niter: int = 0  # 0 = exact SVD; >0 = torch.svd_lowrank subspace iterations

    @classmethod
    def from_init_method(cls, init_method: str) -> Optional["PissaConfig"]:
        """Parse an ``init_method`` string into a PissaConfig, or None if not PiSSA.

        Mirrors PEFT: "pissa" -> exact SVD; "pissa_niter_<N>" -> fast randomized SVD.
        """
        if init_method == PISSA_PREFIX:
            return cls(niter=0)
        if init_method.startswith(f"{PISSA_PREFIX}_niter_"):
            return cls(niter=int(init_method.rsplit("_", 1)[1]))
        return None


def pissa_decompose(weight: torch.Tensor, rank: int, scale: float, niter: int = 0):
    """Decompose a base weight into PiSSA adapter factors + frozen residual.

    Pure tensor math, unit-testable on CPU. Operates on the full, unsharded weight
    in ``(out_features, in_features)`` layout (torch ``nn.Linear`` convention, ``y = x Wᵀ``).

    Args:
      weight: full base weight, shape (out, in). Upcast to float32 internally.
      rank: LoRA rank r (number of principal components to keep).
      scale: the adapter's ``alpha/dim`` forward scaling factor.
      niter: 0 for exact SVD; >0 uses ``torch.svd_lowrank`` with this many subspace
        iterations (the paper's fast-SVD variant, "pissa_niter_N").

    Returns:
      (linear_in, linear_out, residual) all float32:
        linear_in:  (r, in)   — adapter A
        linear_out: (out, r)  — adapter B
        residual:   (out, in) — frozen base, == weight − scale·linear_out·linear_in
    """
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

    principal = (u_r * s_r.unsqueeze(0)) @ vh_r  # (out, in) == scale·linear_out·linear_in
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
    """Write a PiSSA residual base model + principal adapter as HF/PEFT artifacts.

    Decomposes ``model_path`` once (offline, full-rank) and emits:
      - ``<out_dir>/residual_base``: an HF model whose target weights are W_res.
      - ``<out_dir>/adapter``: the principal A/B as a PEFT LoRA adapter, such that
        ``residual_base + adapter == model_path`` at load time.

    A deployment then serves PiSSA as plain LoRA over ``residual_base`` (the
    trainer and inference engine load the same base), and the run starts the
    adapter from ``adapter`` — so multi-tenant ``merge_lora=false`` works with no
    runtime base mutation or weight-sync changes.

    Returns (residual_base_dir, adapter_dir).
    """
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
    # Stop PiSSA's SVD from re-running when the adapter is reloaded for training/serving.
    peft_model.peft_config["default"].init_lora_weights = True

    adapter_dir = os.path.join(out_dir, "adapter")
    residual_base_dir = os.path.join(out_dir, "residual_base")
    peft_model.save_pretrained(adapter_dir)  # principal A/B
    residual_model = peft_model.unload()  # strip adapters -> base_layer holds W_res
    residual_model.save_pretrained(residual_base_dir)
    # Convenience copy so residual_base is servable; weights (the load-bearing
    # output) are already written, so a tokenizer-less source is non-fatal.
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
