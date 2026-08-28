"""Task-gradient LoRA initialization and residual-base materialization.

LoRA-GA (Wang et al., 2024, https://arxiv.org/abs/2407.05000) initializes the
two adapter factors from disjoint leading singular subspaces of a task gradient.
This module estimates that gradient from a saved, on-policy GSPO batch and emits
a standard PEFT adapter plus a frozen residual base. Their sum preserves the
source model at initialization.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

LORAGA_DIRECTION = "ArB2r"
LORAGA_STABLE_GAMMA = 16.0
TRAINABLE_STATUSES = {"trainable", "summarization_boundary"}


@dataclasses.dataclass(frozen=True)
class GSPODatum:
    """One autoregressive datum in the saved Trajectory training-batch schema."""

    input_ids: list[int]
    target_ids: list[int]
    token_mask: list[int]
    advantage: float


@dataclasses.dataclass(frozen=True)
class LoraGADecomposition:
    """LoRA-GA factors, identity-preserving residual, and gradient spectrum."""

    linear_in: torch.Tensor
    linear_out: torch.Tensor
    residual: torch.Tensor
    singular_values: torch.Tensor
    captured_energy: float


def decompose_loraga(
    weight: torch.Tensor,
    gradient: torch.Tensor,
    rank: int,
    scale: float,
    *,
    stable_gamma: float = LORAGA_STABLE_GAMMA,
) -> LoraGADecomposition:
    """Initialize rank-``r`` factors from the first ``2r`` gradient modes.

    This is the reference LoRA-GA ``ArB2r`` direction: A uses the first ``r``
    right singular vectors and B uses the next ``r`` left singular vectors.
    Consequently, the first-order adapter tangent covers the first ``2r``
    singular modes even though each factor has rank ``r``.
    """
    if weight.ndim != 2 or gradient.shape != weight.shape:
        raise ValueError(
            f"weight and gradient must be matching matrices, got {tuple(weight.shape)} and {tuple(gradient.shape)}"
        )
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if stable_gamma <= 0:
        raise ValueError(f"stable_gamma must be positive, got {stable_gamma}")

    out_features, in_features = gradient.shape
    max_rank = min(out_features, in_features)
    if 2 * rank > max_rank:
        raise ValueError(
            f"LoRA-GA {LORAGA_DIRECTION} needs 2 * rank <= min(out, in); "
            f"got rank={rank} and gradient shape={tuple(gradient.shape)}"
        )

    gradient_f32 = gradient.float()
    q = min(4 * rank, max_rank)
    u, singular_values, v = torch.svd_lowrank(gradient_f32, q=q, niter=4)

    factor_scale = out_features**0.25 / math.sqrt(stable_gamma)
    linear_in = v[:, :rank].mT.contiguous() * factor_scale
    linear_out = u[:, rank : 2 * rank].contiguous() * factor_scale
    adapter = scale * (linear_out @ linear_in)
    residual = weight.float() - adapter

    gradient_energy = gradient_f32.square().sum().item()
    captured_energy = (
        singular_values[: 2 * rank].square().sum().item() / gradient_energy if gradient_energy > 0 else 0.0
    )
    return LoraGADecomposition(
        linear_in=linear_in,
        linear_out=linear_out,
        residual=residual,
        singular_values=singular_values,
        captured_energy=captured_energy,
    )


def load_gspo_datums(batch_path: str) -> tuple[list[GSPODatum], str, dict[str, Any]]:
    """Load trainable steps and scalar advantages from a saved TrainingBatch."""
    if "://" in batch_path:
        from skyrl.backends.skyrl_train.utils.io.io import open_file

        with open_file(batch_path, "rb") as stream:
            payload_bytes = stream.read()
    else:
        payload_bytes = Path(batch_path).read_bytes()
    payload = json.loads(payload_bytes)

    trajectories = payload["trajectories"]
    advantages = payload["advantages"]
    if len(trajectories) != len(advantages):
        raise ValueError(f"training batch has {len(trajectories)} trajectories but {len(advantages)} advantages")

    datums: list[GSPODatum] = []
    for trajectory, scalar_advantage in zip(trajectories, advantages, strict=True):
        advantage = float(scalar_advantage)
        for step in trajectory["steps"]:
            if step["trainable_status"] not in TRAINABLE_STATUSES:
                continue
            tokens = step["tokens"]
            masks = step["token_masks"]
            if len(tokens) < 2 or len(tokens) != len(masks):
                raise ValueError(f"invalid trainable step lengths: tokens={len(tokens)}, masks={len(masks)}")
            shifted_mask = [int(value) for value in masks[1:]]
            if sum(shifted_mask) == 0:
                raise ValueError("trainable step has no trainable target tokens")
            if step.get("advantages"):
                token_advantages = step["advantages"]
                if len(token_advantages) != len(tokens):
                    raise ValueError("per-token advantages do not align with tokens")
                nonzero = {float(value) for value, mask in zip(token_advantages[1:], shifted_mask, strict=True) if mask}
                if len(nonzero) != 1:
                    raise ValueError("LoRA-GA producer currently requires one scalar advantage per trainable step")
                advantage = nonzero.pop()
            if advantage == 0.0:
                continue
            datums.append(
                GSPODatum(
                    input_ids=[int(token) for token in tokens[:-1]],
                    target_ids=[int(token) for token in tokens[1:]],
                    token_mask=shifted_mask,
                    advantage=advantage,
                )
            )

    if not datums:
        raise ValueError("training batch contains no nonzero-advantage trainable datums")
    summary = {
        "n_trajectories": len(trajectories),
        "n_datums": len(datums),
        "n_trainable_tokens": sum(sum(datum.token_mask) for datum in datums),
        "temperature": float(payload.get("temperature", 1.0)),
    }
    return datums, hashlib.sha256(payload_bytes).hexdigest(), summary


def batch_gspo_datums(datums: list[GSPODatum], max_tokens: int) -> Iterator[list[GSPODatum]]:
    """Length-bucket datums without exceeding the padded-token budget."""
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    current: list[GSPODatum] = []
    current_max_length = 0
    for datum in sorted(datums, key=lambda item: len(item.input_ids)):
        candidate_max = max(current_max_length, len(datum.input_ids))
        candidate_tokens = candidate_max * (len(current) + 1)
        if current and candidate_tokens > max_tokens:
            yield current
            current = []
            current_max_length = 0
        current.append(datum)
        current_max_length = max(current_max_length, len(datum.input_ids))
    if current:
        yield current


def compute_gspo_gradient(
    model: torch.nn.Module,
    datums: list[GSPODatum],
    *,
    pad_token_id: int,
    max_batch_tokens: int,
) -> dict[str, float]:
    """Backpropagate the exact on-policy, step-zero sequence-mean GSPO loss."""
    device = next(model.parameters()).device
    device_type = device.type
    model.train()
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    n_batches = 0

    for batch in batch_gspo_datums(datums, max_batch_tokens):
        max_length = max(len(datum.input_ids) for datum in batch)
        input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long, device=device)
        target_ids = torch.full_like(input_ids, pad_token_id)
        attention_mask = torch.zeros_like(input_ids)
        trainable_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        advantages = torch.tensor([datum.advantage for datum in batch], dtype=torch.float32, device=device)
        for row, datum in enumerate(batch):
            length = len(datum.input_ids)
            input_ids[row, :length] = torch.tensor(datum.input_ids, device=device)
            target_ids[row, :length] = torch.tensor(datum.target_ids, device=device)
            attention_mask[row, :length] = 1
            trainable_mask[row, :length] = torch.tensor(datum.token_mask, dtype=torch.float32, device=device)

        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            negative_logprobs = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                reduction="none",
            ).reshape_as(target_ids)
            mean_logprobs = -(negative_logprobs.float() * trainable_mask).sum(dim=1) / trainable_mask.sum(dim=1)
            loss = -(advantages * mean_logprobs).sum() / len(datums)

        loss.backward()
        total_loss += loss.detach().item()
        n_batches += 1

    return {"loss": total_loss, "n_batches": n_batches}


def _collect_lora_layers(model: torch.nn.Module) -> list[tuple[str, Any]]:
    from peft.tuners.lora.layer import LoraLayer

    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, LoraLayer)]
    if not layers:
        raise ValueError("PEFT did not create any LoRA target layers")
    return layers


def _hash_files(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def materialize_loraga(
    model_path: str,
    batch_path: str,
    output_dir: str,
    *,
    rank: int,
    alpha: int,
    code_sha: str,
    device: str = "cuda",
    max_batch_tokens: int = 8192,
    svd_seed: int = 0,
) -> dict[str, Any]:
    """Estimate target gradients and write a residual base plus initial adapter."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not code_sha:
        raise ValueError("code_sha is required for artifact provenance")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    datums, batch_sha256, batch_summary = load_gspo_datums(batch_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,
    )
    peft_model = get_peft_model(model, lora_config).to(device)
    peft_model.config.use_cache = False
    lora_layers = _collect_lora_layers(peft_model)

    peft_model.requires_grad_(False)
    for _, layer in lora_layers:
        layer.get_base_layer().weight.requires_grad_(True)

    validation_ids = torch.tensor(
        [datums[0].input_ids[: min(len(datums[0].input_ids), 128)]],
        dtype=torch.long,
        device=device,
    )
    peft_model.eval()
    with torch.no_grad():
        reference_logits = peft_model(input_ids=validation_ids, use_cache=False).logits[:, -1].float()

    gradient_metrics = compute_gspo_gradient(
        peft_model,
        datums,
        pad_token_id=tokenizer.pad_token_id,
        max_batch_tokens=max_batch_tokens,
    )

    torch.manual_seed(svd_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(svd_seed)
    scale = alpha / rank
    layer_metrics: dict[str, dict[str, Any]] = {}
    total_gradient_energy = 0.0
    total_captured_energy = 0.0
    max_reconstruction_error = 0.0

    for name, layer in lora_layers:
        base_weight = layer.get_base_layer().weight
        if base_weight.grad is None:
            raise ValueError(f"target layer {name} has no full-weight gradient")
        gradient = base_weight.grad.detach()
        decomposition = decompose_loraga(base_weight.data, gradient, rank, scale)
        layer.lora_A["default"].weight.data.copy_(decomposition.linear_in.to(layer.lora_A["default"].weight))
        layer.lora_B["default"].weight.data.copy_(decomposition.linear_out.to(layer.lora_B["default"].weight))
        base_weight.data.copy_(decomposition.residual.to(base_weight))

        reconstructed = base_weight.data.float() + scale * (
            layer.lora_B["default"].weight.data.float() @ layer.lora_A["default"].weight.data.float()
        )
        expected = decomposition.residual + scale * (decomposition.linear_out @ decomposition.linear_in)
        reconstruction_error = (reconstructed - expected).abs().max().item()
        max_reconstruction_error = max(max_reconstruction_error, reconstruction_error)

        gradient_energy = gradient.float().square().sum().item()
        total_gradient_energy += gradient_energy
        total_captured_energy += decomposition.captured_energy * gradient_energy
        layer_metrics[name] = {
            "shape": list(gradient.shape),
            "gradient_frobenius": math.sqrt(gradient_energy),
            "captured_energy_top_2r": decomposition.captured_energy,
            "singular_values": decomposition.singular_values.detach().cpu().tolist(),
        }
        base_weight.grad = None

    peft_model.eval()
    with torch.no_grad():
        initialized_logits = peft_model(input_ids=validation_ids, use_cache=False).logits[:, -1].float()
    logit_delta = (initialized_logits - reference_logits).abs()
    identity = {
        "parameter_reconstruction_max_abs": max_reconstruction_error,
        "logit_max_abs": logit_delta.max().item(),
        "logit_mean_abs": logit_delta.mean().item(),
    }
    if identity["logit_max_abs"] > 1e-3:
        raise ValueError(f"LoRA-GA initialization failed function-identity check: {identity}")

    peft_model.peft_config["default"].init_lora_weights = True
    adapter_dir = output / "adapter"
    residual_dir = output / "residual_base"
    peft_model.to("cpu")
    peft_model.save_pretrained(adapter_dir, safe_serialization=True)
    residual_model = peft_model.unload()
    residual_model.save_pretrained(residual_dir, safe_serialization=True)
    tokenizer.save_pretrained(residual_dir)

    reloaded_base = AutoModelForCausalLM.from_pretrained(residual_dir, torch_dtype=torch.float32)
    reloaded = PeftModel.from_pretrained(reloaded_base, adapter_dir).eval()
    with torch.no_grad():
        reloaded_logits = reloaded(input_ids=validation_ids.cpu(), use_cache=False).logits[:, -1].float()
    reload_delta = (reloaded_logits - reference_logits.cpu()).abs()
    identity["reload_logit_max_abs"] = reload_delta.max().item()
    identity["reload_logit_mean_abs"] = reload_delta.mean().item()
    if identity["reload_logit_max_abs"] > 1e-3:
        raise ValueError(f"saved LoRA-GA artifacts failed reload identity check: {identity}")

    spectra_path = output / "gradient_spectra.json"
    spectra_path.write_text(json.dumps(layer_metrics, indent=2, sort_keys=True))
    manifest = {
        "schema_version": 1,
        "method": "LoRA-GA",
        "source_model": model_path,
        "source_model_revision": getattr(model.config, "_commit_hash", None),
        "source_batch": batch_path,
        "source_batch_sha256": batch_sha256,
        "skyrl_code_sha": code_sha,
        "rank": rank,
        "alpha": alpha,
        "adapter_scale": scale,
        "target_modules": "all-linear",
        "direction": LORAGA_DIRECTION,
        "stable_gamma": LORAGA_STABLE_GAMMA,
        "svd_seed": svd_seed,
        "gradient_objective": {
            "algorithm": "GSPO",
            "policy_state": "on-policy step 0",
            "importance_sampling_level": "sequence_token",
            "loss_normalization": "sequence_mean",
        },
        "batch": batch_summary,
        "gradient": {
            **gradient_metrics,
            "n_target_layers": len(lora_layers),
            "frobenius": math.sqrt(total_gradient_energy),
            "captured_energy_top_2r": (
                total_captured_energy / total_gradient_energy if total_gradient_energy > 0 else 0.0
            ),
        },
        "identity": identity,
        "saved_dtype": "float32",
    }
    manifest["files_sha256"] = _hash_files(output)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", required=True, help="Saved Trajectory TrainingBatch JSON")
    parser.add_argument("--out", required=True, help="Local directory or gs:// prefix")
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--svd-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    from skyrl.backends.skyrl_train.utils.io.io import exists, upload_directory

    args = _parse_args()
    alpha = args.alpha if args.alpha is not None else args.rank
    if exists(args.out):
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.out}")

    if args.out.startswith(("gs://", "gcs://", "s3://")):
        with tempfile.TemporaryDirectory() as temporary_dir:
            local_output = os.path.join(temporary_dir, "artifact")
            manifest = materialize_loraga(
                args.model,
                args.batch,
                local_output,
                rank=args.rank,
                alpha=alpha,
                code_sha=args.code_sha,
                device=args.device,
                max_batch_tokens=args.max_batch_tokens,
                svd_seed=args.svd_seed,
            )
            upload_directory(local_output, args.out)
    else:
        manifest = materialize_loraga(
            args.model,
            args.batch,
            args.out,
            rank=args.rank,
            alpha=alpha,
            code_sha=args.code_sha,
            device=args.device,
            max_batch_tokens=args.max_batch_tokens,
            svd_seed=args.svd_seed,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
