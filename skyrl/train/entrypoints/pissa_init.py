"""Produce paired PiSSA initialization artifacts with Megatron."""

import argparse
import json
from pathlib import Path

from skyrl.utils.log import logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backend-config", type=json.loads, default={})
    return parser.parse_args()


def _write_manifest(output_dir: Path, base_model: str, rank: int, cfg) -> None:
    lora = cfg.trainer.policy.model.lora
    manifest = {
        "schema_version": 1,
        "source_model": base_model,
        "policy_model_path": "residual_base",
        "resume_path": "global_step_0",
        "reference_model": base_model,
        "rank": rank,
        "alpha": rank,
        "target_modules": lora.target_modules,
        "exclude_modules": lora.exclude_modules,
    }
    with (output_dir / "pissa_init.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def main() -> None:
    args = _parse_args()

    import ray
    import torch

    from skyrl.backends.skyrl_train_backend import (
        MegatronBackendOverrides,
        SkyRLTrainBackend,
    )
    from skyrl.train.config import get_config_as_dict

    if args.rank <= 0:
        raise ValueError("--rank must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    backend_config = dict(args.backend_config)
    backend_config.setdefault("trainer.placement.colocate_all", False)
    if backend_config["trainer.placement.colocate_all"]:
        raise ValueError("PiSSA artifact production requires trainer.placement.colocate_all=false")
    overrides = MegatronBackendOverrides.model_validate(backend_config)
    backend = SkyRLTrainBackend(args.base_model, overrides)
    model_id = "pissa_init"
    try:
        cfg = backend.create_pissa_model(model_id, args.rank)
        checkpoint_dir = args.output_dir / "global_step_0"
        backend.save_checkpoint_directory(str(checkpoint_dir / "policy"), model_id)
        torch.save(
            {"global_step": 0, "config": get_config_as_dict(cfg)},
            checkpoint_dir / "trainer_state.pt",
        )
        backend.save_pissa_residual(str(args.output_dir / "residual_base"), model_id)
        _write_manifest(args.output_dir, args.base_model, args.rank, cfg)
        logger.info(f"PiSSA initialization artifacts saved to {args.output_dir}")
    finally:
        if backend.has_model(model_id):
            backend.delete_model(model_id)
        elif ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
