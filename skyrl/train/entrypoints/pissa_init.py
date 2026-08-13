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
    parser.add_argument("--config-overrides", type=json.loads, default={})
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

    from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import PolicyWorker
    from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
    from skyrl.train.config import SkyRLTrainConfig, get_config_as_dict
    from skyrl.train.utils.utils import initialize_ray
    from skyrl.utils.tok import get_tokenizer

    if args.rank <= 0:
        raise ValueError("--rank must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    overrides = dict(args.config_overrides)
    overrides.update(
        {
            "trainer.strategy": "megatron",
            "trainer.policy.model.path": args.base_model,
            "trainer.policy.model.lora.rank": args.rank,
            "trainer.policy.model.lora.alpha": args.rank,
            "trainer.policy.model.lora.target_modules": "all-linear",
            "trainer.policy.model.lora.exclude_modules": None,
            "trainer.placement.colocate_all": False,
            "trainer.algorithm.use_kl_loss": False,
            "trainer.policy.optimizer_config.scheduler": "constant_with_warmup",
            "trainer.policy.optimizer_config.num_warmup_steps": 0,
        }
    )
    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    tokenizer = get_tokenizer(args.base_model)

    try:
        initialize_ray(cfg)
        policy = PPORayActorGroup(
            cfg.trainer,
            cfg.trainer.placement.policy_num_nodes,
            cfg.trainer.placement.policy_num_gpus_per_node,
            PolicyWorker,
            num_gpus_per_actor=1,
            colocate_all=False,
            sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
            record_memory=cfg.trainer.policy.record_memory,
        )
        ray.get(policy.async_run_ray_method("pass_through", "enable_pissa_init"))
        ray.get(policy.async_init_model(args.base_model, num_training_steps=1e9))
        ray.get(policy.async_run_ray_method("pass_through", "_set_pad_token_id", tokenizer.pad_token_id))
        ray.get(policy.async_run_ray_method("pass_through", "prime_optimizer_state"))

        checkpoint_dir = args.output_dir / "global_step_0"
        ray.get(
            policy.async_run_ray_method(
                "pass_through",
                "save_checkpoint",
                str(checkpoint_dir / "policy"),
                tokenizer,
            )
        )
        torch.save(
            {"global_step": 0, "config": get_config_as_dict(cfg)},
            checkpoint_dir / "trainer_state.pt",
        )
        ray.get(
            policy.async_run_ray_method(
                "pass_through",
                "save_pissa_residual",
                str(args.output_dir / "residual_base"),
                tokenizer,
            )
        )
        _write_manifest(args.output_dir, args.base_model, args.rank, cfg)
        logger.info(f"PiSSA initialization artifacts saved to {args.output_dir}")
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
