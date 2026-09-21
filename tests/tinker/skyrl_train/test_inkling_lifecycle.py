"""Tinker FSDP seed and durable adapter contracts; requires the SkyRL training environment."""

import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from skyrl.backends.skyrl_train_backend import (
    FSDPBackendOverrides,
    SkyRLTrainBackend,
    _build_skyrl_train_config,
)
from skyrl.tinker import types


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_requested_adapter_seed_reaches_training_workers(seed):
    config = _build_skyrl_train_config(
        "thinkingmachines/Inkling-Small",
        FSDPBackendOverrides(**{"trainer.seed": 7}),
        types.LoraConfig(rank=32, alpha=32, seed=seed),
    )
    assert config.trainer.seed == seed


def test_fsdp_durable_sampler_contains_live_adapter_without_exporting_base(tmp_path):
    model_id = "model-test"
    adapter_root = tmp_path / "adapters"
    adapter_dir = adapter_root / model_id
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter_dir / "adapter_config.json").write_text('{"target_modules":["wq_du"]}')
    backend = object.__new__(SkyRLTrainBackend)
    backend._cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            strategy="fsdp",
            policy=SimpleNamespace(
                model=SimpleNamespace(lora=SimpleNamespace(rank=32, lora_sync_path=str(adapter_root)))
            ),
        )
    )
    backend._validate_model_state = Mock()
    backend._get_role = Mock(return_value="policy")
    backend._ensure_inference_engines = Mock()
    backend._base_lora_signature = (32, 32)
    backend._dispatch = Mock()
    backend._dispatch.save_weights_for_sampler = AsyncMock()
    backend._inference_adapter_ids = set()
    output = tmp_path / "sampler.tar"
    backend.save_sampler_checkpoint(str(output), model_id, persist=True)
    backend._dispatch.save_hf_model.assert_not_called()
    backend._dispatch.save_weights_for_sampler.assert_awaited_once_with(model_id=model_id)
    with tarfile.open(output) as archive:
        assert {name for name in archive.getnames() if name != "."} == {
            "./adapter_config.json",
            "./adapter_model.safetensors",
        }
