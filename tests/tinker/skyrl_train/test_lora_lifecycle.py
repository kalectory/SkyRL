"""The Tinker adapter seed must reach each training backend."""

import pytest

from skyrl.backends.skyrl_train_backend import (
    FSDPBackendOverrides,
    MegatronBackendOverrides,
    _build_skyrl_train_config,
)
from skyrl.tinker import types


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("overrides_class", [FSDPBackendOverrides, MegatronBackendOverrides])
def test_requested_adapter_seed_reaches_training_workers(seed, overrides_class):
    config = _build_skyrl_train_config(
        "test/model",
        overrides_class(**{"trainer.seed": 7}),
        types.LoraConfig(rank=32, alpha=32, seed=seed),
    )
    assert config.trainer.seed == seed
