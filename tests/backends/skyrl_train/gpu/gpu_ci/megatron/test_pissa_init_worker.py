"""Tests for PiSSA decomposition and tensor-parallel initialization."""

import json
from types import SimpleNamespace

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.workers.megatron import pissa_init_worker
from skyrl.backends.skyrl_train.workers.megatron.pissa_init_worker import (
    PiSSAInitWorker,
)
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.entrypoints.pissa_init import _write_manifest
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_worker import (
    MODEL_NAME,
    get_test_actor_config,
    get_test_training_batch,
)
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    ray_init_for_tests,
)

pytestmark = pytest.mark.megatron


def _principal(weight, rank):
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return (u[:, :rank] * s[:rank].unsqueeze(0)) @ vh[:rank, :]


@pytest.mark.parametrize("shape", [(64, 48), (48, 64), (32, 32)])
def test_pissa_decomposition_uses_principal_components(shape):
    torch.manual_seed(0)
    w = torch.randn(*shape)
    rank = 4
    scale = 0.5
    linear_in, linear_out, residual = pissa_init_worker.pissa_decompose(w, rank, scale)

    principal = scale * (linear_out @ linear_in)
    torch.testing.assert_close(principal, _principal(w, rank), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(residual + principal, w, atol=1e-6, rtol=1e-6)


def test_rank_too_large_raises():
    w = torch.randn(8, 12)
    with pytest.raises(ValueError):
        pissa_init_worker.pissa_decompose(w, 9, 1.0)  # 9 > min(8, 12)


@pytest.mark.parametrize(
    ("rank", "lora_type"),
    [
        (0, "lora"),
        (32, "canonical_lora"),
    ],
)
def test_invalid_pissa_config_raises(rank, lora_type):
    with pytest.raises(ValueError):
        pissa_init_worker.validate_pissa_config(rank, lora_type)


def test_valid_pissa_config_is_accepted():
    pissa_init_worker.validate_pissa_config(32, "lora")


def test_pissa_manifest_records_source_model_and_rank(tmp_path):
    _write_manifest(tmp_path, "Qwen/test-model", 32)

    manifest = json.loads((tmp_path / "pissa_init.json").read_text())
    assert manifest == {
        "schema_version": 1,
        "source_model": "Qwen/test-model",
        "rank": 32,
    }


@pytest.mark.parametrize("export_raises", [False, True])
def test_residual_export_restores_adapter_weights(monkeypatch, export_raises):
    class ParallelLinearAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_out = torch.nn.Linear(3, 4, bias=False)

    class LoRALinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = ParallelLinearAdapter()

    monkeypatch.setattr(pissa_init_worker, "LoRALinear", LoRALinear)
    monkeypatch.setattr(pissa_init_worker, "ParallelLinearAdapter", ParallelLinearAdapter)

    model = torch.nn.Sequential(LoRALinear())
    weight = model[0].adapter.linear_out.weight
    original = weight.detach().clone()
    original_device = weight.device

    if export_raises:
        with pytest.raises(RuntimeError), pissa_init_worker.zeroed_adapters(model):
            torch.testing.assert_close(weight, torch.zeros_like(weight))
            assert weight.device == original_device
            raise RuntimeError("export failed")
    else:
        with pissa_init_worker.zeroed_adapters(model):
            torch.testing.assert_close(weight, torch.zeros_like(weight))
            assert weight.device == original_device

    torch.testing.assert_close(weight, original)


@pytest.mark.parametrize("input_is_parallel", [False, True], ids=["column_parallel", "row_parallel"])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_pissa_initialization_reshards_parallel_adapters(monkeypatch, input_is_parallel, tp_rank):
    torch.manual_seed(6)
    full_weight = torch.randn(12, 8)
    rank = 4
    tp_size = 2
    base_shard_dim = 1 if input_is_parallel else 0
    linear_in_shard_dim = 1 if input_is_parallel else 0
    base_weight = full_weight.chunk(tp_size, dim=base_shard_dim)[tp_rank].clone()
    linear_in_shape = [rank, full_weight.shape[1]]
    linear_in_shape[linear_in_shard_dim] //= tp_size

    base_linear = SimpleNamespace(weight=torch.nn.Parameter(base_weight))
    adapter = SimpleNamespace(
        input_is_parallel=input_is_parallel,
        dim=rank,
        alpha=rank,
        linear_in=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(linear_in_shape))),
        linear_out=SimpleNamespace(weight=torch.nn.Parameter(torch.empty(full_weight.shape[0] // tp_size, rank))),
    )
    monkeypatch.setattr(pissa_init_worker, "_all_gather", lambda *args: full_weight)

    pissa_init_worker._init_one_adapter(base_linear, adapter, tp_size, tp_rank, None)

    linear_in, linear_out, residual = pissa_init_worker.pissa_decompose(full_weight, rank, scale=1.0)
    torch.testing.assert_close(base_linear.weight, residual.chunk(tp_size, dim=base_shard_dim)[tp_rank])
    torch.testing.assert_close(adapter.linear_in.weight, linear_in.chunk(tp_size, dim=linear_in_shard_dim)[tp_rank])
    torch.testing.assert_close(adapter.linear_out.weight, linear_out.chunk(tp_size, dim=0)[tp_rank])


@pytest.mark.asyncio
async def test_pissa_worker_preserves_base_model_forward(ray_init_fixture):
    batch = get_test_training_batch(4)

    def base_cfg():
        cfg = get_test_actor_config(model_name=MODEL_NAME)
        cfg.trainer.strategy = "megatron"
        cfg.trainer.placement.policy_num_gpus_per_node = 4
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
        cfg.trainer.bf16 = False
        return cfg

    def megatron_forward(cfg, worker_cls=None):
        if worker_cls is None:
            actor_group = init_worker_with_type(
                "policy",
                shared_pg=None,
                colocate_all=False,
                num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
                cfg=cfg,
            )
        else:
            actor_group = PPORayActorGroup(
                cfg.trainer,
                num_nodes=1,
                num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
                ray_actor_type=worker_cls,
                num_gpus_per_actor=0.75,
                colocate_all=False,
                sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
                record_memory=cfg.trainer.policy.record_memory,
            )
            ray.get(actor_group.async_init_model(cfg.trainer.policy.model.path))
        all_rank = ray.get(actor_group.async_run_ray_method("mesh", "forward", data=batch))
        output = WorkerOutput.cat(actor_group.actor_infos, all_rank)
        return loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")

    pissa_cfg = base_cfg()
    pissa_cfg.trainer.policy.model.lora.rank = 32
    pissa_cfg.trainer.policy.model.lora.alpha = 32
    logprobs_pissa = megatron_forward(pissa_cfg, worker_cls=PiSSAInitWorker)

    ray.shutdown()
    ray_init_for_tests()

    logprobs_base = megatron_forward(base_cfg())
    torch.testing.assert_close(logprobs_pissa, logprobs_base, rtol=1e-4, atol=1e-4)
