import copy
import inspect
import json
from unittest.mock import patch

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from transformers import AutoConfig, AutoModelForImageTextToText, InklingConfig
from transformers.models.inkling import modeling_inkling

from skyrl.backends.skyrl_train.distributed.lora_checkpoint import (
    load_trainable_state_dict,
    trainable_state_dict,
)
from skyrl.backends.skyrl_train.patches.inkling import (
    correct_inkling_expert_size,
    inkling_fp32_modules,
    inkling_lora_for_inference,
    install_inkling_precision,
    validate_inkling_lora_targets,
)
from skyrl.backends.skyrl_train.patches.inkling_attention import (
    _compact_flex_attention,
    install_inkling_flex_attention,
)


@pytest.fixture(autouse=True)
def native_torch_convolution_for_cpu_tests(monkeypatch, request):
    if request.node.name.startswith("test_fsdp2_"):
        return
    # GPU images install causal_conv1d, whose extension accepts CUDA tensors only.
    # CPU reference tests use the native undecorated implementation; GPU tests
    # retain the installed extension and production dispatch.
    for name in ("causal_conv1d_fn", "causal_conv1d_update"):
        monkeypatch.setattr(modeling_inkling, name, inspect.unwrap(getattr(modeling_inkling, name)))


def tiny_config(hidden_size=32):
    config = InklingConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": hidden_size,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": hidden_size // 4,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": hidden_size // 4,
            "d_rel": 4,
            "rel_extent": 16,
            "local_layer_ids": [0],
            "sliding_window_size": 4,
            "mlp_layer_types": ["dense", "sparse"],
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 2,
            "logits_mup_width_multiplier": 1.0,
            "pad_token_id": 0,
        },
        audio_config={"n_mel_bins": 4, "mel_vocab_size": 16, "text_hidden_size": hidden_size},
        vision_config={
            "text_hidden_size": hidden_size,
            "patch_size": 4,
            "temporal_patch_size": 2,
            "num_hidden_layers": 2,
        },
    )
    config._attn_implementation = "sdpa"
    return config


def test_published_expert_width_controls_loaded_tensor_shape(tmp_path):
    published = tiny_config().to_dict()
    published["text_config"].pop("moe_intermediate_size")
    published["text_config"].update(intermediate_size=16, dense_intermediate_size=64)
    (tmp_path / "config.json").write_text(json.dumps(published))
    config = AutoConfig.from_pretrained(tmp_path)
    assert config.text_config.moe_intermediate_size == 3072
    correct_inkling_expert_size(config, tmp_path, {})
    model = AutoModelForImageTextToText.from_config(config)
    assert model.model.language_model.layers[1].mlp.experts.gate_up_proj.shape == (4, 32, 32)
    config.text_config.moe_intermediate_size = 24
    correct_inkling_expert_size(config, tmp_path, {"text_config": {"moe_intermediate_size": 24}})
    assert config.text_config.moe_intermediate_size == 24


def test_router_precision_survives_checkpoint_meta_init_and_shared_expert_backward(tmp_path):
    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

    torch.manual_seed(17)
    torch.set_num_threads(2)
    original = AutoModelForImageTextToText.from_config(tiny_config(), dtype=torch.bfloat16)
    gate = original.model.language_model.layers[1].mlp.gate
    with torch.no_grad():
        gate.weight.normal_(std=1e-5)
        gate.e_score_correction_bias.data = torch.tensor([0.5004, 0.5003, 0.5002, 0.5001])
        gate.global_scale.data = torch.tensor([1.0001234])
    expected_bias = gate.e_score_correction_bias.detach().clone()
    expected_scale = gate.global_scale.detach().clone()
    assert not torch.equal(expected_bias, expected_bias.bfloat16().float())
    original.save_pretrained(tmp_path)

    options = dict(bf16=True, lora_rank=4, target_modules=["q_proj", "r_proj", "o_proj"])
    loaded = HFModelWrapper(str(tmp_path), **options).model
    meta = HFModelWrapper(str(tmp_path), meta_init=True, **options).model
    loaded_dtypes = {name: parameter.dtype for name, parameter in loaded.named_parameters()}
    meta_dtypes = {name: parameter.dtype for name, parameter in meta.named_parameters()}
    assert loaded_dtypes == meta_dtypes
    dense_gate = next(name for name in loaded_dtypes if name.endswith("layers.0.mlp.gate_proj.weight"))
    assert loaded_dtypes[dense_gate] == torch.bfloat16
    router = next(module for module in loaded.modules() if isinstance(module, modeling_inkling.InklingTopkRouter))
    meta_router = next(module for module in meta.modules() if isinstance(module, modeling_inkling.InklingTopkRouter))
    assert all(parameter.dtype == torch.float32 for parameter in router.parameters())
    assert all(parameter.is_meta and parameter.dtype == torch.float32 for parameter in meta_router.parameters())
    torch.testing.assert_close(router.e_score_correction_bias, expected_bias, rtol=0, atol=0)
    torch.testing.assert_close(router.global_scale, expected_scale, rtol=0, atol=0)
    hidden = torch.randn(2, 32, dtype=torch.bfloat16, requires_grad=True)
    expected = modeling_inkling.InklingTopkRouter.forward(router, hidden.float())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        observed = router(hidden)
    for actual, reference in zip(observed, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert observed[0].dtype == observed[1].dtype == observed[3].dtype == torch.float32
    assert observed[2].sort().values.tolist() == [[0, 1], [0, 1]]
    # Production FSDP does not provide an outer autocast. FP32 gammas must be
    # multiplied before casting the shared activation for its BF16 down GEMM.
    moe = next(module for module in loaded.modules() if isinstance(module, modeling_inkling.InklingMoE))
    output = moe(hidden)
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
    output.float().square().sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0


def test_sdpa_matches_eager_and_padding_preserves_valid_token_logits():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    sdpa = AutoModelForImageTextToText.from_config(tiny_config()).eval()
    eager = copy.deepcopy(sdpa)
    eager.set_attn_implementation("eager")
    tokens = torch.tensor([[10, 11, 12, 13, 14, 15]])
    with torch.no_grad():
        expected = sdpa(tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
        eager_logits = eager(tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
        torch.testing.assert_close(expected, eager_logits, atol=1e-5, rtol=1e-4)
        for left, right in [(0, 3), (3, 0), (2, 2)]:
            padded = torch.nn.functional.pad(tokens, (left, right))
            attention_mask = padded.ne(0)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(~attention_mask, 1)
            observed = sdpa(padded, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).logits[
                :, left : left + tokens.shape[1]
            ]
            torch.testing.assert_close(expected, observed, atol=1e-5, rtol=1e-4)


def test_grouped_experts_match_eager_forward_and_backward():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    grouped = AutoModelForImageTextToText.from_config(tiny_config())
    assert grouped.config.text_config._experts_implementation == "grouped_mm"
    eager = copy.deepcopy(grouped)
    eager.set_experts_implementation("eager")
    tokens = torch.tensor([[10, 11, 12, 13, 14, 15]])
    grouped_output = grouped(tokens, labels=tokens, use_cache=False)
    eager_output = eager(tokens, labels=tokens, use_cache=False)
    torch.testing.assert_close(grouped_output.logits, eager_output.logits, atol=1e-5, rtol=1e-4)
    grouped_output.loss.backward()
    eager_output.loss.backward()
    for (_, grouped_parameter), (_, eager_parameter) in zip(grouped.named_parameters(), eager.named_parameters()):
        if grouped_parameter.grad is not None:
            torch.testing.assert_close(grouped_parameter.grad, eager_parameter.grad, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_compact_attention_matches_native_multimodal_text_forward(padding_side):
    torch.manual_seed(17)
    config = tiny_config()
    config.text_config.log_scaling_n_floor = 4
    native = AutoModelForImageTextToText.from_config(config).eval()
    for name, parameter in native.named_parameters():
        if "rel_logits_proj.proj" in name:
            torch.nn.init.normal_(parameter, std=0.3)
    compact = copy.deepcopy(native)
    install_inkling_flex_attention(compact)
    assert compact.config._attn_implementation == "sdpa"
    assert compact.config.text_config._attn_implementation == "skyrl_inkling_flex_attention"
    tokens = torch.randint(1, 128, (2, 12))
    padding = slice(0, 3) if padding_side == "left" else slice(-3, None)
    tokens[1, padding] = 0
    mask = tokens.ne(0)
    with torch.no_grad():
        expected = native(tokens, attention_mask=mask, position_ids=None, use_cache=False).logits
        observed = compact(tokens, attention_mask=mask, position_ids=None, use_cache=False).logits
    torch.testing.assert_close(expected[mask], observed[mask], atol=1e-5, rtol=1e-4)


def test_bf16_frozen_base_keeps_fp32_adapters_gradients_and_optimizer():
    torch.manual_seed(17)
    base_model = AutoModelForImageTextToText.from_config(tiny_config(), dtype=torch.bfloat16)
    fp32_modules = inkling_fp32_modules(base_model)
    install_inkling_precision(base_model)
    fp32_parameters = {id(parameter) for module in fp32_modules for parameter in module.parameters()}
    model = get_peft_model(
        base_model,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert all(parameter.dtype == torch.float32 for parameter in trainable)
    for parameter in model.parameters():
        if not parameter.requires_grad:
            assert parameter.dtype == (torch.float32 if id(parameter) in fp32_parameters else torch.bfloat16)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    tokens = torch.tensor([[10, 11, 12, 13, 14, 15]])
    model(tokens, labels=tokens, use_cache=False).loss.backward()
    assert all(parameter.grad.dtype == torch.float32 for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)
    optimizer.step()
    assert all(state["exp_avg"].dtype == torch.float32 for state in optimizer.state.values())
    assert all(state["exp_avg_sq"].dtype == torch.float32 for state in optimizer.state.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention regression requires CUDA")
def test_compact_attention_short_gqa_cuda_forward_and_backward():
    from torch.nn.attention.flex_attention import create_block_mask

    torch.manual_seed(17)
    # AUTO decoding selects a 256-row tile for Q34/GQA4, incompatible with the
    # native 128-row sparse block. Exercise the actual published head dimensions.
    query = torch.randn(1, 34, 32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(1, 34, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    bias = torch.randn(1, 34, 32, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = create_block_mask(lambda b, h, q, k: q >= k, 1, None, 34, 34, device="cuda")
    inputs = [tensor.transpose(1, 2) for tensor in (query, key, value)]
    module = torch.nn.Module().eval()
    with torch.no_grad():
        expected, _ = _compact_flex_attention(module, *inputs, mask, 1 / 128, bias.transpose(1, 2))
    module.train()
    output, _ = _compact_flex_attention(module, *inputs, mask, 1 / 128, bias.transpose(1, 2))
    assert output.shape == (1, 34, 32, 128)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected)
    output.float().square().mean().backward()
    for tensor in (query, key, value, bias):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 requires CUDA")
@pytest.mark.parametrize("compact_attention", [False, True])
def test_fsdp2_bf16_lora_preserves_native_fp32_convolutions(tmp_path, compact_attention):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from skyrl.backends.skyrl_train.distributed.fsdp_utils import apply_fsdp2
    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
    from skyrl.train.config import FSDPConfig

    torch.distributed.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        checkpoint = tmp_path / "base"
        # CUDA FlexAttention requires head_dim >= 16 (the published model uses 128).
        config = tiny_config(hidden_size=64)
        # Published Inkling crops padded-vocabulary logits, returning a view with
        # FSDP's pre-backward hook; in-place temperature scaling drops that hook.
        config.text_config.unpadded_vocab_size = 120
        AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16).save_pretrained(checkpoint)
        wrapper = HFModelWrapper(
            str(checkpoint),
            bf16=True,
            lora_rank=4,
            target_modules=["q_proj", "r_proj", "o_proj"],
            inkling_flex_attention=compact_attention,
        ).cuda()
        wrapper.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = wrapper.model
        observed = []
        router_observed = []
        for module in inkling_fp32_modules(model):
            if isinstance(module, modeling_inkling.InklingShortConvolution):
                module.register_forward_hook(
                    lambda module, inputs, output: observed.append((module.conv1d.weight.dtype, output.dtype))
                )
            else:
                module.register_forward_hook(
                    lambda module, inputs, output: router_observed.append(
                        (module.weight.dtype, module.e_score_correction_bias.dtype, inputs[0].dtype, output[1].dtype)
                    )
                )
        apply_fsdp2(
            model,
            {
                "mesh": init_device_mesh("cuda", (1,)),
                "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
                "reshard_after_forward": True,
            },
            FSDPConfig(),
        )
        tokens = torch.tensor([[0, 0, 12, 13, 14, 15]], device="cuda")
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            log_probs, output = wrapper(
                tokens,
                num_actions=3,
                attention_mask=tokens.ne(0),
                return_output=True,
                compute_entropy=True,
                entropy_requires_grad=False,
            )
        assert output.logits.shape[-1] == 120
        assert torch.isfinite(log_probs).all() and torch.isfinite(output["entropy"]).all()
        (-log_probs.mean()).backward()
        assert observed and set(observed) == {(torch.float32, torch.bfloat16)}
        assert router_observed and set(router_observed) == {(torch.float32,) * 4}
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert all(parameter.dtype == torch.float32 for parameter in trainable)
        assert all(parameter.grad.dtype == torch.float32 for parameter in trainable)
        assert all(torch.isfinite(parameter.grad.full_tensor()).all() for parameter in trainable)
    finally:
        torch.distributed.destroy_process_group()


def _fsdp2_inkling_mixed_backend_worker(rank, checkpoint, rendezvous):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    from skyrl.backends.skyrl_train.distributed import fsdp_strategy
    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
    from skyrl.train.config import FSDPConfig

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "cpu:gloo,cuda:nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(42)
        wrapper = HFModelWrapper(
            checkpoint,
            bf16=True,
            lora_rank=4,
            target_modules=["q_proj", "r_proj", "o_proj"],
            inkling_flex_attention=True,
            meta_init=rank != 0,
        )
        wrapper.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        wrapper.model.register_buffer(
            "test_nonpersistent",
            torch.tensor([1.25, -2.5], device="cpu" if rank == 0 else "meta"),
            persistent=False,
        )
        expected = [wrapper.model.state_dict() if rank == 0 else None]
        torch.distributed.broadcast_object_list(expected)
        strategy = fsdp_strategy.FSDPStrategy(fsdp_config=FSDPConfig())
        strategy.device_mesh = init_device_mesh("cuda", (2,))
        broadcast_devices = set()
        original_broadcast = torch.distributed.broadcast

        def observe_broadcast(tensor, *args, **kwargs):
            broadcast_devices.add(tensor.device.type)
            return original_broadcast(tensor, *args, **kwargs)

        with patch("torch.distributed.broadcast", side_effect=observe_broadcast):
            wrapper.model = strategy._fsdp_init_model(wrapper, is_wrapped=True)

        assert broadcast_devices == {"cuda"}
        parameters = list(wrapper.model.parameters())
        assert all(isinstance(p, DTensor) and p.to_local().is_cuda for p in parameters)
        assert sum(p.to_local().numel() for p in parameters) <= sum(p.numel() for p in parameters) * 0.51
        assert torch.equal(wrapper.model.test_nonpersistent, torch.tensor([1.25, -2.5]))
        restored = wrapper.model.state_dict()
        assert set(restored) == set(expected[0])
        for name, tensor in restored.items():
            actual = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
            torch.testing.assert_close(actual.cpu(), expected[0][name], rtol=0, atol=0)
        tokens = torch.tensor([[0, 0, 12, 13, 14, 15]], device="cuda")
        (-wrapper(tokens, num_actions=3, attention_mask=tokens.ne(0)).mean()).backward()
        trainable = [p for p in parameters if p.requires_grad]
        assert all(p.dtype == torch.float32 and p.grad.dtype == torch.float32 for p in trainable)
        assert all(torch.isfinite(p.grad.to_local()).all() for p in trainable)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="FSDP2 initialization regression requires two CUDA ranks")
def test_fsdp2_mixed_backend_initialization_restores_cuda_shards_and_buffers(tmp_path):
    checkpoint = tmp_path / "base"
    AutoModelForImageTextToText.from_config(tiny_config(hidden_size=64), dtype=torch.bfloat16).save_pretrained(
        checkpoint
    )
    torch.multiprocessing.spawn(
        _fsdp2_inkling_mixed_backend_worker,
        args=(str(checkpoint), str(tmp_path / "rendezvous")),
        nprocs=2,
        join=True,
    )


def test_attention_adapter_export_preserves_factors_and_uses_published_names():
    model = get_peft_model(
        AutoModelForImageTextToText.from_config(tiny_config()),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=4, target_modules=["q_proj", "k_proj", "v_proj", "r_proj", "o_proj"]
        ),
    )
    native = get_peft_model_state_dict(model)
    peft_config = {"target_modules": ["q_proj", "k_proj", "v_proj", "r_proj", "o_proj"], "r": 4}
    exported, exported_config = inkling_lora_for_inference(model.config, native, peft_config)
    assert len(exported) == len(native) == 20
    assert exported_config["target_modules"] == ["wq_du", "wk_dv", "wv_dv", "wr_du", "wo_ud"]
    assert peft_config["target_modules"][0] == "q_proj"
    for hf_name, original_name in zip(peft_config["target_modules"], exported_config["target_modules"]):
        for layer in range(2):
            for factor in ("A", "B"):
                hf_key = (
                    f"base_model.model.model.language_model.layers.{layer}.self_attn.{hf_name}.lora_{factor}.weight"
                )
                original_key = f"base_model.model.model.llm.layers.{layer}.attn.{original_name}.lora_{factor}.weight"
                assert exported[original_key] is native[hf_key]
    with pytest.raises(ValueError, match="explicit attention targets"):
        validate_inkling_lora_targets(model.config, "all-linear")


def test_native_adapter_checkpoint_restores_optimizer_and_next_update(tmp_path):
    torch.manual_seed(17)
    torch.set_num_threads(2)
    model = get_peft_model(
        AutoModelForImageTextToText.from_config(tiny_config()),
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, target_modules=["q_proj", "v_proj"]),
    )
    reloaded = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    restored_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=0.01)
    tokens = torch.tensor([[10, 11, 12, 13, 14, 15]])

    def step(current_model, current_optimizer):
        current_optimizer.zero_grad()
        loss = current_model(tokens, labels=tokens, use_cache=False).loss
        loss.backward()
        current_optimizer.step()

    step(model, optimizer)
    model.base_model.model.model.language_model.layers[1].mlp.gate.e_score_correction_bias.fill_(0.1)
    state = trainable_state_dict(model)
    frozen_names = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    assert state and not set(state) & frozen_names
    checkpoint = tmp_path / "adapter_training.pt"
    torch.save({"model": state, "optimizer": optimizer.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    load_trainable_state_dict(reloaded, saved["model"])
    restored_optimizer.load_state_dict(saved["optimizer"])
    step(model, optimizer)
    step(reloaded, restored_optimizer)
    for name, parameter in trainable_state_dict(model).items():
        torch.testing.assert_close(parameter, trainable_state_dict(reloaded)[name], rtol=0, atol=0)
    incomplete = dict(saved["model"])
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(RuntimeError, match="missing="):
        load_trainable_state_dict(reloaded, incomplete)
