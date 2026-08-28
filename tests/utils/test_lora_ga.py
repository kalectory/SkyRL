"""CPU tests for task-gradient LoRA initialization."""

import json

import pytest
import torch

from skyrl.utils.lora_ga import (
    _group_lora_layers,
    batch_gspo_datums,
    decompose_loraga,
    load_gspo_datums,
    materialize_loraga,
)


class _Layer:
    pass


def test_group_lora_layers_matches_megatron_fusions():
    layers = [
        ("model.layers.0.self_attn.k_proj", _Layer()),
        ("model.layers.0.self_attn.q_proj", _Layer()),
        ("model.layers.0.self_attn.v_proj", _Layer()),
        ("model.layers.0.self_attn.o_proj", _Layer()),
        ("model.layers.0.mlp.gate_proj", _Layer()),
        ("model.layers.0.mlp.up_proj", _Layer()),
        ("model.layers.0.mlp.down_proj", _Layer()),
    ]

    groups = _group_lora_layers(layers)

    assert [(group.name, [name for name, _ in group.members]) for group in groups] == [
        (
            "model.layers.0.self_attn.linear_qkv",
            [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
            ],
        ),
        ("model.layers.0.self_attn.o_proj", ["model.layers.0.self_attn.o_proj"]),
        (
            "model.layers.0.mlp.linear_fc1",
            ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"],
        ),
        ("model.layers.0.mlp.down_proj", ["model.layers.0.mlp.down_proj"]),
    ]


def test_decompose_loraga_covers_first_two_rank_gradient_modes():
    torch.manual_seed(0)
    weight = torch.randn(8, 8)
    singular_values = torch.arange(8, 0, -1, dtype=torch.float32)
    gradient = torch.diag(singular_values)

    result = decompose_loraga(weight, gradient, rank=2, scale=1.0)

    assert torch.allclose(result.linear_in[:, 2:], torch.zeros(2, 6), atol=1e-5)
    assert torch.allclose(result.linear_out[:2], torch.zeros(2, 2), atol=1e-5)
    assert torch.allclose(result.linear_out[4:], torch.zeros(4, 2), atol=1e-5)
    assert result.captured_energy == pytest.approx(
        singular_values[:4].square().sum().item() / singular_values.square().sum().item(),
        rel=1e-5,
    )
    reconstructed = result.residual + result.linear_out @ result.linear_in
    assert torch.allclose(reconstructed, weight, atol=1e-5)


def test_decompose_loraga_requires_two_rank_modes():
    weight = torch.randn(4, 8)
    with pytest.raises(ValueError, match=r"2 \* rank"):
        decompose_loraga(weight, torch.randn_like(weight), rank=3, scale=1.0)


def test_load_gspo_datums_shifts_tokens_and_uses_trainable_steps(tmp_path):
    payload = {
        "trajectories": [
            {
                "steps": [
                    {
                        "tokens": [1, 2, 3, 4],
                        "token_masks": [0, 0, 1, 1],
                        "trainable_status": "trainable",
                        "advantages": [],
                    },
                    {
                        "tokens": [4, 5],
                        "token_masks": [0, 1],
                        "trainable_status": "not_trainable",
                        "advantages": [],
                    },
                ]
            }
        ],
        "advantages": [1.5],
        "temperature": 1.0,
    }
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(payload))

    datums, digest, summary = load_gspo_datums(str(path))

    assert len(digest) == 64
    assert summary == {
        "n_trajectories": 1,
        "n_datums": 1,
        "n_trainable_tokens": 2,
        "temperature": 1.0,
    }
    assert datums[0].input_ids == [1, 2, 3]
    assert datums[0].target_ids == [2, 3, 4]
    assert datums[0].token_mask == [0, 1, 1]
    assert datums[0].advantage == 1.5


def test_batch_gspo_datums_respects_padded_token_budget(tmp_path):
    payload = {
        "trajectories": [
            {
                "steps": [
                    {
                        "tokens": list(range(length + 1)),
                        "token_masks": [0] + [1] * length,
                        "trainable_status": "trainable",
                        "advantages": [],
                    }
                ]
            }
            for length in (2, 3, 5)
        ],
        "advantages": [1.0, 1.0, 1.0],
    }
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(payload))
    datums, _, _ = load_gspo_datums(str(path))

    batches = list(batch_gspo_datums(datums, max_tokens=6))

    assert [[len(datum.input_ids) for datum in batch] for batch in batches] == [
        [2, 3],
        [5],
    ]


def test_materialize_loraga_preserves_tiny_model_function(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    safetensors = pytest.importorskip("safetensors.torch")

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def save_pretrained(self, output_dir):
            (output_dir / "tokenizer_config.json").write_text("{}")

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda _model_path: _Tokenizer(),
    )
    config = transformers.LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=32,
    )
    torch.manual_seed(0)
    source = tmp_path / "source"
    transformers.LlamaForCausalLM(config).save_pretrained(source)
    batch = {
        "trajectories": [
            {
                "steps": [
                    {
                        "tokens": [1, 2, 3, 4, 5],
                        "token_masks": [0, 0, 1, 1, 1],
                        "trainable_status": "trainable",
                        "advantages": [],
                    }
                ]
            },
            {
                "steps": [
                    {
                        "tokens": [1, 3, 2, 5, 4],
                        "token_masks": [0, 0, 1, 1, 1],
                        "trainable_status": "trainable",
                        "advantages": [],
                    }
                ]
            },
        ],
        "advantages": [1.0, -1.0],
    }
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps(batch))

    manifest = materialize_loraga(
        str(source),
        str(batch_path),
        str(tmp_path / "artifact"),
        rank=2,
        alpha=2,
        code_sha="test-sha",
        device="cpu",
        max_batch_tokens=16,
    )

    assert manifest["adapter_parameterization"] == "megatron-fused-qkv-gate-up"
    assert manifest["gradient"]["n_target_groups"] == 4
    assert manifest["gradient"]["n_target_modules"] == 7
    assert manifest["identity"]["reload_logit_max_abs"] < 1e-3
    adapter_path = tmp_path / "artifact" / "adapter" / "adapter_model.safetensors"
    assert adapter_path.exists()
    assert (tmp_path / "artifact" / "residual_base" / "model.safetensors").exists()

    adapter = safetensors.load_file(adapter_path)

    def get_factor(projection):
        matches = [
            value for key, value in adapter.items() if f".{projection}." in key and key.endswith("lora_A.weight")
        ]
        assert len(matches) == 1
        return matches[0]

    assert torch.equal(get_factor("q_proj"), get_factor("k_proj"))
    assert torch.equal(get_factor("k_proj"), get_factor("v_proj"))
    assert torch.equal(get_factor("gate_proj"), get_factor("up_proj"))
