"""CPU checks for the installed native vLLM Inkling router."""

import pytest
import torch

moe = pytest.importorskip("vllm.models.inkling.nvidia.moe")
pytestmark = pytest.mark.vllm

from skyrl.backends.skyrl_train.patches.inkling.patch_vllm import (  # noqa: E402
    apply_inkling_vllm_patch,
)


@pytest.fixture(scope="session", autouse=True)
def ray_init():
    # Native gate construction needs no Ray actors or GPU.
    yield


def test_router_projection_is_fp32_under_autocast_without_changing_parameters(monkeypatch):
    torch.manual_seed(17)
    monkeypatch.setattr(moe.InklingGate, "compute_logits", moe.InklingGate.compute_logits)
    apply_inkling_vllm_patch()
    installed = moe.InklingGate.compute_logits
    apply_inkling_vllm_patch()
    assert moe.InklingGate.compute_logits is installed

    def forbidden_helper(*_args):
        raise AssertionError("The BF16 router helper must not run")

    monkeypatch.setattr(moe, "_linear_with_fp32_out", forbidden_helper)
    gate = moe.InklingGate(64, 4, 2, 2, 8.0, use_global_scale=True, use_gate_bias=True)
    gate.weight.data = gate.weight.data.bfloat16()
    gate._load_weight(gate.weight, torch.randn(6, 64, dtype=torch.bfloat16) / 8)
    gate.bias.data.copy_(torch.linspace(-0.01, 0.01, 4))
    gate.global_scale.data.fill_(1.0001234)
    before = {name: value.clone() for name, value in gate.state_dict().items()}

    with torch.inference_mode():
        for leading in ((1,), (64,), (65,), (433,), (2, 3)):
            hidden = torch.randn(*leading, 64, dtype=torch.bfloat16)
            expected = torch.nn.functional.linear(hidden.float(), gate.weight.float())
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                actual = gate.compute_logits(hidden)
            assert actual.dtype == torch.float32
            assert actual.shape == (*leading, 8)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.count_nonzero(actual[..., 6:]) == 0

    for name, value in gate.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert gate.weight.dtype == torch.bfloat16
    assert gate.bias.dtype == gate.global_scale.dtype == torch.float32
