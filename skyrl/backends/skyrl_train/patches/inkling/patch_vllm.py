"""Use the training router's FP32 projection for vLLM Inkling gates."""

import torch
import torch.nn.functional as F


def _compute_logits(self, x: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        return F.linear(x.float(), self.weight.float())


def apply_inkling_vllm_patch() -> None:
    """Install the FP32 router projection before vLLM constructs the model."""
    try:
        from vllm.models.inkling.nvidia.moe import InklingGate
    except ModuleNotFoundError as exc:
        # vLLM is optional, and older versions do not include Inkling.
        if exc.name in {
            "vllm",
            "vllm.models",
            "vllm.models.inkling",
            "vllm.models.inkling.nvidia",
            "vllm.models.inkling.nvidia.moe",
        }:
            return
        raise

    InklingGate.compute_logits = _compute_logits
