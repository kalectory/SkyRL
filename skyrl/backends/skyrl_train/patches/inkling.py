"""Inkling checkpoint compatibility for native Transformers and vLLM."""

from types import MethodType

import torch
from transformers import PretrainedConfig

ATTENTION_PROJECTIONS = {
    "q_proj": "wq_du",
    "k_proj": "wk_dv",
    "v_proj": "wv_dv",
    "r_proj": "wr_du",
    "o_proj": "wo_ud",
}


def inkling_fp32_modules(model):
    if model.config.model_type != "inkling_mm_model":
        return []
    from transformers.models.inkling.modeling_inkling import (
        InklingShortConvolution,
        InklingTopkRouter,
    )

    return [module for module in model.modules() if isinstance(module, (InklingShortConvolution, InklingTopkRouter))]


def prepare_inkling_precision(config):
    if config.model_type != "inkling_mm_model":
        return
    from transformers.models.inkling.modeling_inkling import InklingPreTrainedModel

    # Do this before from_pretrained: restoring FP32 after loading would retain
    # rounded BF16 checkpoint bias/scale values. The gate matrix is stored BF16.
    InklingPreTrainedModel._keep_in_fp32_modules_strict = {
        *InklingPreTrainedModel._keep_in_fp32_modules_strict,
        r"mlp\.gate\.",
    }


def _router_forward(self, hidden_states):
    from transformers.models.inkling.modeling_inkling import InklingTopkRouter

    # BF16-valued inputs/weights, FP32 logits and selection, as in vLLM.
    # torch.mm(..., out_dtype=float32) currently has no autograd derivative.
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        return InklingTopkRouter.forward(self, hidden_states.float())


def _shared_experts_forward(self, hidden_states, gammas):
    # Native HF forward with vLLM's post-gamma, pre-down-projection cast.
    input_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(1, -1, input_shape[-1]).expand(self.n_shared_experts, -1, -1)
    gammas = gammas.reshape(-1, self.n_shared_experts, 1).transpose(0, 1)
    gate = torch.bmm(hidden_states, self.gate_proj.transpose(1, 2))
    up = torch.bmm(hidden_states, self.up_proj.transpose(1, 2))
    activated = (self.act_fn(gate) * up * gammas).to(hidden_states.dtype)
    down = torch.bmm(activated, self.down_proj.transpose(1, 2))
    return down.float().sum(dim=0).to(hidden_states.dtype).view(input_shape)


def install_inkling_precision(model):
    if model.config.model_type != "inkling_mm_model":
        return
    from transformers.models.inkling.modeling_inkling import (
        InklingSharedExperts,
        InklingTopkRouter,
    )

    for module in inkling_fp32_modules(model):
        module.float()
        if isinstance(module, InklingTopkRouter):
            module.forward = MethodType(_router_forward, module)
    for module in model.modules():
        if isinstance(module, InklingSharedExperts):
            module.forward = MethodType(_shared_experts_forward, module)


def correct_inkling_expert_size(config, model_path, config_kwargs):
    if config.model_type != "inkling_mm_model":
        return
    published, _ = PretrainedConfig.get_config_dict(model_path, **config_kwargs)
    text = published["text_config"]
    overrides = config_kwargs.get("text_config", {})
    if (
        "dense_intermediate_size" in text
        and "moe_intermediate_size" not in text
        and "moe_intermediate_size" not in overrides
    ):
        # Transformers 5.16.1 overwrites intermediate_size with the dense width.
        config.text_config.moe_intermediate_size = text["intermediate_size"]


def validate_inkling_lora_targets(config, target_modules):
    if config.model_type == "inkling_mm_model" and (
        isinstance(target_modules, str) or not set(target_modules).issubset(ATTENTION_PROJECTIONS)
    ):
        raise ValueError("Inkling LoRA requires explicit attention targets: q_proj, k_proj, v_proj, r_proj, o_proj")


def inkling_lora_for_inference(model_config, lora_params, peft_config):
    if model_config.model_type != "inkling_mm_model":
        return lora_params, peft_config
    validate_inkling_lora_targets(model_config, peft_config["target_modules"])
    renamed = {}
    for name, tensor in lora_params.items():
        name = name.replace("model.language_model.layers.", "model.llm.layers.")
        for hf_name, checkpoint_name in ATTENTION_PROJECTIONS.items():
            name = name.replace(f".self_attn.{hf_name}.", f".attn.{checkpoint_name}.")
        renamed[name] = tensor
    peft_config = dict(peft_config)
    peft_config["target_modules"] = [ATTENTION_PROJECTIONS[name] for name in peft_config["target_modules"]]
    return renamed, peft_config
