"""Native LoRA training checkpoint state without immutable base weights."""


def trainable_state_dict(model):
    frozen_names = {
        name for name, parameter in model.named_parameters(remove_duplicate=False) if not parameter.requires_grad
    }
    return {name: tensor for name, tensor in model.state_dict().items() if name not in frozen_names}


def load_trainable_state_dict(model, state_dict, strict=True):
    if strict:
        expected = set(trainable_state_dict(model))
        actual = set(state_dict)
        if actual != expected:
            raise RuntimeError(
                f"LoRA checkpoint trainable parameters differ: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
    model.load_state_dict(state_dict, strict=False)
