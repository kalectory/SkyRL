"""CPU tests for the Megatron PiSSA glue (no Megatron/GPU needed).

The decomposition math, PissaConfig parsing, and offline materializer are tested
in tests/utils/test_pissa.py; here we cover the megatron-side hook factory.
"""

from skyrl.backends.skyrl_train.workers.megatron.pissa_init import pissa_pre_wrap_hook
from skyrl.utils.pissa import PissaConfig


def test_pissa_pre_wrap_hook_returns_callable():
    # The hook is registered after the LoRA transform to overwrite A/B + base;
    # actually applying it needs Megatron, so just verify the factory wiring here.
    hook = pissa_pre_wrap_hook(PissaConfig(niter=4))
    assert callable(hook)
