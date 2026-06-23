"""CPU tests for the Megatron PiSSA helpers (no Megatron/GPU needed).

The pure decomposition math and offline materializer are tested in
tests/utils/test_pissa.py; here we cover the megatron-bridge-specific glue.
"""

import pytest

from skyrl.backends.skyrl_train.workers.megatron.pissa_init import bridge_a_init_method


@pytest.mark.parametrize("method", ["kaiming", "xavier", "normal", "zero"])
def test_bridge_a_init_method_passes_through_standard_methods(method):
    assert bridge_a_init_method(method) == method


@pytest.mark.parametrize("method", ["pissa", "pissa_niter_4"])
def test_bridge_a_init_method_placeholder_for_pissa(method):
    # PiSSA overwrites A post-transform; the bridge gets a valid placeholder.
    assert bridge_a_init_method(method) == "kaiming"
