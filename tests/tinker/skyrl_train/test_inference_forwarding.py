from unittest.mock import AsyncMock

import httpx
import pytest

from skyrl.tinker.config import EngineConfig
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)


@pytest.mark.asyncio
async def test_forwarding_client_allows_long_generations():
    client = SkyRLTrainInferenceForwardingClient(
        EngineConfig(base_model="test-model"),
        db_engine=None,
    )

    try:
        assert client._http_client.timeout.read == 1800.0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_read_timeout_does_not_replay_generation():
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client._cached_proxy_url = "http://inference"
    client._resolve_proxy_url = AsyncMock(return_value="http://inference")
    client._forward = AsyncMock(side_effect=httpx.ReadTimeout("generation timed out"))

    with pytest.raises(httpx.ReadTimeout):
        await client._forward_with_retry(object(), "adapter", base_model=None)

    client._resolve_proxy_url.assert_awaited_once_with()
    assert client._forward.await_count == 1


@pytest.mark.asyncio
async def test_connect_error_refreshes_endpoint_and_retries_once():
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client._cached_proxy_url = "http://old-inference"
    client._resolve_proxy_url = AsyncMock(side_effect=["http://old-inference", "http://new-inference"])
    client._forward = AsyncMock(side_effect=[httpx.ConnectError("connection failed"), "result"])

    result = await client._forward_with_retry(object(), "adapter", base_model=None)

    assert result == "result"
    assert client._resolve_proxy_url.await_args_list[1].kwargs == {"force_refresh": True}
    assert client._forward.await_count == 2
