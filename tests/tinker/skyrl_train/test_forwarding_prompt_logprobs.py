import asyncio
from types import SimpleNamespace

import httpx
import pytest

from skyrl.tinker.config import EngineConfig
from skyrl.tinker.extra import skyrl_train_inference_forwarding
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)


class _Response:
    def __init__(self):
        self.status_code = 200
        self.headers = {}
        self.text = ""

    def json(self):
        return {
            "choices": [
                {
                    "token_ids": [201],
                    "logprobs": {"token_logprobs": [None, -1.25, -0.75, -0.5]},
                    "prompt_logprobs": [
                        None,
                        {"102": {"logprob": -1.25}},
                        {"103": {"logprob": -0.75}},
                    ],
                    "finish_reason": "length",
                }
            ]
        }


class _HttpClient:
    def __init__(self):
        self.payload = None

    async def post(self, url, *, json, headers):
        self.payload = json
        return _Response()


def test_forwarding_timeout_matches_router_request_timeout():
    client = SkyRLTrainInferenceForwardingClient(
        EngineConfig(base_model="Qwen/Qwen3-4B-Instruct-2507"),
        db_engine=None,
    )

    assert client._http_client.timeout.read == 1800.0

    asyncio.run(client.aclose())


def test_forwarding_does_not_retry_read_timeout(monkeypatch):
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    resolved = []
    forwarded = []

    async def resolve_proxy_url(*, force_refresh=False):
        resolved.append(force_refresh)
        return "http://proxy"

    async def forward(proxy_url, sample_req, model_id, *, base_model):
        forwarded.append((proxy_url, sample_req, model_id, base_model))
        request = httpx.Request("POST", f"{proxy_url}/v1/completions")
        raise httpx.ReadTimeout("timed out", request=request)

    monkeypatch.setattr(client, "_resolve_proxy_url", resolve_proxy_url)
    monkeypatch.setattr(client, "_forward", forward)

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(
            client._forward_with_retry(
                sample_req="sample",
                model_id="adapter",
                base_model=None,
            )
        )

    assert resolved == [False]
    assert forwarded == [("http://proxy", "sample", "adapter", None)]


async def _forward(client):
    sample_req = SimpleNamespace(
        prompt=SimpleNamespace(to_types=lambda: object()),
        sampling_params=SimpleNamespace(
            seed=0,
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            stop=None,
        ),
        num_samples=1,
        prompt_logprobs=True,
        sampling_session_id=None,
        seq_id=None,
    )
    return await client._forward(
        "http://proxy",
        sample_req,
        model_id="",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )


def test_forwarding_returns_requested_prompt_logprobs(monkeypatch):
    monkeypatch.setattr(
        skyrl_train_inference_forwarding,
        "render_model_input",
        lambda _: [SimpleNamespace(prompt_ids=[101, 102, 103])],
    )
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client._http_client = _HttpClient()
    client._serves_lora_adapters = True
    client.engine_config = SimpleNamespace(base_model="Qwen/Qwen3-4B-Instruct-2507")

    output = asyncio.run(_forward(client))

    assert client._http_client.payload["echo"] is True
    assert output.prompt_logprobs == [None, -1.25, -0.75]
    assert output.sequences[0].tokens == [201]
    assert output.sequences[0].logprobs == [-0.5]
