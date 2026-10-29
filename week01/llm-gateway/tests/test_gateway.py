from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from conftest import MockUpstream, make_settings


AUTH = {"Authorization": "Bearer test-gateway-key"}


def make_client(
    tmp_path: Path,
    upstream: MockUpstream,
    *,
    sleeps: list[float] | None = None,
    **settings_overrides,
):
    async def fake_sleep(delay: float) -> None:
        if sleeps is not None:
            sleeps.append(delay)

    settings = make_settings(tmp_path, **settings_overrides)
    app = create_app(
        settings,
        upstream_transport=httpx.MockTransport(upstream.handler),
        sleep=fake_sleep,
    )
    return TestClient(app)


def test_auth_health_and_dynamic_protocols(tmp_path: Path) -> None:
    upstream = MockUpstream()
    with make_client(tmp_path, upstream) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json()["status"] == "ready"
        unauthorized = client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert unauthorized.status_code == 401
        assert unauthorized.json()["error"]["code"] == "INVALID_GATEWAY_KEY"

        for model in ("deepseek-v4-pro", "deepseek-v4-flash"):
            response = client.post(
                "/v1/chat/completions",
                headers=AUTH,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "return JSON"}],
                    "response_format": {"type": "json_object"},
                },
            )
            assert response.status_code == 200
            payload = response.json()
            assert json.loads(payload["choices"][0]["message"]["content"]) == {"ok": True}

        assert upstream.requests[0].url.path.endswith("/responses")
        assert upstream.requests[1].url.path.endswith("/messages")

        records = client.get("/v1/calls", headers=AUTH).json()
        by_protocol = {record["protocol"]: record for record in records}
        assert set(by_protocol) == {
            "openai_responses",
            "anthropic_messages",
        }
        for record in records:
            assert record["status"] == "success"
            assert record["input_tokens"] > 0
            assert record["output_tokens"] > 0
            assert record["total_tokens"] > 0
            assert record["total_latency_ms"] >= 0
        assert by_protocol["openai_responses"]["cached_input_tokens"] == 3
        assert by_protocol["openai_responses"]["reasoning_tokens"] == 2
        assert by_protocol["anthropic_messages"]["cached_input_tokens"] == 4


@pytest.mark.parametrize(
    ("model", "expected_input", "expected_cached"),
    [
        ("deepseek-v4-pro", 11, 3),
        ("deepseek-v4-flash", 13, 4),
    ],
)
def test_streaming_metrics_and_ttft(
    tmp_path: Path,
    model: str,
    expected_input: int,
    expected_cached: int,
) -> None:
    upstream = MockUpstream()
    with make_client(tmp_path, upstream) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "stream this"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        chunks = [json.loads(line[6:]) for line in lines[:-1]]
        text = "".join(
            chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            for chunk in chunks
        )
        assert text
        final = chunks[-1]
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final["usage"]["input_tokens"] == expected_input
        assert final["usage"]["cached_input_tokens"] == expected_cached
        request_id = final["gateway"]["request_id"]
        record = client.get(f"/v1/calls/{request_id}", headers=AUTH).json()
        assert record["ttft_ms"] is not None
        assert record["ttft_ms"] >= 0
        assert record["input_tokens"] == expected_input
        assert record["output_tokens"] > 0


def test_prompt_versions_render_and_remain_immutable(tmp_path: Path) -> None:
    upstream = MockUpstream()
    with make_client(tmp_path, upstream) as client:
        first = client.post(
            "/v1/prompts",
            headers=AUTH,
            json={"name": "summarize", "template": "总结：{{ text }}"},
        )
        second = client.post(
            "/v1/prompts/summarize/versions",
            headers=AUTH,
            json={"template": "三点总结：{{ text }}"},
        )
        assert first.status_code == 201
        assert second.status_code == 201
        assert first.json()["version"] == 1
        assert second.json()["version"] == 2

        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "执行"}],
                "prompt": {
                    "name": "summarize",
                    "version": 1,
                    "variables": {"text": "一段文本"},
                },
            },
        )
        assert response.status_code == 200
        sent = json.loads(upstream.requests[-1].content)
        assert "总结：一段文本" in sent["instructions"]
        assert response.json()["choices"][0]["message"]["content"] == sent["instructions"]

        old_version = client.get("/v1/prompts/summarize/versions/1", headers=AUTH)
        assert old_version.json()["template"] == "总结：{{ text }}"

        missing = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "执行"}],
                "prompt": {"name": "summarize", "version": 1, "variables": {}},
            },
        )
        assert missing.status_code == 422
        assert missing.json()["error"]["code"] == "PROMPT_VARIABLE_MISSING"


def test_invalid_structured_output_is_rejected(tmp_path: Path) -> None:
    upstream = MockUpstream()
    upstream.invalid_structured_output = True
    with make_client(tmp_path, upstream) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "JSON"}],
                "response_format": {"type": "json_object"},
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_STRUCTURED_OUTPUT"
        record = client.get(
            f"/v1/calls/{response.json()['error']['request_id']}",
            headers=AUTH,
        ).json()
        assert record["error_code"] == "INVALID_STRUCTURED_OUTPUT"


def test_exponential_retry_then_success(tmp_path: Path) -> None:
    upstream = MockUpstream()
    upstream.pro_failures_remaining = 2
    sleeps: list[float] = []
    with make_client(tmp_path, upstream, sleeps=sleeps, pro_burst=10) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "retry"}],
            },
        )
        assert response.status_code == 200
        assert response.json()["gateway"]["attempt_count"] == 3
        assert response.json()["gateway"]["retry_count"] == 2
        assert sleeps == [0.001, 0.002]
        record = client.get(
            f"/v1/calls/{response.json()['gateway']['request_id']}",
            headers=AUTH,
        ).json()
        assert record["attempt_count"] == 3
        assert record["retry_count"] == 2


def test_retry_stops_after_three_retries(tmp_path: Path) -> None:
    upstream = MockUpstream()
    upstream.pro_failures_remaining = 10
    sleeps: list[float] = []
    with make_client(tmp_path, upstream, sleeps=sleeps, pro_burst=1) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "retry until failure"}],
            },
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "UPSTREAM_UNAVAILABLE"
        assert len(upstream.requests) == 4
        assert sleeps == [0.001, 0.002, 0.004]
        record = client.get(
            f"/v1/calls/{response.json()['error']['request_id']}",
            headers=AUTH,
        ).json()
        assert record["attempt_count"] == 4
        assert record["retry_count"] == 3


def test_rate_limits_are_independent_per_model(tmp_path: Path) -> None:
    upstream = MockUpstream()
    with make_client(
        tmp_path,
        upstream,
        pro_rpm=1,
        pro_burst=1,
        flash_rpm=100,
        flash_burst=10,
    ) as client:
        payload = {"messages": [{"role": "user", "content": "rate"}]}
        first = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={**payload, "model": "deepseek-v4-pro"},
        )
        limited = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={**payload, "model": "deepseek-v4-pro"},
        )
        other_model = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={**payload, "model": "deepseek-v4-flash"},
        )

        assert first.status_code == 200
        assert limited.status_code == 429
        assert limited.json()["error"]["code"] == "MODEL_RATE_LIMITED"
        assert int(limited.headers["retry-after"]) >= 1
        assert other_model.status_code == 200

        records = client.get("/v1/calls", headers=AUTH).json()
        limited_record = next(
            record for record in records if record["error_code"] == "MODEL_RATE_LIMITED"
        )
        assert limited_record["attempt_count"] == 0
