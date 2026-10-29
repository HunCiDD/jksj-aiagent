from __future__ import annotations

import json
import os
import time

import httpx


BASE_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("GATEWAY_API_KEY", "dev-gateway-key")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def require_ok(response: httpx.Response) -> dict:
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text}")
    return response.json()


def main() -> None:
    with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=120) as client:
        print("health:", require_ok(client.get("/health")))

        prompt_name = f"smoke-{int(time.time())}"
        require_ok(
            client.post(
                "/v1/prompts",
                json={"name": prompt_name, "template": "请总结：{{ text }}"},
            )
        )
        require_ok(
            client.post(
                f"/v1/prompts/{prompt_name}/versions",
                json={"template": "请用三个要点总结：{{ text }}"},
            )
        )

        for model in ("deepseek-v4-pro", "deepseek-v4-flash"):
            result = require_ok(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": model,
                        "prompt": {
                            "name": prompt_name,
                            "version": 1,
                            "variables": {"text": "LLM Gateway 统一了模型协议。"},
                        },
                        "response_format": {"type": "json_object"},
                    },
                )
            )
            content = result["choices"][0]["message"]["content"]
            json.loads(content)
            request_id = result["gateway"]["request_id"]
            record = require_ok(client.get(f"/v1/calls/{request_id}"))
            print(
                model,
                {
                    "content": content,
                    "input_tokens": record["input_tokens"],
                    "output_tokens": record["output_tokens"],
                    "total_latency_ms": record["total_latency_ms"],
                },
            )

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "请流式输出一句话"}],
                "stream": True,
            },
        ) as response:
            if response.status_code >= 400:
                raise RuntimeError(f"{response.status_code}: {response.read().decode()}")
            saw_delta = False
            request_id = None
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                saw_delta = saw_delta or bool(delta)
                request_id = chunk.get("gateway", {}).get("request_id", request_id)

        if not saw_delta or not request_id:
            raise RuntimeError("streaming smoke test did not receive text or request_id")
        record = require_ok(client.get(f"/v1/calls/{request_id}"))
        print(
            "stream",
            {
                "request_id": request_id,
                "ttft_ms": record["ttft_ms"],
                "total_latency_ms": record["total_latency_ms"],
                "tokens": record["total_tokens"],
            },
        )
        print("live smoke passed")


if __name__ == "__main__":
    main()
