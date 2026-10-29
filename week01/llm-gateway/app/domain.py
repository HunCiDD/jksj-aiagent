from __future__ import annotations

from dataclasses import dataclass

from app.schemas import Message, ResponseFormat, TokenUsage


@dataclass(slots=True)
class UnifiedRequest:
    model: str
    messages: list[Message]
    stream: bool
    response_format: ResponseFormat | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(slots=True)
class UnifiedResponse:
    content: str
    finish_reason: str | None
    usage: TokenUsage


@dataclass(slots=True)
class UnifiedStreamEvent:
    type: str
    content: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    ttft_ms: float | None = None
    total_latency_ms: float | None = None
    retry_count: int = 0
    attempt_count: int = 0
    error: dict | None = None
