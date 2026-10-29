from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Scalar = str | int | float | bool


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content_size(cls, value: str) -> str:
        if len(value) > 200_000:
            raise ValueError("message content is too long")
        return value


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_object"]


class PromptReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    version: int = Field(ge=1)
    variables: dict[str, Scalar] = Field(default_factory=dict)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(min_length=1)
    messages: list[Message] = Field(default_factory=list)
    stream: bool = False
    response_format: ResponseFormat | None = None
    prompt: PromptReference | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=128_000)

    @model_validator(mode="after")
    def validate_content_source(self) -> "ChatCompletionRequest":
        if not self.messages and self.prompt is None:
            raise ValueError("messages or prompt is required")
        return self


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0


class PromptCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    template: str = Field(min_length=1, max_length=200_000)


class PromptVersionCreateRequest(BaseModel):
    template: str = Field(min_length=1, max_length=200_000)


class PromptResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    version: int
    template: str
    created_at: datetime


class CallRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    request_id: str
    tenant_id: str
    requested_model: str
    resolved_model: str
    provider: str
    protocol: str
    prompt_name: str | None
    prompt_version: int | None
    status: str
    error_code: str | None
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    total_tokens: int
    ttft_ms: float | None
    total_latency_ms: float | None
    attempt_count: int
    retry_count: int
    created_at: datetime
