from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


@dataclass(frozen=True, slots=True)
class ModelConfig:
    alias: str
    provider: str
    protocol: str
    upstream_model: str
    base_url: str
    api_key: str
    timeout_seconds: float = 60.0
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 10


@dataclass(frozen=True, slots=True)
class Settings:
    gateway_api_key: str
    database_url: str
    models: dict[str, ModelConfig]
    max_retries: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 5.0
    anthropic_version: str = "2023-06-01"
    default_max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "Settings":
        project_dir = Path(__file__).resolve().parents[2]
        default_db = project_dir / "data" / "gateway.db"
        default_db.parent.mkdir(parents=True, exist_ok=True)

        pro = ModelConfig(
            alias="deepseek-v4-pro",
            provider="deepseek",
            protocol="openai_responses",
            upstream_model=os.getenv(
                "DEEPSEEK_RESPONSES_MODEL",
                os.getenv("DEEPSEEK_PRO_RESPONSES_MODEL", "deepseek-v4-pro"),
            ),
            base_url=os.getenv(
                "DEEPSEEK_RESPONSES_BASE_URL",
                os.getenv("DEEPSEEK_PRO_RESPONSES_BASE_URL", ""),
            ),
            api_key=os.getenv(
                "DEEPSEEK_RESPONSES_API_KEY",
                os.getenv("DEEPSEEK_PRO_RESPONSES_API_KEY", ""),
            ),
            timeout_seconds=_env_float("DEEPSEEK_RESPONSES_TIMEOUT", 60.0),
            rate_limit_rpm=_env_int("DEEPSEEK_RESPONSES_RPM", 60),
            rate_limit_burst=_env_int("DEEPSEEK_RESPONSES_BURST", 10),
        )
        flash = ModelConfig(
            alias="deepseek-v4-flash",
            provider="deepseek",
            protocol="anthropic_messages",
            upstream_model=os.getenv(
                "DEEPSEEK_MESSAGES_MODEL",
                os.getenv("DEEPSEEK_FLASH_MESSAGES_MODEL", "deepseek-v4-flash"),
            ),
            base_url=os.getenv(
                "DEEPSEEK_MESSAGES_BASE_URL",
                os.getenv("DEEPSEEK_FLASH_MESSAGES_BASE_URL", ""),
            ),
            api_key=os.getenv(
                "DEEPSEEK_MESSAGES_API_KEY",
                os.getenv("DEEPSEEK_FLASH_MESSAGES_API_KEY", ""),
            ),
            timeout_seconds=_env_float("DEEPSEEK_MESSAGES_TIMEOUT", 60.0),
            rate_limit_rpm=_env_int("DEEPSEEK_MESSAGES_RPM", 60),
            rate_limit_burst=_env_int("DEEPSEEK_MESSAGES_BURST", 10),
        )

        return cls(
            gateway_api_key=os.getenv("GATEWAY_API_KEY", "dev-gateway-key"),
            database_url=os.getenv(
                "GATEWAY_DB_URL",
                f"sqlite+aiosqlite:///{default_db.as_posix()}",
            ),
            models={pro.alias: pro, flash.alias: flash},
            max_retries=_env_int("GATEWAY_MAX_RETRIES", 3),
            retry_base_delay=_env_float("GATEWAY_RETRY_BASE_DELAY", 0.5),
            retry_max_delay=_env_float("GATEWAY_RETRY_MAX_DELAY", 5.0),
            anthropic_version=os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
            default_max_tokens=_env_int("GATEWAY_DEFAULT_MAX_TOKENS", 1024),
        )
