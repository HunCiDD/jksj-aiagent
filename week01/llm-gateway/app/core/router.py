from __future__ import annotations

from app.core.config import ModelConfig
from app.core.errors import GatewayError


class ModelRouter:
    def __init__(self, models: dict[str, ModelConfig]) -> None:
        self._models = models

    def resolve(self, model_alias: str) -> ModelConfig:
        try:
            return self._models[model_alias]
        except KeyError as exc:
            raise GatewayError(
                "MODEL_NOT_FOUND",
                f"Unknown model: {model_alias}",
                status_code=404,
                details={"model": model_alias},
            ) from exc
