from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}
        self.retry_after = retry_after

    def to_payload(self, request_id: str) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "request_id": request_id,
                "retryable": self.retryable,
                "details": self.details,
            }
        }


def invalid_request(message: str, details: dict[str, Any] | None = None) -> GatewayError:
    return GatewayError(
        "INVALID_REQUEST",
        message,
        status_code=400,
        details=details,
    )
