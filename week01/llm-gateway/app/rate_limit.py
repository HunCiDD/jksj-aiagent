from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    retry_after: int | None = None


class TokenBucket:
    def __init__(
        self,
        requests_per_minute: int,
        burst: int,
        *,
        clock=time.monotonic,
    ) -> None:
        if requests_per_minute <= 0 or burst <= 0:
            raise ValueError("rate limit values must be positive")
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.rate_per_second = requests_per_minute / 60.0
        self.clock = clock
        self.updated_at = clock()
        self.lock = asyncio.Lock()

    async def acquire(self) -> RateLimitDecision:
        async with self.lock:
            now = self.clock()
            elapsed = max(0.0, now - self.updated_at)
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_second)
            self.updated_at = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return RateLimitDecision(allowed=True)
            missing = 1.0 - self.tokens
            retry_after = max(1, math.ceil(missing / self.rate_per_second))
            return RateLimitDecision(allowed=False, retry_after=retry_after)


class ModelRateLimiter:
    def __init__(self, buckets: dict[str, TokenBucket]) -> None:
        self.buckets = buckets

    async def acquire(self, model: str) -> RateLimitDecision:
        bucket = self.buckets.get(model)
        if bucket is None:
            return RateLimitDecision(allowed=True)
        return await bucket.acquire()
