import asyncio
import math
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from wald_agent.errors import CircuitOpen


def retry_after_seconds(header: str | None) -> float | None:
    if header is None:
        return None
    try:
        value = float(header)
    except ValueError:
        try:
            date = parsedate_to_datetime(header)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            value = (date - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, value) if math.isfinite(value) else None


class CircuitBreaker:
    """Per-client circuit with one half-open probe and generation-safe completions."""

    def __init__(self, threshold: int, cooldown: float):
        self.threshold, self.cooldown = threshold, cooldown
        self.failures = 0
        self.open_until = 0.0
        self.generation = 0
        self.probing = False
        self.lock = asyncio.Lock()

    async def acquire(self) -> tuple[int, bool]:
        async with self.lock:
            remaining = self.open_until - time.monotonic()
            if self.probing or remaining > 0:
                raise CircuitOpen(
                    "Upstream circuit is temporarily open.", retry_after=max(1, remaining)
                )
            probe = self.open_until != 0
            if probe:
                self.probing = True
            return self.generation, probe

    async def finish(self, ticket: tuple[int, bool], success: bool | None) -> None:
        async with self.lock:
            generation, probe = ticket
            if generation != self.generation:
                return
            if probe:
                self.probing = False
            if success is True:
                self.failures = 0
                self.open_until = 0
            elif success is False:
                self.failures += 1
                if probe or self.failures >= self.threshold:
                    self.open_until = time.monotonic() + self.cooldown
                    self.generation += 1
            elif probe:
                self.open_until = time.monotonic() + self.cooldown
                self.generation += 1
