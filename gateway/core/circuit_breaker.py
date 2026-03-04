"""Circuit breaker implementation for provider resilience."""

import time
import asyncio
from typing import Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failure threshold reached, requests blocked
    HALF_OPEN = "half_open"  # Testing if provider recovered


class CircuitBreaker:
    """
    Circuit breaker for provider resilience.

    Opens after 3 failures, retries after 30 seconds.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def can_execute(self) -> bool:
        """Check if request can be executed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if (
                    self._last_failure_time
                    and (time.time() - self._last_failure_time) >= self.recovery_timeout
                ):
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(
                        f"Circuit breaker for {self.name} entering HALF_OPEN state"
                    )
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return True

    async def record_success(self):
        """Record a successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                logger.info(f"Circuit breaker for {self.name} CLOSED (recovered)")
            else:
                self._failure_count = 0

    async def record_failure(self):
        """Record a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker for {self.name} OPENED again (half-open test failed)"
                )
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker for {self.name} OPENED after {self._failure_count} failures"
                )

    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "last_failure": self._last_failure_time,
            "is_closed": self.is_closed,
        }


class CircuitBreakerRegistry:
    """Registry for managing circuit breakers for multiple providers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self, name: str, failure_threshold: int = 3, recovery_timeout: float = 30.0
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for a provider."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_status() for name, breaker in self._breakers.items()}

    async def reset(self, name: str):
        """Reset a circuit breaker."""
        breaker = self._breakers.get(name)
        if breaker:
            async with breaker._lock:
                breaker._state = CircuitState.CLOSED
                breaker._failure_count = 0
                breaker._last_failure_time = None
                breaker._half_open_calls = 0
            logger.info(f"Circuit breaker for {name} manually reset")
