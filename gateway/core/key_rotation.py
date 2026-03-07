"""Key rotation and quota management for providers."""

import time
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class KeyStatus(Enum):
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    SESSION_EXPIRED = "session_expired"
    DISABLED = "disabled"


@dataclass
class ProviderKey:
    """Represents a provider API key with its metadata."""

    key: str
    index: int = 0  # La position de la clé (1, 2, 3...)
    models: List[str] = field(default_factory=list)
    status: KeyStatus = KeyStatus.AVAILABLE
    last_used: Optional[float] = None
    session_start_time: Optional[float] = None
    error_count: int = 0
    cooldown_until: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class KeyRotationStrategy(Enum):
    SEQUENTIAL = "sequential"
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    FALLBACK = "fallback"


class KeyManager:
    """
    Manages API key rotation and quota tracking.

    Features:
    - Sequential key rotation
    - Rate limit detection
    - Quota exhaustion with 5-minute cooldown
    - Error tracking
    """

    DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes
    DISABLED_COOLDOWN_SECONDS = 3600  # 1 hour
    SESSION_LIMIT_SECONDS = 7200  # 2 hours
    MAX_ERROR_COUNT = 5

    def __init__(
        self,
        provider_name: str,
        keys: List[Dict[str, Any]],
        strategy: KeyRotationStrategy = KeyRotationStrategy.SEQUENTIAL,
        cooldown_seconds: float = 300,
    ):
        self.provider_name = provider_name
        self.strategy = strategy
        self.cooldown_seconds = cooldown_seconds

        self._keys: List[ProviderKey] = []
        self._current_index = 0
        self._lock = asyncio.Lock()
        self._total_keys = len(keys)

        # Initialize keys
        for index, key_data in enumerate(keys):
            if isinstance(key_data, dict):
                self._keys.append(
                    ProviderKey(
                        key=key_data.get("key", ""),
                        index=index + 1,
                        models=[m.strip() for m in key_data.get("models", [])],
                        metadata=key_data.get("metadata", {}),
                    )
                )
            else:
                self._keys.append(ProviderKey(key=key_data, index=index + 1))

        logger.info(f"Initialized {self.provider_name} with {len(self._keys)} keys")

    def _is_key_available(self, key: ProviderKey) -> bool:
        """Check if a key is available for use."""
        now = time.time()

        # Check if cooldown has expired
        if key.cooldown_until and now >= key.cooldown_until:
            logger.info(
                f"{self.provider_name}: Key cooldown expired, resetting status from {key.status.value}"
            )
            key.cooldown_until = None
            key.status = KeyStatus.AVAILABLE
            key.error_count = 0
            key.session_start_time = None  # Reset session on recovery
            return True

        if key.status == KeyStatus.DISABLED:
            return False

        if key.cooldown_until and now < key.cooldown_until:
            return False

        # Check for 2-hour session limit
        if (
            key.session_start_time
            and (now - key.session_start_time) > self.SESSION_LIMIT_SECONDS
        ):
            logger.warning(
                f"{self.provider_name}: Key usage exceeded 2h limit, rotating..."
            )
            key.status = KeyStatus.SESSION_EXPIRED
            key.cooldown_until = (
                now + self.DEFAULT_COOLDOWN_SECONDS
            )  # 5 min cooldown before reuse
            return False

        return True

    def _get_available_keys(self) -> List[ProviderKey]:
        """Get list of currently available keys."""
        return [k for k in self._keys if self._is_key_available(k)]

    async def get_next_key(
        self, model: Optional[str] = None, alias: Optional[str] = None
    ) -> Optional[ProviderKey]:
        """
        Get the next available key based on rotation strategy.

        Args:
            model: Optional technical model name to filter keys by
            alias: Optional original model alias to filter keys by

        Returns:
            Next available key or None if all keys are exhausted
        """
        async with self._lock:
            available = self._get_available_keys()

            if not available:
                logger.warning(f"{self.provider_name}: All keys exhausted")
                return None

            # Filter by model or alias if specified
            if model or alias:
                logger.debug(
                    f"Provider {self.provider_name} filtering keys for model: '{model}', alias: '{alias}'"
                )
                available_with_model = [
                    k
                    for k in available
                    if not k.models
                    or (model and model in k.models)
                    or (alias and alias in k.models)
                ]
                if not available_with_model:
                    logger.warning(
                        f"{self.provider_name}: No keys match model='{model}' or alias='{alias}'. Checked keys: {[k.models for k in available]}"
                    )
                available = available_with_model

            if not available:
                logger.warning(
                    f"No available keys for provider {self.provider_name} and model {model}"
                )
                return None

            if self.strategy == KeyRotationStrategy.SEQUENTIAL:
                # Find next available key starting from current index
                for i in range(len(self._keys)):
                    idx = (self._current_index + i) % len(self._keys)
                    key = self._keys[idx]
                    if key in available:
                        self._current_index = (idx + 1) % len(self._keys)
                        key.last_used = time.time()
                        if not key.session_start_time:
                            key.session_start_time = key.last_used
                        return key

            elif self.strategy == KeyRotationStrategy.ROUND_ROBIN:
                key = available[self._current_index % len(available)]
                self._current_index += 1
                key.last_used = time.time()
                if not key.session_start_time:
                    key.session_start_time = key.last_used
                return key

            elif self.strategy == KeyRotationStrategy.FALLBACK:
                # For fallback, always use the first available key
                # (typically the primary, falls back to others when exhausted)
                key = available[0]
                key.last_used = time.time()
                if not key.session_start_time:
                    key.session_start_time = key.last_used
                return key

            else:  # RANDOM
                import random

                key = random.choice(available)
                key.last_used = time.time()
                if not key.session_start_time:
                    key.session_start_time = key.last_used
                return key

            return None

    async def mark_rate_limited(
        self, key: ProviderKey, retry_after: Optional[int] = None
    ):
        """
        Mark a key as rate limited.

        Args:
            key: The key that was rate limited
            retry_after: Optional seconds until retry (from provider)
        """
        async with self._lock:
            key.status = KeyStatus.RATE_LIMITED
            cooldown = retry_after or 60  # Default 60 seconds
            key.cooldown_until = time.time() + cooldown
            key.error_count += 1

            logger.warning(
                f"{self.provider_name}: Key rate limited, cooldown for {cooldown}s"
            )

    async def mark_quota_exhausted(self, key: ProviderKey):
        """Mark a key as having exhausted its quota."""
        async with self._lock:
            key.status = KeyStatus.QUOTA_EXHAUSTED
            key.cooldown_until = time.time() + self.cooldown_seconds
            key.error_count = 0

            logger.warning(
                f"{self.provider_name}: Key quota exhausted, cooldown for {self.cooldown_seconds}s"
            )

    async def mark_error(self, key: ProviderKey):
        """Record an error for a key."""
        async with self._lock:
            key.error_count += 1

            if key.error_count >= self.MAX_ERROR_COUNT:
                key.status = KeyStatus.DISABLED
                key.cooldown_until = time.time() + self.DISABLED_COOLDOWN_SECONDS
                logger.warning(
                    f"{self.provider_name}: Key disabled after {key.error_count} errors. Cooldown for {self.DISABLED_COOLDOWN_SECONDS}s"
                )

    async def mark_success(self, key: ProviderKey):
        """Mark a key as successful, reset error count."""
        async with self._lock:
            key.error_count = 0
            if key.status == KeyStatus.RATE_LIMITED:
                key.status = KeyStatus.AVAILABLE

    def get_key_status(self, index: int) -> Optional[Dict]:
        """Get status of a specific key (without exposing the key itself)."""
        if index < 0 or index >= len(self._keys):
            return None

        key = self._keys[index]
        return {
            "index": index,
            "status": key.status.value,
            "last_used": key.last_used,
            "error_count": key.error_count,
            "models": key.models,
            "cooldown_remaining": max(0, key.cooldown_until - time.time())
            if key.cooldown_until
            else 0,
        }

    def get_all_status(self) -> List[Optional[Dict]]:
        """Get status of all keys."""
        return [self.get_key_status(i) for i in range(len(self._keys))]

    def get_stats(self) -> Dict:
        """Get summary statistics."""
        available = len(self._get_available_keys())
        return {
            "total_keys": len(self._keys),
            "available_keys": available,
            "exhausted_keys": len(self._keys) - available,
            "strategy": self.strategy.value,
        }

    async def reset_key(self, index: int):
        """Manually reset a key to available status."""
        if 0 <= index < len(self._keys):
            async with self._lock:
                key = self._keys[index]
                key.status = KeyStatus.AVAILABLE
                key.error_count = 0
                key.cooldown_until = None
                logger.info(f"{self.provider_name}: Key {index} manually reset")
