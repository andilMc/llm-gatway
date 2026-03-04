"""Base provider interface for LLM Gateway."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProviderResponse:
    """Response from a provider request."""

    def __init__(
        self,
        success: bool,
        data: Optional[Any] = None,
        error: Optional[Exception] = None,
        status_code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        retry_after: Optional[int] = None,
    ):
        self.success = success
        self.data = data
        self.error = error
        self.status_code = status_code
        self.headers = headers or {}
        self.retry_after = retry_after

    @property
    def is_rate_limit(self) -> bool:
        """Check if this is a rate limit response."""
        if self.status_code and self.status_code == 429:
            return True
        if self.error and "rate limit" in str(self.error).lower():
            return True
        return False

    @property
    def is_quota_exhausted(self) -> bool:
        """Check if quota is exhausted."""
        if self.status_code and self.status_code in [402, 403]:
            return True
        if self.error and any(
            x in str(self.error).lower() for x in ["quota", "billing", "payment"]
        ):
            return True
        return False


class ProviderHealth(Enum):
    """Health status of a provider."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement this interface for compatibility
    with the gateway's routing and streaming capabilities.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.health_status = ProviderHealth.UNKNOWN
        self.last_health_check = None
        self.active_requests = 0
        self.total_requests = 0
        self.failed_requests = 0

    @abstractmethod
    async def chat_completion(
        self, messages: List[Dict[str, Any]], model: str, stream: bool = False, alias: Optional[str] = None, **kwargs
    ) -> ProviderResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of messages in the conversation
            model: Model name to use
            stream: Whether to stream the response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            ProviderResponse with result or error
        """
        pass

    @abstractmethod
    async def chat_completion_stream(
        self, messages: List[Dict[str, Any]], model: str, alias: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion tokens.

        Args:
            messages: List of messages in the conversation
            model: Model name to use
            **kwargs: Additional parameters

        Yields:
            SSE-formatted strings
        """
        yield ""

    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from this provider.

        Returns:
            List of model information dictionaries
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @property
    def is_healthy(self) -> bool:
        """Check if provider is currently healthy."""
        return self.health_status == ProviderHealth.HEALTHY

    def get_model_mapping(self, alias: str) -> Optional[str]:
        """
        Map a model alias to the provider's actual model name.

        Args:
            alias: The model alias (e.g., "smart", "fast")

        Returns:
            The actual model name for this provider, or None if not supported
        """
        models = self.config.get("models", {})
        return models.get(alias, alias)

    def supports_model(self, model: str) -> bool:
        """
        Check if this provider supports a given model.

        Args:
            model: Model name or alias

        Returns:
            True if supported
        """
        # Check if it's a direct model name
        supported_models = self.config.get("models", {})
        if model in supported_models.values():
            return True

        # Check if it's an alias
        if model in supported_models:
            return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            "name": self.name,
            "health": self.health_status.value,
            "is_healthy": self.is_healthy,
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                (self.total_requests - self.failed_requests)
                / max(1, self.total_requests)
            ),
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, healthy={self.is_healthy})"
