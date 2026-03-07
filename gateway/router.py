"""Router for selecting and managing providers."""

import time
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

from .providers.base_provider import BaseProvider, ProviderHealth
from .core.circuit_breaker import CircuitBreakerRegistry
from .models import HealthStatus

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategy for provider selection."""

    SEQUENTIAL = "sequential"  # Try providers in order
    FALLBACK = "fallback"  # Primary with fallback
    LOAD_BALANCE = "load_balance"  # Distribute load evenly
    SMART = "smart"  # Based on health and latency


@dataclass
class RouteResult:
    """Result of a routing decision."""

    provider: BaseProvider
    model: str
    fallback_providers: List[BaseProvider]


class ProviderRouter:
    """
    Routes requests to appropriate providers based on configuration,
    health status, and routing strategy.

    Features:
    - Model aliasing (smart → ollama/nvidia)
    - Provider selection based on health
    - Automatic fallback
    - Load balancing
    - Circuit breaker integration
    """

    def __init__(
        self, providers: Dict[str, BaseProvider], models_config: Dict[str, Any]
    ):
        self.providers = providers
        self.models_config = models_config
        self.circuit_breakers = CircuitBreakerRegistry()
        self._provider_stats: Dict[str, Dict] = {}

        logger.info(f"Initialized router with {len(providers)} providers")

    def get_provider_for_model(
        self, model_alias: str, strategy: RoutingStrategy = RoutingStrategy.FALLBACK
    ) -> Optional[RouteResult]:
        """
        Select provider(s) for a model alias.

        Args:
            model_alias: Model alias or actual model name
            strategy: Routing strategy to use

        Returns:
            RouteResult with primary and fallback providers
        """
        # Get configured providers for this model alias
        model_config = self.models_config.get(model_alias)

        if model_config:
            provider_names = model_config.get("providers", [])
            logger.debug(f"Resolved {model_alias} from config. Providers: {provider_names}")
        else:
            # If no alias, try to find provider that supports this model directly
            provider_names = [
                name
                for name, provider in self.providers.items()
                if provider.supports_model(model_alias)
            ]
            if not provider_names:
                # Default to first provider if no match
                provider_names = list(self.providers.keys())[:1]
                logger.debug(f"Direct model match failed for {model_alias}. Defaulting to: {provider_names}")

        # Filter to healthy providers
        available_providers = []
        for name in provider_names:
            if name in self.providers:
                provider = self.providers[name]
                circuit_breaker = self.circuit_breakers.get_or_create(name)

                if provider.is_healthy and circuit_breaker.is_closed:
                    available_providers.append(provider)
                else:
                    logger.debug(f"Provider {name} unhealthy or circuit open. Healthy: {provider.is_healthy}, CB: {circuit_breaker.state}")

        if not available_providers:
            logger.warning(f"No healthy providers available for model {model_alias}")
            # Try any provider as last resort
            available_providers = [
                p
                for p in self.providers.values()
                if self.circuit_breakers.get_or_create(p.name).is_closed
            ]

        if not available_providers:
            logger.error(f"All providers unavailable for model {model_alias}")
            return None

        # Select based on strategy
        if strategy == RoutingStrategy.SEQUENTIAL:
            primary = available_providers[0]
            fallback = available_providers[1:]
        elif strategy == RoutingStrategy.LOAD_BALANCE:
            # Select least loaded
            primary = min(available_providers, key=lambda p: p.active_requests)
            fallback = [p for p in available_providers if p != primary]
        elif strategy == RoutingStrategy.SMART:
            # Select based on health and success rate
            primary = max(
                available_providers, key=lambda p: p.get_stats()["success_rate"]
            )
            fallback = [p for p in available_providers if p != primary]
        else:  # FALLBACK
            primary = available_providers[0]
            fallback = available_providers[1:]

        return RouteResult(
            provider=primary, model=model_alias, fallback_providers=fallback
        )

    async def route_request(
        self,
        model_alias: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """
        Route a chat completion request to the appropriate provider.

        Args:
            model_alias: Model alias or actual model name
            messages: List of conversation messages
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            ProviderResponse or AsyncGenerator for streaming
        """
        route = self.get_provider_for_model(model_alias)

        if not route:
            raise Exception(f"No providers available for model {model_alias}")

        providers_to_try = [route.provider] + route.fallback_providers
        last_error = None

        for provider in providers_to_try:
            logger.debug(f"Trying provider {provider.name} for model {model_alias}")
            circuit_breaker = self.circuit_breakers.get_or_create(provider.name)

            if not await circuit_breaker.can_execute():
                logger.info(f"Skipping {provider.name} - circuit breaker open")
                continue

            try:
                # Map model alias to provider-specific model
                actual_model = provider.get_model_mapping(model_alias) or model_alias

                if stream:
                    # For streaming, return the generator
                    return await self._execute_stream_with_fallback(
                        provider, providers_to_try, messages, actual_model, alias=model_alias, **kwargs
                    )
                else:
                    # For non-streaming, try the request
                    response = await provider.chat_completion(
                        messages=messages, model=actual_model, stream=False, alias=model_alias, **kwargs
                    )

                    if response.success:
                        await circuit_breaker.record_success()
                        # Inject provider info
                        data = response.data
                        if isinstance(data, dict):
                            data["provider"] = provider.name
                        return data

                    # Handle specific errors
                    if response.is_rate_limit:
                        logger.warning(f"Rate limited on {provider.name}")
                        last_error = response.error
                        continue

                    if response.is_quota_exhausted:
                        logger.warning(f"Quota exhausted on {provider.name}")
                        last_error = response.error
                        continue

                    # General failure
                    await circuit_breaker.record_failure()
                    last_error = response.error

            except Exception as e:
                logger.error(f"Error with provider {provider.name}: {e}")
                await circuit_breaker.record_failure()
                last_error = e

        # All providers failed
        raise Exception(f"All providers failed. Last error: {last_error}")

    async def _execute_stream_with_fallback(
        self,
        primary: BaseProvider,
        fallback_providers: List[BaseProvider],
        messages: List[Dict],
        model: str,
        alias: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Execute streaming request with fallback support.

        Yields SSE-formatted chunks from the successful provider.
        """
        providers_to_try = [primary] + [p for p in fallback_providers if p != primary]

        for provider in providers_to_try:
            circuit_breaker = self.circuit_breakers.get_or_create(provider.name)

            if not await circuit_breaker.can_execute():
                continue

            try:
                # Import here to avoid circularity
                from .streaming import stream_relay

                actual_model = provider.get_model_mapping(model) or model
                stream_gen = provider.chat_completion_stream(
                    messages=messages, model=actual_model, alias=alias, **kwargs
                )

                # Track if we received any data
                received_data = False

                async for chunk in stream_relay(stream_gen, actual_model, provider.name):
                    received_data = True
                    yield chunk

                if received_data:
                    await circuit_breaker.record_success()
                    return

            except Exception as e:
                logger.error(f"Streaming error with {provider.name}: {e}")
                await circuit_breaker.record_failure()
                continue

        # All providers failed for streaming
        error_data = {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            "error": {
                "message": "All providers failed",
                "type": "provider_error",
                "param": None,
                "code": "all_providers_failed"
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"

    async def route_stream(
        self, model_alias: str, messages: List[Dict[str, Any]], **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Route a streaming chat completion request.

        Args:
            model_alias: Model alias or actual model name
            messages: List of conversation messages
            **kwargs: Additional parameters

        Yields:
            SSE-formatted chunks
        """
        route = self.get_provider_for_model(model_alias)

        if not route:
            error_chunk = {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                "error": {
                    "message": "No providers available",
                    "type": "server_error",
                    "param": None,
                    "code": "no_providers_available"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        async for chunk in self._execute_stream_with_fallback(
            route.provider, route.fallback_providers, messages, model_alias, alias=model_alias, **kwargs
        ):
            yield chunk

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models from all providers.

        Returns:
            List of model information dictionaries
        """
        models = []
        seen_models = set()

        # Add configured model aliases
        for alias, config in self.models_config.items():
            models.append(
                {
                    "id": alias,
                    "object": "model",
                    "owned_by": "llm-gateway",
                    "providers": config.get("providers", []),
                }
            )
            seen_models.add(alias)

        # Add provider-specific models
        for name, provider in self.providers.items():
            for alias, actual in provider.config.get("models", {}).items():
                if alias not in seen_models:
                    models.append(
                        {
                            "id": alias,
                            "object": "model",
                            "owned_by": name,
                            "actual_model": actual,
                        }
                    )
                    seen_models.add(alias)

        return models

    def get_health_status(self) -> Dict[str, HealthStatus]:
        """
        Get health status of all providers.

        Returns:
            Dictionary mapping provider names to HealthStatus
        """
        statuses = {}

        for name, provider in self.providers.items():
            stats = provider.get_stats()
            circuit_status = self.circuit_breakers.get(name)

            # Get key status from provider
            available_keys = 0
            total_keys = 0
            if hasattr(provider, "key_manager"):
                key_stats = provider.key_manager.get_stats()
                available_keys = key_stats["available_keys"]
                total_keys = key_stats["total_keys"]

            statuses[name] = HealthStatus(
                status="ok" if provider.is_healthy else "unhealthy",
                provider=name,
                healthy=provider.is_healthy,
                available_keys=available_keys,
                total_keys=total_keys,
                active_requests=stats["active_requests"],
                last_error=None,
            )

        return statuses

    async def run_health_checks(self):
        """Run health checks on all providers."""
        logger.info("Running health checks on all providers")

        for name, provider in self.providers.items():
            try:
                healthy = await provider.health_check()
                circuit_breaker = self.circuit_breakers.get_or_create(name)

                if healthy:
                    await circuit_breaker.record_success()
                    logger.info(f"Provider {name} is healthy")
                else:
                    await circuit_breaker.record_failure()
                    logger.warning(f"Provider {name} health check failed")

            except Exception as e:
                logger.error(f"Health check error for {name}: {e}")
                circuit_breaker = self.circuit_breakers.get_or_create(name)
                await circuit_breaker.record_failure()
