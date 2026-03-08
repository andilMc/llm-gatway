"""Google Gemini provider implementation."""

import json
import logging
import time
from typing import AsyncGenerator, Dict, Any, List, Optional

from .base_provider import BaseProvider, ProviderResponse, ProviderHealth
from ..core.key_rotation import KeyManager, KeyRotationStrategy

logger = logging.getLogger(__name__)


class GoogleProvider(BaseProvider):
    """
    Google Gemini API provider using OpenAI-compatible endpoint.

    Supports streaming and non-streaming chat completions.
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    DEFAULT_TIMEOUT = 120.0

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)

        self.base_url = config.get("base_url", self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = config.get("timeout", self.DEFAULT_TIMEOUT)
        self.max_retries = config.get("max_retries", 3)

        # Initialize key manager
        key_strategy = config.get("strategy", "sequential")
        self.key_manager = KeyManager(
            provider_name=name,
            keys=config.get("keys", []),
            strategy=KeyRotationStrategy(key_strategy),
        )

        self._client = None
        logger.info(f"Initialized Google provider with base_url: {self.base_url}")

    async def _get_client(self):
        """Get or create httpx client."""
        import httpx

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _get_headers(self, api_key: str) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _map_request(
        self, messages: List[Dict], model: str, stream: bool, **kwargs
    ) -> Dict:
        """Map gateway request to Google format."""
        actual_model = self.get_model_mapping(model) or model

        request = {
            "model": actual_model,
            "messages": messages,
            "stream": stream,
        }

        # Map optional parameters
        for param in [
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "seed",
        ]:
            if param in kwargs and kwargs[param] is not None:
                request[param] = kwargs[param]

        if "tools" in kwargs and kwargs["tools"]:
            request["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs and kwargs["tool_choice"]:
            request["tool_choice"] = kwargs["tool_choice"]

        return request

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        alias: Optional[str] = None,
        **kwargs,
    ) -> ProviderResponse:
        """Send chat completion request to Google with automatic key rotation."""
        import httpx

        self.total_requests += 1
        model = model.strip()

        stats = self.key_manager.get_stats()
        total_keys = stats["total_keys"]

        if total_keys == 0:
            self.failed_requests += 1
            return ProviderResponse(
                success=False,
                error=Exception("Aucune clé API configurée pour Google"),
                status_code=503,
            )

        tried_keys = set()
        for _ in range(total_keys):
            key_obj = await self.key_manager.get_next_key(model, alias=alias)
            if not key_obj or key_obj.key in tried_keys:
                break

            tried_keys.add(key_obj.key)
            request_body = self._map_request(messages, model, stream, **kwargs)

            try:
                client = await self._get_client()
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(key_obj.key),
                    json=request_body,
                )

                if response.status_code == 200:
                    await self.key_manager.mark_success(key_obj)
                    return ProviderResponse(
                        success=True, data=response.json(), status_code=200
                    )

                # Handle errors
                error_data = (
                    response.json()
                    if response.headers.get("content-type") == "application/json"
                    else response.text
                )
                logger.warning(
                    f"Google API error (status {response.status_code}): {error_data}"
                )

                if response.status_code == 429:
                    await self.key_manager.mark_rate_limited(key_obj)
                    continue

                await self.key_manager.mark_error(key_obj)

            except Exception as e:
                logger.error(f"Request to Google failed: {e}")
                await self.key_manager.mark_error(key_obj)

        self.failed_requests += 1
        return ProviderResponse(
            success=False,
            error=Exception("Tous les essais ont échoué pour Google"),
            status_code=500,
        )

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        alias: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion tokens from Google."""
        import httpx

        self.total_requests += 1
        model = model.strip()

        key_obj = await self.key_manager.get_next_key(model, alias=alias)
        if not key_obj:
            error_data = {
                "error": {"message": "Aucune clé API disponible", "type": "auth_error"}
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
            return

        request_body = self._map_request(messages, model, True, **kwargs)

        try:
            client = await self._get_client()
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(key_obj.key),
                json=request_body,
                timeout=self.timeout,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(
                        f"Google stream error ({response.status_code}): {error_text}"
                    )
                    yield f"data: {error_text.decode()}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield line + "\n"
                        if "[DONE]" in line:
                            break

                await self.key_manager.mark_success(key_obj)

        except Exception as e:
            logger.error(f"Google streaming exception: {e}")
            error_data = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
            await self.key_manager.mark_error(key_obj)

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from Google."""
        # For simplicity, returning common Gemini models if API fails or as default
        return [
            {"id": "gemini-1.5-pro", "object": "model", "owned_by": "google"},
            {"id": "gemini-1.5-flash", "object": "model", "owned_by": "google"},
            {"id": "gemini-2.0-flash-exp", "object": "model", "owned_by": "google"},
        ]

    async def health_check(self) -> bool:
        """Check Google provider health using a minimal chat completion."""
        import httpx
        import time

        try:
            key_obj = await self.key_manager.get_next_key()
            if not key_obj:
                self.health_status = ProviderHealth.UNHEALTHY
                return False

            async with httpx.AsyncClient(timeout=10.0) as client:
                # Use a minimal chat completion instead of /models which might not be supported
                dummy_request = {
                    "model": "gemini-3.1-flash-lite-preview",  # Use a model from the allowed list
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1
                }
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(key_obj.key),
                    json=dummy_request
                )

                if response.status_code == 200:
                    self.health_status = ProviderHealth.HEALTHY
                    self.last_health_check = time.time()
                    return True
                else:
                    logger.warning(f"Google health check failed with status {response.status_code}: {response.text}")
                    self.health_status = ProviderHealth.UNHEALTHY
                    return False
        except Exception as e:
            logger.warning(f"Google health check exception: {e}")
            self.health_status = ProviderHealth.UNHEALTHY
            return False
