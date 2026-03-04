"""Ollama Cloud provider implementation."""

import json
from typing import AsyncGenerator, Dict, Any, List, Optional
import logging

from .base_provider import BaseProvider, ProviderResponse, ProviderHealth
from ..core.key_rotation import KeyManager, KeyRotationStrategy
from ..core.quota_manager import QuotaManager

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """
    Ollama Cloud API provider.

    Supports streaming and non-streaming chat completions.
    Implements OpenAI-compatible API format.
    """

    DEFAULT_BASE_URL = "https://ollama.com/v1"
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

        # Initialize httpx client
        import httpx

        self._client = None

        logger.info(f"Initialized Ollama provider with base_url: {self.base_url}")

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
            "Accept": "application/json",
        }

    def _map_request(
        self, messages: List[Dict], model: str, stream: bool, **kwargs
    ) -> Dict:
        """Map gateway request to Ollama format."""
        # Map model alias to actual model
        actual_model = self.get_model_mapping(model) or model

        request = {
            "model": actual_model,
            "messages": messages,
            "stream": stream,
        }

        # Map optional parameters
        if "temperature" in kwargs:
            request["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            request["max_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            request["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            request["stop"] = kwargs["stop"]
        if "presence_penalty" in kwargs:
            request["presence_penalty"] = kwargs["presence_penalty"]
        if "frequency_penalty" in kwargs:
            request["frequency_penalty"] = kwargs["frequency_penalty"]
        if "seed" in kwargs:
            request["seed"] = kwargs["seed"]

        return request

    async def chat_completion(
        self, messages: List[Dict[str, Any]], model: str, stream: bool = False, alias: Optional[str] = None, **kwargs
    ) -> ProviderResponse:
        """Send chat completion request to Ollama with automatic key rotation on failure."""
        import httpx

        self.total_requests += 1
        model = model.strip()
        
        # We try until we run out of valid keys or reach max_retries
        tried_keys = set()
        last_response = None
        
        for attempt in range(self.max_retries * 2): # Allow more attempts if we have many keys
            key_obj = await self.key_manager.get_next_key(model, alias=alias)
            
            # If no more keys or we've tried all of them and they are failing
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
                
                last_response = response

                if response.status_code == 200:
                    await self.key_manager.mark_success(key_obj)
                    return ProviderResponse(
                        success=True,
                        data=response.json(),
                        status_code=200,
                        headers=dict(response.headers),
                    )

                # Handle errors and rotate key
                error_data = (
                    response.json()
                    if response.text
                    else {"error": {"message": response.reason_phrase}}
                )

                if response.status_code == 429:
                    retry_after = QuotaManager.extract_retry_after(response)
                    await self.key_manager.mark_rate_limited(key_obj, retry_after)
                    logger.warning(f"{self.name}: Key rate limited, retrying with next key...")
                    continue # Try next key

                if response.status_code in [402, 403]:
                    await self.key_manager.mark_quota_exhausted(key_obj)
                    logger.warning(f"{self.name}: Key quota exhausted, retrying with next key...")
                    continue # Try next key

                # Other API errors (5xx etc)
                await self.key_manager.mark_error(key_obj)
                logger.warning(f"{self.name}: API error {response.status_code}, retrying with next key...")
                continue

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                await self.key_manager.mark_error(key_obj)
                logger.warning(f"{self.name}: Network/Timeout error, retrying with next key...")
                continue
            except Exception as e:
                await self.key_manager.mark_error(key_obj)
                logger.error(f"{self.name}: Unexpected request error: {e}")
                continue

        # If we reach here, we exhausted available keys
        self.failed_requests += 1
        error_msg = "All API keys exhausted"
        if last_response is not None:
             error_msg = f"All API keys exhausted. Last error {last_response.status_code}: {last_response.text}"
             
        return ProviderResponse(
            success=False,
            error=Exception(error_msg),
            status_code=503,
        )

    async def chat_completion_stream(
        self, messages: List[Dict[str, Any]], model: str, alias: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion from Ollama with automatic key rotation on failure."""
        import httpx

        self.total_requests += 1
        self.active_requests += 1
        model = model.strip()

        try:
            tried_keys = set()
            last_error = "All API keys exhausted"

            for attempt in range(self.max_retries * 2):
                key_obj = await self.key_manager.get_next_key(model, alias=alias)
                
                if not key_obj or key_obj.key in tried_keys:
                    break
                    
                tried_keys.add(key_obj.key)
                request_body = self._map_request(messages, model, stream=True, **kwargs)

                try:
                    client = await self._get_client()
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers=self._get_headers(key_obj.key),
                        json=request_body,
                    ) as response:
                        if response.status_code == 200:
                            await self.key_manager.mark_success(key_obj)
                            async for line in response.aiter_lines():
                                if line:
                                    yield f"{line}\n"
                            return

                        # Handle errors and rotate key
                        body = await response.aread()
                        error_data = (
                            json.loads(body)
                            if body
                            else {"error": {"message": response.reason_phrase}}
                        )
                        last_error = f"API error {response.status_code}: {error_data}"

                        if response.status_code == 429:
                            retry_after = QuotaManager.extract_retry_after(response)
                            await self.key_manager.mark_rate_limited(key_obj, retry_after)
                            continue
                            
                        if response.status_code in [402, 403]:
                            await self.key_manager.mark_quota_exhausted(key_obj)
                            continue
                            
                        await self.key_manager.mark_error(key_obj)
                        continue

                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    await self.key_manager.mark_error(key_obj)
                    last_error = str(e)
                    continue

            # If we reach here, we exhausted available keys
            error_data = {
                "error": {
                    "message": last_error,
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "keys_exhausted"
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            self.active_requests -= 1

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from Ollama."""
        import httpx

        try:
            key_obj = await self.key_manager.get_next_key()
            if not key_obj:
                logger.warning("No available keys for listing models")
                return []

            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/models", headers=self._get_headers(key_obj.key)
            )

            if response.status_code == 200:
                data = response.json()
                # Map to OpenAI-compatible format
                models = []
                for model in data.get("data", []):
                    models.append(
                        {
                            "id": model.get("id", model.get("name", "unknown")),
                            "object": "model",
                            "owned_by": "ollama",
                        }
                    )
                return models

            logger.error(f"Failed to list models: {response.status_code}")
            return []

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def health_check(self) -> bool:
        """Check Ollama provider health."""
        import httpx
        import time

        try:
            key_obj = await self.key_manager.get_next_key()
            if not key_obj:
                self.health_status = ProviderHealth.UNHEALTHY
                return False

            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try to list models as health check
                response = await client.get(
                    f"{self.base_url}/models", headers=self._get_headers(key_obj.key)
                )

                if response.status_code == 200:
                    self.health_status = ProviderHealth.HEALTHY
                    self.last_health_check = time.time()
                    return True
                else:
                    self.health_status = ProviderHealth.UNHEALTHY
                    return False

        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            self.health_status = ProviderHealth.UNHEALTHY
            return False
