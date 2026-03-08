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
        """Send chat completion request to Ollama with automatic key rotation on failure."""
        import httpx

        self.total_requests += 1
        model = model.strip()

        # Get total number of keys to try each one exactly once
        stats = self.key_manager.get_stats()
        total_keys = stats["total_keys"]

        if total_keys == 0:
            self.failed_requests += 1
            return ProviderResponse(
                success=False,
                error=Exception("Aucune clé API configurée"),
                status_code=503,
            )

        # Track failures per key: {key_index: error_reason}
        key_failures = []
        tried_keys = set()

        # Try each key exactly once (one complete round)
        for attempt in range(total_keys):
            key_obj = await self.key_manager.get_next_key(model, alias=alias)

            # If no key available or we've tried all available keys
            if not key_obj:
                break

            # Avoid trying the same key twice in this round
            if key_obj.key in tried_keys:
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
                    response_data = response.json()

                    # Ajouter l'info de la clé API utilisée au dernier message
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        choice = response_data["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            content = choice["message"]["content"]
                            choice["message"]["content"] = (
                                content + f"\n\n(Cle utilisée numero:{key_obj.index})"
                            )
                    elif (
                        "message" in response_data
                        and "content" in response_data["message"]
                    ):
                        content = response_data["message"]["content"]
                        response_data["message"]["content"] = (
                            content + f"\n\n(Cle utilisée numero:{key_obj.index})"
                        )

                    return ProviderResponse(
                        success=True,
                        data=response_data,
                        status_code=200,
                        headers=dict(response.headers),
                    )

                # Handle errors - store detailed failure reason
                error_msg_detail = response.reason_phrase
                try:
                    error_data = response.json()
                    if "error" in error_data and "message" in error_data["error"]:
                        error_msg_detail = error_data["error"]["message"]
                except:
                    pass

                if response.status_code == 429:
                    failure_reason = (
                        f"Rate limit dépassé (HTTP 429) - {error_msg_detail}"
                    )
                    retry_after = QuotaManager.extract_retry_after(response)
                    await self.key_manager.mark_rate_limited(key_obj, retry_after)
                    logger.warning(
                        f"{self.name}: Clé #{key_obj.index} rate limited, tentative avec clé suivante..."
                    )
                elif response.status_code in [402, 403]:
                    failure_reason = f"Quota épuisé (HTTP {response.status_code}) - {error_msg_detail}"
                    await self.key_manager.mark_quota_exhausted(key_obj)
                    logger.warning(
                        f"{self.name}: Clé #{key_obj.index} quota épuisé, tentative avec clé suivante..."
                    )
                else:
                    failure_reason = (
                        f"Erreur API (HTTP {response.status_code}) - {error_msg_detail}"
                    )
                    await self.key_manager.mark_error(key_obj)
                    logger.warning(
                        f"{self.name}: Clé #{key_obj.index} erreur {response.status_code}, tentative avec clé suivante..."
                    )

                key_failures.append((key_obj.index, failure_reason))
                continue

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                failure_reason = f"Erreur réseau - {str(e)}"
                await self.key_manager.mark_error(key_obj)
                logger.warning(
                    f"{self.name}: Clé #{key_obj.index} erreur réseau, tentative avec clé suivante..."
                )
                key_failures.append((key_obj.index, failure_reason))
                continue
            except Exception as e:
                failure_reason = f"Erreur inattendue - {str(e)}"
                await self.key_manager.mark_error(key_obj)
                logger.error(
                    f"{self.name}: Clé #{key_obj.index} erreur inattendue: {e}"
                )
                key_failures.append((key_obj.index, failure_reason))
                continue

        # If we reach here, all keys have failed - create detailed human-readable message
        self.failed_requests += 1

        if key_failures:
            error_lines = [
                "❌ Échec de toutes les clés API après tentative complète :\n"
            ]
            for key_idx, reason in key_failures:
                error_lines.append(f"  • Clé #{key_idx} : {reason}")
            error_msg = "\n".join(error_lines)
        else:
            error_msg = "❌ Aucune clé API disponible pour effectuer la requête"

        return ProviderResponse(
            success=False,
            error=Exception(error_msg),
            status_code=503,
        )

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        alias: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion from Ollama with automatic key rotation on failure."""
        import httpx

        self.total_requests += 1
        self.active_requests += 1
        model = model.strip()

        try:
            # Get total number of keys to try each one exactly once
            stats = self.key_manager.get_stats()
            total_keys = stats["total_keys"]

            if total_keys == 0:
                error_chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                    "error": {"message": "❌ Aucune clé API configurée"},
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Track failures per key: [(key_index, error_reason), ...]
            key_failures = []
            tried_keys = set()

            # Try each key exactly once (one complete round)
            for attempt in range(total_keys):
                key_obj = await self.key_manager.get_next_key(model, alias=alias)

                if not key_obj:
                    break

                # Avoid trying the same key twice in this round
                if key_obj.key in tried_keys:
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
                            final_chunk_found = False
                            async for line in response.aiter_lines():
                                if line:
                                    # Vérifier si c'est le dernier chunk [DONE]
                                    if line.strip() == "data: [DONE]":
                                        final_chunk_found = True
                                        break
                                    yield f"{line}\n"

                            # Ajouter un chunk final avec l'information de la clé
                            key_info_chunk = {
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": f"\n(Cle utilisée numero:{key_obj.index})"
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(key_info_chunk)}\n\n"

                            if not final_chunk_found:
                                yield "data: [DONE]\n\n"
                            return

                        # Handle errors - store detailed failure reason
                        body = await response.aread()
                        error_msg_detail = response.reason_phrase
                        try:
                            error_data = json.loads(body) if body else {}
                            if (
                                "error" in error_data
                                and "message" in error_data["error"]
                            ):
                                error_msg_detail = error_data["error"]["message"]
                        except:
                            pass

                        if response.status_code == 429:
                            failure_reason = (
                                f"Rate limit dépassé (HTTP 429) - {error_msg_detail}"
                            )
                            retry_after = QuotaManager.extract_retry_after(response)
                            await self.key_manager.mark_rate_limited(
                                key_obj, retry_after
                            )
                            logger.warning(
                                f"{self.name}: Clé #{key_obj.index} rate limited, tentative avec clé suivante..."
                            )
                        elif response.status_code in [402, 403]:
                            failure_reason = f"Quota épuisé (HTTP {response.status_code}) - {error_msg_detail}"
                            await self.key_manager.mark_quota_exhausted(key_obj)
                            logger.warning(
                                f"{self.name}: Clé #{key_obj.index} quota épuisé, tentative avec clé suivante..."
                            )
                        else:
                            failure_reason = f"Erreur API (HTTP {response.status_code}) - {error_msg_detail}"
                            await self.key_manager.mark_error(key_obj)
                            logger.warning(
                                f"{self.name}: Clé #{key_obj.index} erreur {response.status_code}, tentative avec clé suivante..."
                            )

                        key_failures.append((key_obj.index, failure_reason))
                        continue

                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    failure_reason = f"Erreur réseau - {str(e)}"
                    await self.key_manager.mark_error(key_obj)
                    logger.warning(
                        f"{self.name}: Clé #{key_obj.index} erreur réseau, tentative avec clé suivante..."
                    )
                    key_failures.append((key_obj.index, failure_reason))
                    continue
                except Exception as e:
                    failure_reason = f"Erreur inattendue - {str(e)}"
                    await self.key_manager.mark_error(key_obj)
                    logger.error(
                        f"{self.name}: Clé #{key_obj.index} erreur inattendue: {e}"
                    )
                    key_failures.append((key_obj.index, failure_reason))
                    continue

            # If we reach here, all keys have failed - create detailed human-readable message
            if key_failures:
                error_lines = [
                    "❌ Échec de toutes les clés API après tentative complète :"
                ]
                for key_idx, reason in key_failures:
                    error_lines.append(f"• Clé #{key_idx} : {reason}")
                error_msg = "\n".join(error_lines)
            else:
                error_msg = "❌ Aucune clé API disponible pour effectuer la requête"

            error_data = {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                "error": {
                    "message": error_msg,
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "keys_exhausted",
                },
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            error_chunk = {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                "error": {"message": str(e)},
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
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
