"""FastAPI server for LLM Gateway."""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelListResponse,
    ModelInfo,
    HealthResponse,
)
from .config_loader import ConfigLoader
from .router import ProviderRouter
from .providers import OllamaProvider, NvidiaProvider
from .streaming import stream_relay, create_streaming_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
router: Optional[ProviderRouter] = None
config_loader: Optional[ConfigLoader] = None


async def run_health_checks_periodically():
    """Run health checks every 20 seconds."""
    import asyncio

    while True:
        try:
            if router:
                await router.run_health_checks()
            await asyncio.sleep(20)
        except Exception as e:
            logger.error(f"Health check error: {e}")
            await asyncio.sleep(20)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global router, config_loader

    # Startup
    logger.info("Starting LLM Gateway...")

    # Load configuration
    config_path = os.getenv("GATEWAY_CONFIG", "config/providers.yml")
    config_loader = ConfigLoader(config_path)

    try:
        config_loader.load()
        config_loader.validate()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Initialize providers
    providers_config = config_loader.get_providers_config()
    providers = {}

    if "ollama" in providers_config:
        providers["ollama"] = OllamaProvider("ollama", providers_config["ollama"])

    if "nvidia" in providers_config:
        providers["nvidia"] = NvidiaProvider("nvidia", providers_config["nvidia"])

    if not providers:
        logger.error("No providers configured")
        sys.exit(1)

    # Initialize router
    models_config = config_loader.get_models_config()
    router = ProviderRouter(providers, models_config)

    # Run initial health check
    await router.run_health_checks()

    # Start health check task
    import asyncio

    health_task = asyncio.create_task(run_health_checks_periodically())

    logger.info(f"LLM Gateway started with {len(providers)} providers")

    yield

    # Shutdown
    logger.info("Shutting down LLM Gateway...")
    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass

    logger.info("LLM Gateway shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LLM Gateway",
    description="OpenAI-compatible API gateway for multiple AI providers",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not router:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    provider_statuses = router.get_health_status()
    overall_healthy = all(s.healthy for s in provider_statuses.values())

    return HealthResponse(
        status="ok" if overall_healthy else "degraded",
        providers={name: status for name, status in provider_statuses.items()},
    )


@app.get("/v1/models")
async def list_models():
    """List available models."""
    if not router:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    models = router.list_models()
    model_infos = [
        ModelInfo(id=m["id"], owned_by=m.get("owned_by", "llm-gateway")) for m in models
    ]

    return ModelListResponse(data=model_infos)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Chat completions endpoint."""
    if not router:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    try:
        # Parse request
        body = await request.json()
        chat_request = ChatCompletionRequest(**body)

        # Convert messages to dicts
        messages = [
            {"role": msg.role, "content": msg.content} for msg in chat_request.messages
        ]

        # Get additional parameters
        kwargs = {
            "temperature": chat_request.temperature,
            "max_tokens": chat_request.max_tokens,
            "top_p": chat_request.top_p,
            "stop": chat_request.stop,
            "presence_penalty": chat_request.presence_penalty,
            "frequency_penalty": chat_request.frequency_penalty,
            "seed": chat_request.seed,
        }
        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        logger.info(
            f"Chat completion request: model={chat_request.model}, stream={chat_request.stream}"
        )

        if chat_request.stream:
            # Router now handles relay and provider tagging internally
            stream_gen = router.route_stream(
                model_alias=chat_request.model, messages=messages, **kwargs
            )
            return create_streaming_response(stream_gen)
        else:
            # Non-streaming response
            result = await router.route_request(
                model_alias=chat_request.model,
                messages=messages,
                stream=False,
                **kwargs,
            )

            return JSONResponse(content=result)

    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": {"message": str(e.detail), "type": "invalid_request_error"}},
        )
    except Exception as e:
        logger.error(f"Error processing chat completion: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}},
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_error"}},
    )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Gateway")
    parser.add_argument(
        "--host",
        default=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GATEWAY_PORT", "8000")),
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--config",
        default=os.getenv("GATEWAY_CONFIG", "config/providers.yml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("GATEWAY_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Set config path
    os.environ["GATEWAY_CONFIG"] = args.config

    # Run server
    uvicorn.run(
        "gateway.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=True,
        reload=False,
    )


if __name__ == "__main__":
    main()
