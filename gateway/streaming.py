"""Streaming utilities for SSE relay."""

import json
import asyncio
from typing import AsyncGenerator, Optional
from fastapi.responses import StreamingResponse
import logging

logger = logging.getLogger(__name__)


async def stream_relay(
    upstream_stream: AsyncGenerator[str, None],
    model: str,
    provider: Optional[str] = None,
    client_disconnected: Optional[asyncio.Event] = None,
) -> AsyncGenerator[str, None]:
    """
    Relay SSE stream from upstream provider to client with minimal latency.
    Injects provider metadata if available.
    """
    try:
        async for line in upstream_stream:
            if client_disconnected and client_disconnected.is_set():
                break

            if not line or line.startswith(":"):
                continue

            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break

                try:
                    # Parse and inject provider/model info
                    # We do a lightweight injection to maintain speed
                    if provider:
                        chunk_data = json.loads(data_str)
                        chunk_data["provider"] = provider
                        if (
                            "model" not in chunk_data
                            or chunk_data["model"] == "unknown"
                        ):
                            chunk_data["model"] = model

                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    else:
                        yield f"{line}\n\n"

                except Exception as parse_error:
                    # Fallback to direct relay if JSON is malformed
                    logger.debug(f"Stream JSON parse error: {parse_error}")
                    yield f"{line}\n\n"
            else:
                yield f"{line}\n"

    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"Error in stream relay: {e}")
        error_chunk = {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


def create_streaming_response(
    generator: AsyncGenerator[str, None], media_type: str = "text/event-stream"
) -> StreamingResponse:
    """
    Create a StreamingResponse with proper headers.

    Args:
        generator: The async generator producing SSE events
        media_type: The media type for the response

    Returns:
        StreamingResponse configured for SSE
    """
    return StreamingResponse(
        generator,
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


async def parse_sse_stream(response) -> AsyncGenerator[str, None]:
    """
    Parse SSE stream from httpx response with byte-level processing for speed.
    """
    buffer = b""
    try:
        async for chunk in response.aiter_bytes():
            buffer += chunk

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                yield line.decode("utf-8").rstrip("\r")

        if buffer:
            yield buffer.decode("utf-8").rstrip("\r")

    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"Error parsing SSE stream: {e}")
        raise
