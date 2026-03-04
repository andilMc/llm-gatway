"""Pydantic models for request/response handling."""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class Message(BaseModel):
    role: str = Field(
        ...,
        description="The role of the message author (system, user, assistant, tool)",
    )
    content: Union[str, List[Dict[str, Any]]] = Field(
        ..., description="The content of the message"
    )
    name: Optional[str] = Field(
        None, description="The name of the author of this message"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tool calls made by the assistant"
    )
    tool_call_id: Optional[str] = Field(
        None, description="Tool call that this message is responding to"
    )


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[Message] = Field(
        ..., description="A list of messages comprising the conversation"
    )
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(
        1, ge=1, description="How many chat completion choices to generate"
    )
    stream: Optional[bool] = Field(
        False, description="Whether to stream back partial progress"
    )
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    max_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens to generate"
    )
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, int]] = Field(
        None, description="Modify likelihood of specified tokens"
    )
    user: Optional[str] = Field(
        None, description="A unique identifier representing your end-user"
    )
    response_format: Optional[Dict[str, str]] = Field(
        None, description="Response format configuration"
    )
    seed: Optional[int] = Field(None, description="Seed for deterministic sampling")
    tools: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of tools the model may call"
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None, description="Tool choice configuration"
    )


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: Optional[str] = Field(
        None, description="stop, length, tool_calls, content_filter, or null"
    )
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: Dict[str, Any] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[StreamChoice]
    system_fingerprint: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    owned_by: str = "llm-gateway"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class HealthStatus(BaseModel):
    status: str
    provider: str
    healthy: bool
    available_keys: int
    total_keys: int
    active_requests: int
    last_error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    providers: Dict[str, HealthStatus]
