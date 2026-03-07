"""Pydantic models for request/response handling."""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class ToolCallFunction(BaseModel):
    """Represents the function called by a tool call."""

    name: str = Field(..., description="The name of the function to call")
    arguments: str = Field(
        ..., description="The arguments to call the function with, as a JSON string"
    )


class ToolCall(BaseModel):
    """Represents a tool call in a message."""

    id: str = Field(..., description="The ID of the tool call")
    type: Literal["function"] = Field("function", description="The type of tool call")
    function: ToolCallFunction = Field(..., description="The function to call")


class FunctionDefinition(BaseModel):
    """Definition of a function that can be called by the model."""

    name: str = Field(..., description="The name of the function")
    description: Optional[str] = Field(
        None, description="A description of the function"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="The parameters the function accepts, as a JSON Schema object",
    )


class ToolDefinition(BaseModel):
    """Definition of a tool available to the model."""

    type: Literal["function"] = Field("function", description="The type of the tool")
    function: FunctionDefinition = Field(..., description="The function definition")


class ToolChoiceFunction(BaseModel):
    """Function specification when forcing a specific tool choice."""

    name: str = Field(..., description="The name of the function to call")


class ToolChoice(BaseModel):
    """Specification for which tool the model should use."""

    type: Literal["function"] = Field("function", description="The type of the tool")
    function: ToolChoiceFunction = Field(..., description="The function to use")


class Message(BaseModel):
    role: str = Field(
        ...,
        description="The role of the message author (system, user, assistant, tool)",
    )
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        None, description="The content of the message"
    )
    name: Optional[str] = Field(
        None, description="The name of the author of this message"
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        None, description="Tool calls made by the assistant"
    )
    tool_call_id: Optional[str] = Field(
        None, description="Tool call that this message is responding to"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format suitable for API requests."""
        result: Dict[str, Any] = {"role": self.role}

        if self.content is not None:
            result["content"] = self.content

        if self.name is not None:
            result["name"] = self.name

        if self.tool_calls is not None:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]

        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id

        return result


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
    tools: Optional[List[ToolDefinition]] = Field(
        None, description="List of tools the model may call"
    )
    tool_choice: Optional[Union[str, ToolChoice]] = Field(
        None, description="Tool choice configuration ('auto', 'none', or specific tool)"
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
