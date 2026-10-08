"""Serializable chapter 5 turn state and shared runtime contracts."""
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Annotated, Literal, Protocol, TypedDict

from langgraph.graph.message import add_messages

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

ActionKind = Literal['handoff', 'create_ticket']


class ActionSuggestion(TypedDict, total=False):
    kind: ActionKind
    description: str
    ticket_type: Literal['售后', '投诉', '咨询']


@dataclass(frozen=True)
class Usage:
    """Measured provider counts and estimated fallback tokens stay separate."""
    input_tokens: int = 0
    output_tokens: int = 0
    estimated: int = 0

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.input_tokens, self.output_tokens, self.estimated)):
            raise ValueError('Token usage must be nonnegative integers')

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.estimated

    def to_dict(self) -> dict:
        return asdict(self)


Emit = Callable[[str, dict], Awaitable[None]]


class AgentModel(Protocol):
    async def select(self, messages: list, tools: list, *, max_tokens: int) -> AIMessage: ...
    def stream_reply(self, messages: list, *, max_tokens: int) -> AsyncIterator[AIMessageChunk]: ...


@dataclass
class AgentResult:
    messages: list[BaseMessage]
    answer: str
    suggestions: list[ActionSuggestion]
    usage: Usage
    model_calls: int
    tool_calls: int
    stop_reason: str
    citations: list[dict] = field(default_factory=list)


class WorkflowState(TypedDict, total=False):
    conversation_id: int
    turn_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    raw_question: str
    resolved_question: str
    intent: str
    route: str
    filters: dict
    evidence: list[dict]
    citations: list[dict]
    confidence: dict
    model_calls: int
    tool_calls: int
    usage: dict
    suggestions: list[ActionSuggestion]
    answer: str
    status: str
    stop_reason: str
    message_id: str
    low_confidence_recorded: bool
    logging_error: bool
