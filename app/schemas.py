"""HTTP request and after-sales extraction schemas."""
from typing import Annotated, Literal
from app.knowledge_types import KnowledgeFilters
from pydantic import AfterValidator, BaseModel, ConfigDict, StrictBool, StringConstraints

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def validate_conversation_id(value: str) -> str:
    if not 1 <= int(value) <= 18446744073709551615:
        raise ValueError('Conversation ID must be a positive unsigned BIGINT decimal string')
    return value


ConversationId = Annotated[
    str,
    StringConstraints(pattern=r'^[0-9]+$', min_length=1, max_length=20),
    AfterValidator(validate_conversation_id),
]


class ChatRequest(BaseModel):
    message: NonEmptyText
    conversation_id: ConversationId | None = None
    filters: KnowledgeFilters | None = None


class ExtractRequest(BaseModel):
    description: NonEmptyText


class AfterSalesExtraction(BaseModel):
    order_id: str | None = None
    request_type: Literal['退货', '换货', '退款', '维修', '补发', '其他']
    expected_solution: str | None = None


class PinRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    is_pinned: StrictBool


class TicketConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    action_id: Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r'\S')]
    description: NonEmptyText
    ticket_type: Literal['售后', '投诉', '咨询']
