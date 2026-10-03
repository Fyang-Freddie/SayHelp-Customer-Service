"""HTTP request and after-sales extraction schemas."""

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChatRequest(BaseModel):
    message: NonEmptyText
    conversation_id: str | None = None


class ExtractRequest(BaseModel):
    description: NonEmptyText


class AfterSalesExtraction(BaseModel):
    order_id: str | None = None
    request_type: Literal["退货", "换货", "退款", "维修", "补发", "其他"]
    expected_solution: str | None = None
