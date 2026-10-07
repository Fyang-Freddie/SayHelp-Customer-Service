"""Strict seven-label classification and deterministic workflow routing."""
import json
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict

from app.intent_prompts import intent_messages
from app.workflow_types import Usage

if TYPE_CHECKING:
    from app.model_service import ModelService

Intent = Literal['物流', '订单', '商品咨询', '退款退货', '售后', '投诉', '闲聊']
Route = Literal['knowledge', 'business', 'complaint', 'chitchat']


class IntentDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    intent: Intent


class IntentClassificationError(ValueError):
    """Invalid provider classification; retains paid usage without exposing raw content."""
    def __init__(self, usage: Usage):
        super().__init__('Intent classification returned invalid structured JSON')
        self.usage = usage


def _unique_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('Non-JSON constant')


def _usage(text: str, result: AIMessage) -> Usage:
    metadata = result.usage_metadata
    if metadata is not None:
        return Usage(input_tokens=metadata['input_tokens'], output_tokens=metadata['output_tokens'])
    # Conservative character estimate, including both prompt messages and output.
    # Never label this as provider-measured usage.
    estimate = sum(len(str(message.content)) + 16 for message in intent_messages(text))
    return Usage(estimated=estimate + len(str(result.content)) + 16)


async def classify_intent(text: str, *, model: 'ModelService') -> tuple[IntentDecision, Usage]:
    result = await model.classify_intent(text)
    if not isinstance(result, AIMessage):
        raise IntentClassificationError(Usage())
    usage = _usage(text, result)
    try:
        raw = result.content
        if not isinstance(raw, str) or len(raw) > 50000 or result.tool_calls or result.invalid_tool_calls:
            raise ValueError('Expected bounded JSON without tool calls')
        payload = json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
        return IntentDecision.model_validate(payload), usage
    except (ValueError, TypeError):
        raise IntentClassificationError(usage) from None


def resolve_reference(text: str) -> str:
    return text


def route_intent(intent: str) -> Route:
    routes: dict[Intent, Route] = {
        '物流': 'business', '订单': 'business', '商品咨询': 'knowledge',
        '退款退货': 'knowledge', '售后': 'business', '投诉': 'complaint', '闲聊': 'chitchat',
    }
    try:
        return routes[intent]
    except (KeyError, TypeError):
        raise ValueError('Unknown intent; routing rejected') from None
