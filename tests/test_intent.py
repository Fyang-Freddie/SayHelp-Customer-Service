"""Contracts for strict intent parsing, fixed routing and single-request classification."""
import asyncio
import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.intent import (
    IntentClassificationError, IntentDecision, classify_intent, resolve_reference, route_intent,
)
from app.model_service import ModelService
from app.config import Settings


@pytest.mark.parametrize(('intent', 'route'), [
    ('物流', 'business'), ('订单', 'business'), ('商品咨询', 'knowledge'),
    ('退款退货', 'knowledge'), ('售后', 'business'), ('投诉', 'complaint'), ('闲聊', 'chitchat'),
])
def test_each_intent_has_a_fixed_route(intent, route):
    assert route_intent(intent) == route


@pytest.mark.parametrize('intent', ['', '退货', 'business', ' 物流', None])
def test_unknown_route_is_rejected(intent):
    with pytest.raises(ValueError):
        route_intent(intent)


@pytest.mark.parametrize('payload', [
    {'intent': '订单', 'route': 'business'}, {'intent': '退货'}, {}, {'intent': 1},
])
def test_decision_has_only_one_enum_field(payload):
    with pytest.raises(ValidationError):
        IntentDecision.model_validate(payload)


class ProviderResult:
    def __init__(self, content, usage=True):
        self.calls = 0
        self.texts = []
        self.result = AIMessage(content=content, usage_metadata=(
            {'input_tokens': 41, 'output_tokens': 9, 'total_tokens': 50} if usage else None))

    async def classify_intent(self, text):
        self.calls += 1
        self.texts.append(text)
        return self.result

    async def understand_query(self, text):
        raise AssertionError('Legacy query understanding must never run')


@pytest.mark.parametrize('intent', ['物流', '订单', '商品咨询', '退款退货', '售后', '投诉', '闲聊'])
def test_valid_json_classifies_once_and_keeps_measured_usage(intent):
    model = ProviderResult(json.dumps({'intent': intent}, ensure_ascii=False))
    result, usage = asyncio.run(classify_intent('订单 001001 的物流到哪了，不要退款', model=model))
    assert result.intent == intent
    assert usage.to_dict() == {'input_tokens': 41, 'output_tokens': 9, 'estimated': 0}
    assert model.calls == 1
    assert model.texts == ['订单 001001 的物流到哪了，不要退款']


@pytest.mark.parametrize('raw', [
    'not JSON', '{"intent":"物流",}', '```json\n{"intent":"物流"}\n```',
    '{"intent":"物流","intent":"订单"}', '{"intent":"物流","x":0}',
    '{"intent":"新类别"}', '{"intent":null}', '{"intent":NaN}',
    '[{"intent":"物流"}]', 'null', '{}', '{"intent":"物流"}\n{"intent":"订单"}',
    pytest.param(' ' * 50001, id='overlong-response'),
])
def test_invalid_classification_is_explicit_and_retains_usage(raw):
    model = ProviderResult(raw)
    with pytest.raises(IntentClassificationError) as error:
        asyncio.run(classify_intent('原问题', model=model))
    assert error.value.usage.total == 50
    assert model.calls == 1
    assert raw not in str(error.value)  # Raw provider material is never exposed in errors.


def test_missing_usage_is_counted_as_estimated_tokens():
    model = ProviderResult('{"intent":"闲聊"}', usage=False)
    _, usage = asyncio.run(classify_intent('你好', model=model))
    assert usage.input_tokens == usage.output_tokens == 0
    assert usage.estimated > len('你好')


def test_tool_call_output_cannot_enter_routing():
    model = ProviderResult('{"intent":"订单"}')
    model.result.tool_calls = [{'id': 'c1', 'name': 'query_order', 'args': {}}]
    with pytest.raises(IntentClassificationError):
        asyncio.run(classify_intent('查订单', model=model))


@pytest.mark.parametrize('text', [' 订单 0001001 不是退款\n', '不投诉，也不要改订单0012', '它的物流呢？', ''])
def test_reference_resolution_preserves_every_character(text):
    assert resolve_reference(text) == text


def test_provider_adapter_returns_raw_json_and_usage_in_one_request(monkeypatch):
    import app.model_service as module
    requests = []
    async def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'offline',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '{"intent":"物流"}'},
                         'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 41, 'completion_tokens': 9, 'total_tokens': 50},
        })
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            monkeypatch.setattr(module, 'ChatOpenAI', lambda **kwargs: ChatOpenAI(**kwargs, http_async_client=client))
            service = ModelService(Settings(chat_base_url='https://example.test/v1', chat_model='offline', chat_api_key='offline'))
            result = await service.classify_intent('订单 1001 的物流到哪了')
            assert result.content == '{"intent":"物流"}'
            assert result.usage_metadata['input_tokens'] == 41
    asyncio.run(scenario())
    assert len(requests) == 1
    assert requests[0]['response_format'] == {'type': 'json_object'}
    assert 'tools' not in requests[0]
    assert requests[0]['messages'][-1] == {'role': 'user', 'content': '订单 1001 的物流到哪了'}


def test_provider_transport_failure_does_not_retry(monkeypatch):
    import app.model_service as module
    requests = []
    async def handler(request):
        requests.append(request)
        return httpx.Response(503, json={'error': {'message': 'offline unavailable', 'type': 'server_error'}})
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            monkeypatch.setattr(module, 'ChatOpenAI', lambda **kwargs: ChatOpenAI(**kwargs, http_async_client=client))
            service = ModelService(Settings(chat_base_url='https://example.test/v1', chat_model='offline', chat_api_key='offline'))
            with pytest.raises(Exception):
                await service.classify_intent('你好')
    asyncio.run(scenario())
    assert len(requests) == 1


@pytest.mark.parametrize(('wrong_ids', 'accepted'), [
    ([], True), (['order-1', 'order-2'], True),
    (['order-1', 'order-2', 'order-3'], False), (['logistics-1'], False),
])
def test_evaluation_requires_threshold_and_every_mandatory_case(wrong_ids, accepted):
    from pathlib import Path
    from scripts.evaluate_ch05 import evaluate_intent
    cases = json.loads(Path('eval/ch05/intent_cases.json').read_text(encoding='utf-8-sig'))
    class LabelProvider:
        async def classify_intent(self, text):
            case = next(case for case in cases if case['text'] == text)
            prediction = '闲聊' if case['id'] in wrong_ids else case['expected']
            return AIMessage(content=json.dumps({'intent': prediction}), usage_metadata={
                'input_tokens': 5, 'output_tokens': 5, 'total_tokens': 10})
    report = asyncio.run(evaluate_intent(cases, model=LabelProvider()))
    assert report['accepted'] is accepted
    assert report['core']['correct'] == 35 - len(wrong_ids)
    assert report['boundary']['correct'] == 7
    assert all(case['model_calls'] == 1 and case['success'] for case in report['cases'])
    assert report['usage'] == {'input_tokens': 210, 'output_tokens': 210, 'estimated': 0}
