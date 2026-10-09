"""Offline CLI tests replace dependencies only in tests; runtime has no mock mode."""
import asyncio
from contextlib import contextmanager
import inspect
import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.config import Settings
from scripts import demo_ch05 as demo


class ProviderDouble:
    """Never contact a provider; exercise the actual loop and graph offline."""
    def __init__(self):
        self.results = []

    async def classify_intent(self, text):
        return AIMessage(content='{"intent":"订单"}')

    async def select(self, messages, tools, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        self.results = results
        if not results:
            return AIMessage(content='', tool_calls=[{'id': 'order-1', 'name': 'query_order',
                                                     'args': {'order_id': '1001'}}])
        return AIMessage(content='')

    async def stream_reply(self, messages, **kwargs):
        yield AIMessageChunk(content='数据源尚未接入，请人工核实。')


@pytest.fixture
def real_factories_with_test_dependencies(monkeypatch):
    settings = Settings('https://example.test/v1', 'configured-model', 'offline', database_url='sqlite://')
    observed = []
    provider = ProviderDouble()
    monkeypatch.setattr(demo.Settings, 'from_env', lambda: settings)
    def model_factory(config):
        observed.append(('model', config))
        return provider
    monkeypatch.setattr(demo, 'ModelService', model_factory)
    @contextmanager
    def gate_factory(config, **kwargs):
        observed.append(('knowledge', config))
        yield object()  # Business route must never retrieve knowledge.
    monkeypatch.setattr(demo, 'create_live_knowledge_gate', gate_factory)
    return settings, observed, provider


@pytest.mark.parametrize('mode', ['bare', 'workflow'])
def test_demo_always_uses_configured_model_and_existing_knowledge(mode, real_factories_with_test_dependencies):
    settings, observed, provider = real_factories_with_test_dependencies
    result = asyncio.run(demo.run_demo(mode, 'multi-step'))
    assert observed == [('model', settings), ('knowledge', settings)]
    assert result['model_source'] == 'live configured provider'
    assert result['knowledge_source'] == 'existing read-only corpus'
    assert result['business_data'] == 'data source not connected'
    assert result['model'] == 'configured-model'
    assert result['complete_chat_requests'] == 1
    assert result['logical_model_requests'] == sum(result['model_operations'].values()) > 0
    assert result['model_operations']['classification'] == int(mode == 'workflow')
    assert result['tool_names'] == ['query_order'] and result['ticket_writes'] == 0
    assert json.loads(provider.results[0].content)['available'] is False
    assert '模拟' not in result['question']
    assert result['answer'] == '数据源尚未接入，请人工核实。'


@pytest.mark.parametrize('mode', ['bare', 'workflow'])
def test_demo_configuration_failure_never_falls_back_to_authored_model(mode, monkeypatch):
    def invalid_settings():
        raise ValueError('CHAT_API_KEY is required')
    monkeypatch.setattr(demo.Settings, 'from_env', invalid_settings)
    with pytest.raises(ValueError, match='CHAT_API_KEY'):
        asyncio.run(demo.run_demo(mode, 'logistics'))


def test_demo_has_no_runtime_simulation_switch():
    assert 'live' not in inspect.signature(demo.run_demo).parameters
    assert not hasattr(demo, 'SimulationModel')
    assert not hasattr(demo, 'SimulationRetrieval')


def test_both_prompts_explain_unavailable_business_data():
    from app.prompts import render_service_system_prompt, render_workflow_system_prompt
    for prompt in (render_service_system_prompt(), render_workflow_system_prompt()):
        assert '数据源尚未接入' in prompt
        assert '随机模拟' not in prompt and '模拟证据' not in prompt
