"""Offline behavior contracts for the bounded chapter 5 runtime."""
import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.agent_runtime import AgentLimits, run_bare_agent
from app.workflow_types import Usage


class ScriptModel:
    def __init__(self, decisions, chunks=None):
        self.decisions = iter(decisions)
        self.selections = []
        self.finals = []
        self.chunks = chunks or [AIMessageChunk(content='模拟'), AIMessageChunk(content='结果')]

    async def select(self, messages, tools, *, max_tokens):
        self.selections.append(list(messages))
        return next(self.decisions)

    async def stream_reply(self, messages, *, max_tokens):
        self.finals.append(list(messages))
        for chunk in self.chunks:
            yield chunk


def decision(name, ident='c1', args=None, content=''):
    return AIMessage(content=content, tool_calls=[{'name': name, 'id': ident, 'args': args or {'order_id': '1001'}}])


def scenario(model, tools=None, limits=None, usage=None, messages=None):
    events = []
    async def emit(name, payload):
        events.append((name, payload))
    result = asyncio.run(run_bare_agent(messages or [HumanMessage(content='查询1001')], model=model,
        tools=tools or {}, limits=limits or AgentLimits(), usage=usage or Usage(), emit=emit))
    return result, events


def business_tools(attempts):
    @tool
    async def query_order(order_id: str) -> str:
        """Read order demo."""
        attempts.append('order')
        return '订单已发货（模拟）'
    @tool
    async def query_logistics(order_id: str) -> str:
        """Read logistics demo."""
        attempts.append('logistics')
        return '物流运输中（模拟）'
    return {'query_order': query_order, 'query_logistics': query_logistics}


def test_order_then_logistics_feeds_each_result():
    attempts = []
    model = ScriptModel([decision('query_order'), decision('query_logistics', 'c2'), AIMessage(content='内部草稿')])
    result, events = scenario(model, business_tools(attempts))
    assert attempts == ['order', 'logistics']
    assert any(isinstance(m, ToolMessage) and '已发货' in m.content for m in model.selections[1])
    assert any(isinstance(m, ToolMessage) and '运输中' in m.content for m in model.selections[2])
    assert result.tool_calls == 2 and result.model_calls == 4
    assert result.answer == '模拟结果'
    assert [payload['text'] for name, payload in events if name == 'token'] == ['模拟', '结果']
    assert len(model.finals) == 1
    assert all(m.content != '内部草稿' for m in model.finals[0])


def test_selection_text_before_tool_call_is_internal():
    model = ScriptModel([decision('query_order', content='暂定正文'), AIMessage(content='终止草稿')])
    result, events = scenario(model, business_tools([]))
    assert result.answer == '模拟结果'
    assert '暂定正文' not in ''.join(p['text'] for n,p in events if n == 'token')
    assert all(m.content != '终止草稿' for m in model.finals[0])


def test_budget_reserves_final_generation():
    model = ScriptModel([decision('suggest_actions', str(n), {'actions': [{'kind': 'handoff'}]}) for n in range(10)])
    result, _ = scenario(model)
    assert len(model.selections) == 5 and len(model.finals) == 1
    assert result.model_calls == 6 and result.tool_calls == 0
    assert result.stop_reason == 'model_budget'


def test_classification_usage_can_exhaust_budget_without_model_request():
    model = ScriptModel([])
    result, _ = scenario(model, usage=Usage(input_tokens=11990, output_tokens=10))
    assert not model.selections and not model.finals
    assert result.model_calls == 0 and result.stop_reason == 'token_budget'
    assert result.usage.input_tokens == 11990


def test_measured_and_estimated_usage_are_separate():
    model = ScriptModel([AIMessage(content='draft', usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15})],
        [AIMessageChunk(content='回答'), AIMessageChunk(content='', usage_metadata={'input_tokens': 12, 'output_tokens': 3, 'total_tokens': 15})])
    result, _ = scenario(model, usage=Usage(input_tokens=7, output_tokens=2, estimated=4))
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.estimated) == (29, 10, 4)
    missing, _ = scenario(ScriptModel([AIMessage(content='draft')]))
    assert missing.usage.input_tokens == missing.usage.output_tokens == 0
    assert missing.usage.estimated > 0


def test_repeated_failures_stop():
    attempts = []
    @tool('query_order')
    async def failing(order_id: str) -> str:
        """Read demo."""
        attempts.append(order_id)
        raise RuntimeError('private diagnostic')
    model = ScriptModel([decision('query_order'), decision('query_order', 'c2')])
    result, events = scenario(model, {'query_order': failing})
    assert attempts == ['1001', '1001']
    assert result.tool_calls == 2 and len(model.selections) == 1
    assert result.stop_reason == 'repeated_tool_failure'
    assert len([e for e in events if e[0] == 'tool_attempt']) == 2
    assert 'private diagnostic' not in str(result.messages)


@pytest.mark.parametrize('calls', [
    [{'name': 'query_order', 'id': '', 'args': {'order_id': '1001'}}],
    [{'name': 'query_order', 'id': 'same', 'args': {'order_id': '1001'}},
     {'name': 'query_logistics', 'id': 'same', 'args': {'order_id': '1001'}}],
])
def test_invalid_or_duplicate_ids_stop_before_execution(calls):
    attempts = []
    result, _ = scenario(ScriptModel([AIMessage(content='', tool_calls=calls)]), business_tools(attempts))
    assert attempts == [] and result.tool_calls == 0
    assert result.stop_reason == 'invalid_tool_calls'
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in result.messages)


def test_denies_ticket_even_if_model_requests_it():
    attempts = []
    @tool
    async def create_ticket(order_id: str) -> str:
        """Write ticket."""
        attempts.append(order_id)
        return 'ticket'
    result, _ = scenario(ScriptModel([decision('create_ticket')]), {'create_ticket': create_ticket})
    assert attempts == [] and result.tool_calls == 0
    assert result.stop_reason == 'denied_tool'
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == 'c1' and m.status == 'error' for m in result.messages)


def test_actual_read_retries_cannot_cross_tool_budget():
    attempts = []
    @tool('query_order')
    async def failing(order_id: str) -> str:
        """Read demo."""
        attempts.append(order_id)
        raise RuntimeError('fail')
    result, _ = scenario(ScriptModel([decision('query_order')]), {'query_order': failing}, AgentLimits(tool_calls=1))
    assert attempts == ['1001'] and result.tool_calls == 1
    assert result.stop_reason == 'tool_budget'


def test_suggestion_has_matching_result_and_no_business_execution():
    model = ScriptModel([decision('suggest_actions', args={'actions': [{'kind': 'handoff'}]}), AIMessage(content='ask')])
    result, _ = scenario(model)
    assert result.tool_calls == 0
    assert result.suggestions == [{'kind': 'handoff'}]
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == 'c1' for m in model.selections[1])


def test_unpaired_input_stops_without_request():
    model = ScriptModel([])
    result, _ = scenario(model, messages=[decision('query_order')])
    assert result.stop_reason == 'invalid_messages'
    assert not model.selections and not model.finals


def test_model_service_select_binds_tools_and_final_stream_is_unbound():
    from app.config import Settings
    from app.model_service import ModelService
    class Transport:
        def __init__(self):
            self.bindings = []
            self.requests = []
        def bind_tools(self, tools, **kwargs):
            self.bindings.append((tools, kwargs))
            return self
        async def ainvoke(self, messages, **kwargs):
            self.requests.append(kwargs)
            return AIMessage(content='draft', usage_metadata={'input_tokens': 2,'output_tokens': 1,'total_tokens': 3})
        async def astream(self, messages, **kwargs):
            self.requests.append(kwargs)
            yield AIMessageChunk(content='answer')
            yield AIMessageChunk(content='',usage_metadata={'input_tokens': 3,'output_tokens': 2,'total_tokens': 5})
    service = ModelService(Settings(chat_base_url='https://example.test/v1',chat_model='offline',chat_api_key='offline'))
    transport = Transport()
    service._model = transport
    async def run():
        selected = await service.select([], ['tool-schema'], max_tokens=37)
        chunks = [c async for c in service.stream_reply([], max_tokens=41)]
        return selected, chunks
    selected, chunks = asyncio.run(run())
    assert selected.usage_metadata['input_tokens'] == 2
    assert chunks[-1].usage_metadata['output_tokens'] == 2
    assert transport.bindings == [(['tool-schema'], {})]
    assert transport.requests == [{'max_tokens': 37}, {'max_tokens': 41, 'stream_usage': True}]


def test_reused_id_in_next_selection_stops_before_second_execution():
    attempts = []
    model = ScriptModel([decision('query_order'), decision('query_logistics')])
    result, _ = scenario(model, business_tools(attempts))
    assert attempts == ['order'] and result.stop_reason == 'invalid_tool_calls'


def test_invalid_call_payload_cannot_trigger_business_work():
    attempts = []
    model = ScriptModel([AIMessage(content='', invalid_tool_calls=[{'name':'query_order', 'args':'{', 'id':'bad', 'error':'json'}])])
    result, _ = scenario(model, business_tools(attempts))
    assert attempts == [] and result.stop_reason == 'invalid_tool_calls'


def test_budget_checks_final_input_before_request():
    model = ScriptModel([])
    result, _ = scenario(model, limits=AgentLimits(model_calls=1, turn_tokens=100))
    assert not model.finals and result.stop_reason == 'token_budget'


def test_graph_steps_return_message_deltas_and_serializable_turn_fields():
    from app.agent_runtime import agent_step, execute_calls, stream_answer
    state = {'messages': [HumanMessage(content='1001')], 'usage': {}, 'model_calls': 0, 'tool_calls': 0}
    model = ScriptModel([decision('query_order')])
    async def emit(name, payload):
        json.dumps(payload)
    async def run():
        first = await agent_step(state, model=model, tools=business_tools([]), limits=AgentLimits(), emit=emit)
        assert len(first['messages']) == 1
        state.update({k:v for k,v in first.items() if k != 'messages'})
        state['messages'] += first['messages']
        second = await execute_calls(state, tools=business_tools([]), limits=AgentLimits(), emit=emit)
        assert len(second['messages']) == 1
        state.update({k:v for k,v in second.items() if k != 'messages'})
        state['messages'] += second['messages']
        third = await stream_answer(state, model=model, limits=AgentLimits(), emit=emit)
        assert third['answer'] == '模拟结果'
        json.dumps({k:v for k,v in third.items() if k != 'messages'})
    asyncio.run(run())


def test_chapter5_settings_load_and_reject_excess_call_budget(monkeypatch, tmp_path):
    from app.config import Settings
    monkeypatch.chdir(tmp_path)
    for key, value in {'CHAT_BASE_URL':'https://example.test/v1', 'CHAT_MODEL':'test', 'CHAT_API_KEY':'offline',
        'DATABASE_URL':'sqlite://', 'AGENT_MODEL_CALLS':'4', 'AGENT_TOOL_CALLS':'3', 'AGENT_TURN_TOKENS':'8000'}.items():
        monkeypatch.setenv(key, value)
    settings = Settings.from_env()
    assert (settings.agent_model_calls, settings.agent_tool_calls, settings.agent_turn_tokens) == (4,3,8000)
    monkeypatch.setenv('AGENT_MODEL_CALLS', '7')
    with pytest.raises(ValueError, match='AGENT_MODEL_CALLS'):
        Settings.from_env()


def test_bare_cli_uses_existing_reads_and_prints_paid_final_step(capsys):
    from app.bare_agent import run_demo
    from app.config import Settings
    settings = Settings(chat_base_url='https://example.test/v1',chat_model='test',chat_api_key='offline',database_url='sqlite://')
    model = ScriptModel([decision('query_order'), decision('query_logistics', 'c2'), AIMessage(content='draft')])
    result = asyncio.run(run_demo('先查订单 1001 的订单状态，再查物流', settings=settings, model=model))
    captured = capsys.readouterr().out
    assert result.tool_calls == 2 and result.model_calls == 4
    assert 'phase' in captured and 'final' in captured and 'model_calls' in captured
    assert result.answer == '模拟结果'
    assert 'create_ticket' not in captured


def test_repeated_validation_failures_stop_even_without_actual_execution():
    attempts = []
    model = ScriptModel([decision('query_order', str(n), {'wrong': '1001'}) for n in range(5)])
    result, _ = scenario(model, business_tools(attempts))
    assert attempts == [] and result.tool_calls == 0
    assert len(model.selections) == 2
    assert result.stop_reason == 'repeated_tool_failure'


@pytest.mark.parametrize('sizes', [[60000], [24000, 24000, 24000]])
def test_oversized_tool_results_preserve_final_stream_and_message_pairing(sizes):
    from langchain_core.messages.utils import count_tokens_approximately
    attempts = []
    @tool('query_faq')
    async def faq(keyword: str) -> dict:
        """Read FAQ evidence."""
        attempts.append(keyword)
        return {'answer': 'e' * sizes[int(keyword)], 'useful': True}
    calls = [{'name': 'query_faq', 'id': f'faq-{n}', 'args': {'keyword': str(n)}} for n in range(len(sizes))]
    selected = AIMessage(content='', tool_calls=calls,
        usage_metadata={'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120})
    model = ScriptModel([selected], [AIMessageChunk(content='结果过大，需要核实。')])
    result, events = scenario(model, {'query_faq': faq})
    assert len(model.finals) == 1 and result.answer == '结果过大，需要核实。'
    assert result.model_calls == 2
    # The rejected generated result still cost an execution. Remaining batch
    # calls get matching errors without performing avoidable business work.
    assert result.tool_calls == (1 if len(sizes) == 1 else 2)
    assert len(attempts) == result.tool_calls
    results = [m for m in model.finals[0] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in results] == [f'faq-{n}' for n in range(len(sizes))]
    rejected = results[0 if len(sizes) == 1 else 1]
    assert rejected.status == 'error' and '未采用' in rejected.content
    assert len(rejected.content) < 100 and 'eeee' not in rejected.content
    assert rejected.additional_kwargs['execution_attempts'] == 1
    if len(sizes) > 1:
        assert results[0].status == 'success' and 'e' * 24000 in results[0].content
        assert results[-1].status == 'error'
    assert result.stop_reason == 'tool_result_budget'
    assert 120 + count_tokens_approximately(model.finals[0]) + 512 <= 12000
    assert result.usage.total <= 12000
    assert len([e for e in events if e[0] == 'tool_attempt']) == result.tool_calls


def test_minimum_paired_error_batch_must_fit_before_selection_is_appended():
    attempts = []
    @tool('query_faq')
    async def faq(keyword: str) -> dict:
        """Read FAQ evidence."""
        attempts.append(keyword)
        return {'answer': 'e' * 60000, 'useful': True}
    calls = [{'name': 'query_faq', 'id': f'c{n}', 'args': {'keyword': 'x'}} for n in range(20)]
    selected = AIMessage(content='', tool_calls=calls)
    model = ScriptModel([selected], [AIMessageChunk(content='需要缩小查询范围，请补充问题。')])
    result, _ = scenario(model, {'query_faq': faq}, usage=Usage(estimated=10220))
    assert len(model.finals) == 1 and result.answer == '需要缩小查询范围，请补充问题。'
    assert result.model_calls == 2 and result.tool_calls == 0 and attempts == []
    assert result.stop_reason == 'tool_result_budget'
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in model.finals[0])
    assert not any(isinstance(m, ToolMessage) for m in model.finals[0])
    assert result.usage.total <= 12000
