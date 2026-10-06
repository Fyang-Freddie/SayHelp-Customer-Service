import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError
from sqlalchemy import select

from app.db import Ticket
from app.repository import Repository
from app.tool_executor import ToolExecutor
from app.tools import build_tools
from test_db import database_url, sessions


class FakeRepository:
    def __init__(self):
        self.keywords = []
        self.tickets = []

    def find_faq(self, *args, **kwargs):
        raise AssertionError('Online tools must never use SQL keyword lookup')

    def create_ticket(self, conversation_id, description, ticket_type):
        self.tickets.append((conversation_id, description, ticket_type))
        return 'T-demo'


class FakeKnowledgeSearch:
    """Deterministic external-search double, never SQL keyword fallback."""
    def __init__(self, repository=None):
        self.keywords = getattr(repository, 'keywords', [])

    def search(self, keyword, *, limit=5):
        self.keywords.append(keyword)
        if keyword == '退货':
            return [SimpleNamespace(question='退货政策是什么', answer='七天内申请', category='售后')]
        if keyword in ('邮费', '邮费是多少'):
            return [SimpleNamespace(question='订单运费如何计算？', answer='运费根据收货地区与订单金额计算，请以结算页显示为准。', category='配送')]
        return []


def call(name, args):
    return {'id': 'call-42', 'name': name, 'args': args}


def execute(executor, name, args):
    return asyncio.run(executor.execute(call(name, args)))


def test_registry_and_pydantic_schemas():
    tools = build_tools(FakeRepository(), 123, knowledge_search=FakeKnowledgeSearch())
    expected = {'query_order': {'order_id'}, 'query_product': {'product_query'},
                'query_logistics': {'order_id'}, 'query_faq': {'keyword', 'filters'},
                'create_ticket': {'description', 'ticket_type'}}
    assert set(tools) == set(expected)
    for name, fields in expected.items():
        assert isinstance(tools[name], BaseTool)
        assert set(tools[name].args_schema.model_fields) == fields
    assert tools['query_faq'].args_schema.model_validate({'keyword': '退货'}).filters is None
    assert tools['query_faq'].args_schema.model_validate({'keyword': '退货', 'filters': {'product_category': '猫砂盆'}}).filters.product_category == '猫砂盆'
    schema = tools['create_ticket'].args_schema.model_json_schema()
    assert schema['properties']['ticket_type']['enum'] == ['售后', '投诉', '咨询']
    with pytest.raises(ValidationError):
        tools['query_order'].args_schema.model_validate({'order_id': 1001})


@pytest.mark.parametrize('name,args', [('query_order', {'order_id': '1001'}),
                                      ('query_product', {'product_query': '鞋'}),
                                      ('query_logistics', {'order_id': '1001'})])
def test_random_demo_results_are_labeled(name, args, monkeypatch):
    choices = []
    def choose(values):
        choices.append(values)
        return values[0]
    monkeypatch.setattr('app.tools.random.choice', choose)
    result = execute(ToolExecutor(build_tools(FakeRepository(), 123, knowledge_search=FakeKnowledgeSearch())), name, args)
    payload = json.loads(result.content)
    assert payload['mock'] is True
    assert '模拟' in payload['label']
    assert choices
    assert result.tool_call_id == 'call-42'


def test_query_faq_contract_and_semantic_postage_hit():
    repo = FakeRepository()
    executor = ToolExecutor(build_tools(repo, 123, knowledge_search=FakeKnowledgeSearch(repo)))
    hit = json.loads(execute(executor, 'query_faq', {'keyword': '退货'}).content)
    postage = json.loads(execute(executor, 'query_faq', {'keyword': '邮费'}).content)
    miss = json.loads(execute(executor, 'query_faq', {'keyword': '火星天气'}).content)
    assert hit['matches'][0]['answer'] == '七天内申请'
    assert miss['matches'] == [] and '未找到' in miss['message']
    assert repo.keywords == ['退货', '邮费', '火星天气']
    assert set(postage) == {'keyword', 'matches', 'message'}
    assert postage['keyword'] == '邮费'
    assert set(postage['matches'][0]) == {'question', 'answer', 'category'}
    assert postage['matches'][0]['question'] == '订单运费如何计算？'


def test_ticket_uses_captured_conversation_and_persists(sessions):
    repo = Repository(sessions)
    cid = repo.create_conversation('guest-tool')
    result = execute(ToolExecutor(build_tools(repo, cid, knowledge_search=FakeKnowledgeSearch(repo))), 'create_ticket',
                     {'description': '商品破损', 'ticket_type': '售后'})
    number = json.loads(result.content)['ticket_no']
    with sessions() as session:
        ticket = session.get(Ticket, number)
        assert (ticket.conversation_id, ticket.description, ticket.ticket_type) == (cid, '商品破损', '售后')
    assert repo.get_conversation(cid).status == '已转人工'


@pytest.mark.parametrize('name,args', [('missing', {}), ('query_order', {}),
    ('query_order', {'order_id': 1001}), ('query_order', {'order_id': ' '}),
    ('query_faq', {'keyword': '退货', 'extra': True}),
    ('create_ticket', {'description': '问题', 'ticket_type': '退款'}),
    ('create_ticket', {'description': '问题', 'ticket_type': '售后', 'conversation_id': 999}),
    ('create_ticket', ['wrong-shape'])])
def test_invalid_calls_are_safe_and_have_no_side_effect(name, args):
    repo = FakeRepository()
    result = execute(ToolExecutor(build_tools(repo, 123, knowledge_search=FakeKnowledgeSearch(repo))), name, args)
    assert result.status == 'error'
    assert result.tool_call_id == 'call-42'
    assert repo.keywords == [] and repo.tickets == []


@pytest.mark.parametrize('name', ['query_order', 'query_product', 'query_logistics', 'query_faq'])
def test_read_timeout_retries_once(name):
    attempts = []
    @tool(name)
    async def delayed(value: str) -> str:
        """Read demo."""
        attempts.append(value)
        if len(attempts) == 1:
            await asyncio.sleep(0.1)
        return 'ok'
    result = execute(ToolExecutor({name: delayed}, timeout_seconds=0.01), name, {'value': 'x'})
    assert result.content == 'ok' and result.status == 'success'
    assert attempts == ['x', 'x']


def test_exhausted_failures_are_safe_and_bounded():
    attempts = []
    @tool('query_faq')
    async def failing(keyword: str) -> str:
        """Read demo."""
        attempts.append(keyword)
        raise RuntimeError('mysql+pymysql://user:secret@host/db')
    result = execute(ToolExecutor({'query_faq': failing}), 'query_faq', {'keyword': '退货'})
    assert attempts == ['退货', '退货']
    assert result.status == 'error' and 'secret' not in result.content


def test_ticket_timeout_inserts_once_even_when_worker_finishes_later(sessions):
    repo = Repository(sessions)
    cid = repo.create_conversation('guest-slow-ticket')
    attempts = []
    class SlowRepository:
        def create_ticket(self, conversation_id, description, ticket_type):
            attempts.append(conversation_id)
            time.sleep(0.08)
            return repo.create_ticket(conversation_id, description, ticket_type)
    async def scenario():
        executor = ToolExecutor(build_tools(SlowRepository(), cid, knowledge_search=FakeKnowledgeSearch()), timeout_seconds=0.01)
        result = await executor.execute(call('create_ticket', {'description': '需人工核实', 'ticket_type': '咨询'}))
        assert result.status == 'error' and '确认' in result.content
        assert result.tool_call_id == 'call-42'
        await asyncio.sleep(0.15)
    asyncio.run(scenario())
    with sessions() as session:
        rows = session.scalars(select(Ticket).where(Ticket.conversation_id == cid)).all()
        assert len(rows) == 1
    assert attempts == [cid]


@pytest.mark.parametrize('kwargs', [{'timeout_seconds': 0}, {'timeout_seconds': float('inf')},
                                   {'max_read_attempts': 0}])
def test_executor_rejects_unbounded_configuration(kwargs):
    with pytest.raises(ValueError):
        ToolExecutor({}, **kwargs)


def test_faq_timeout_keeps_result_contract_through_executor():
    class SlowSearch:
        def search(self, keyword, *, limit=5):
            time.sleep(.08)
            return []
    executor = ToolExecutor(build_tools(FakeRepository(), 42, SlowSearch()), timeout_seconds=.01)
    result = execute(executor, 'query_faq', {'keyword': '邮费'})
    payload = json.loads(result.content)
    assert result.status == 'error' and result.tool_call_id == 'call-42'
    assert set(payload) == {'keyword', 'matches', 'message'}
    assert payload['keyword'] == '邮费' and payload['matches'] == []
    assert '暂时不可用' in payload['message']
