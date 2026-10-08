"""Offline regressions for chapter 5 whole-branch review findings."""
import asyncio
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from app.agent_runtime import AgentLimits, _suggestions, agent_step, execute_calls, stream_answer
from app.config import Settings
from app.intent import IntentDecision
from app.workflow_graph import build_workflow
from app.workflow_log import WorkflowLog
from app.workflow_types import Usage
from test_workflow_graph import Model, collect, turn


def knowledge(source, size=20):
    return {'useful': True, 'matches': [{'id': source, 'n': 1, 'answer': 'E' * size}],
            'citations': [{'chunk_id': source, 'n': 1, 'source_digest': source, 'source_file': source}]}


@pytest.mark.parametrize('initial', [False, True])
def test_tool_citations_share_stable_turn_registry(tmp_path, initial):
    class Gate:
        async def prepare(self, question, filters=None):
            value = knowledge('initial')
            return {'evidence': value['matches'], 'citations': value['citations'], 'confidence': {'sufficient': True}}
    @tool
    async def query_faq(keyword: str) -> dict:
        """Read admitted FAQ evidence."""
        return knowledge(keyword)
    async def classify(question):
        return IntentDecision(intent='商品咨询' if initial else '订单'), Usage()
    decisions = [AIMessage(content='', tool_calls=[{'name': 'query_faq', 'id': f'c{i}', 'args': {'keyword': key}}])
                 for i, key in enumerate(['new', 'another', 'new'])] + [AIMessage(content='')]
    model = Model(decisions)
    graph = build_workflow(model=model, classifier=classify, knowledge_gate=Gate(),
        tools_factory=lambda cid: {'query_faq': query_faq}, limits=AgentLimits(), log=WorkflowLog(tmp_path), checkpointer=None)
    state, _, _ = asyncio.run(collect(graph, turn()))
    expected = (['initial'] if initial else []) + ['new', 'another']
    assert [c['chunk_id'] for c in state['citations']] == expected
    assert [c['n'] for c in state['citations']] == list(range(1, len(expected)+1))
    results = [json.loads(m.content) for m in model.finals[0] if isinstance(m, ToolMessage)]
    assert [r['matches'][0]['n'] for r in results] == [1+initial, 2+initial, 1+initial]
    assert [r['citations'][0]['n'] for r in results] == [1+initial, 2+initial, 1+initial]


def test_request_limit_counts_schemas_and_final_evidence():
    async def emit(*args): pass
    async def run():
        model = Model()
        limits = AgentLimits(input_tokens=100)
        state = {'messages': [HumanMessage(content='X'*450)], 'usage': {}, 'model_calls': 0}
        selected = await agent_step(state, model=model, tools={}, limits=limits, emit=emit)
        final = await stream_answer(state, model=model, limits=limits, emit=emit)
        assert selected['stop_reason'] == final['stop_reason'] == 'context_budget'
        assert not model.selected and not model.finals
        state['messages'] = [HumanMessage(content='short')]
        selected = await agent_step(state, model=model, tools={}, limits=limits, emit=emit)
        assert selected['stop_reason'] == 'context_budget' and not model.selected
    asyncio.run(run())


def test_context_rejected_evidence_not_promoted_and_batch_stays_paired():
    @tool
    async def query_faq(keyword: str) -> dict:
        """Read whole evidence."""
        return knowledge('oversized', 18000)
    async def emit(*args): pass
    async def run():
        state = {'messages': [HumanMessage(content='question'), AIMessage(content='', tool_calls=[
            {'name':'query_faq','id':'one','args':{'keyword':'q'}},
            {'name':'query_faq','id':'two','args':{'keyword':'q'}}])], 'usage':{}, 'citations':[]}
        result = await execute_calls(state, tools={'query_faq':query_faq}, limits=AgentLimits(input_tokens=2000), emit=emit)
        assert result.get('citations', []) == []
        assert [m.tool_call_id for m in result['messages']] == ['one','two']
        assert all(m.status == 'error' and 'EEEE' not in m.content for m in result['messages'])
        assert result['tool_calls'] == 1
    asyncio.run(run())


def test_classification_budget_preflight_makes_zero_calls(tmp_path):
    called = []
    async def classify(question):
        called.append(question)
        return IntentDecision(intent='订单'), Usage()
    graph = build_workflow(model=Model(), classifier=classify, knowledge_gate=None,
        tools_factory=lambda cid:{}, limits=AgentLimits(turn_tokens=513), log=WorkflowLog(tmp_path), checkpointer=None)
    result, _, _ = asyncio.run(collect(graph, turn('Q'*3000)))
    assert not called and result['stop_reason'] == 'token_budget' and result['usage'] == Usage().to_dict()


@pytest.mark.parametrize('phase', ['select','final'])
def test_chapter5_http_503_has_one_physical_attempt(monkeypatch, phase):
    import app.model_service as module
    requests = []
    async def handler(request):
        requests.append(request)
        return httpx.Response(503, json={'error':{'message':'offline','type':'server_error'}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
            with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)), trust_env=False) as sync:
                monkeypatch.setattr(module, 'ChatOpenAI', lambda **kwargs: ChatOpenAI(**kwargs, http_async_client=client, http_client=sync, http_socket_options=()))
                service = module.ModelService(Settings('https://example.test/v1','offline','offline'))
                assert service._model.max_retries == 1
                with pytest.raises(Exception):
                    if phase == 'select':
                        await service.select([HumanMessage(content='offline')], [], max_tokens=10)
                    else:
                        async for _ in service.stream_reply([HumanMessage(content='offline')], max_tokens=10): pass
    asyncio.run(run())
    assert len(requests) == 1


@pytest.mark.parametrize('kind', ['handoff','create_ticket'])
def test_duplicate_action_kinds_are_rejected(kind):
    action = {'kind':kind}
    if kind == 'create_ticket': action.update(description='help', ticket_type='投诉')
    with pytest.raises(ValueError): _suggestions({'actions':[action,dict(action)]})


def test_interleaved_graph_logs_are_correlated_safe_and_include_failures(tmp_path):
    class Failure(Model):
        async def select(self, messages, tools, *, max_tokens):
            await asyncio.sleep(0)
            if 'fail' in str(messages): raise RuntimeError('secret raw result')
            return AIMessage(content='', tool_calls=[{'id':'call-safe','name':'query_order','args':{'order_id':'1'}}])
    @tool
    async def query_order(order_id: str) -> str:
        """Sensitive business result is excluded from logs."""
        return 'private customer result'
    async def classify(question):
        await asyncio.sleep(0)
        return IntentDecision(intent='订单'), Usage(input_tokens=4,output_tokens=2)
    graph = build_workflow(model=Failure(), classifier=classify, knowledge_gate=None,
        tools_factory=lambda cid:{'query_order':query_order}, limits=AgentLimits(model_calls=2), log=WorkflowLog(tmp_path), checkpointer=None)
    async def run():
        a = turn('ok','a'); a['conversation_id']=1
        b = turn('fail','b'); b['conversation_id']=2
        return await asyncio.gather(collect(graph,a), collect(graph,b), return_exceptions=True)
    result = asyncio.run(run())
    assert isinstance(result[1], RuntimeError)
    text = (tmp_path/'events.jsonl').read_text(encoding='utf-8')
    assert 'secret raw result' not in text and 'private customer result' not in text
    rows = [json.loads(line) for line in text.splitlines()]
    for row in rows:
        p = row['payload']; assert (p['conversation_id'],p['turn_id']) in [(1,'a'),(2,'b')]
    assert any(r['event']=='tool_end' and r['payload']['tool_call_id']=='call-safe' for r in rows)
    assert any(r['event']=='node_end' and r['payload']['duration_ms']>=0 for r in rows)
    assert any(r['event']=='turn_finished' and r['payload']['status']=='failed' and r['payload']['turn_id']=='b'
               and r['payload']['model_calls']==1 and r['payload']['usage']['estimated']>0 for r in rows)
    assert any(r['event']=='turn_finished' and r['payload']['usage']['input_tokens']>=4 for r in rows)


def test_weak_question_records_identity_and_gate_decision(tmp_path):
    from test_workflow_graph import build
    graph, *_ = build(tmp_path, intent='商品咨询', score=.1)
    asyncio.run(collect(graph, turn('authored weak question','weak-turn')))
    weak = json.loads((tmp_path/'log'/'low-confidence.jsonl').read_text(encoding='utf-8'))
    assert weak['conversation_id'] == 1 and weak['turn_id'] == 'weak-turn'
    rows = [json.loads(line) for line in (tmp_path/'log'/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert any(r['event']=='gate_decision' and r['payload']['sufficient'] is False for r in rows)


def test_cancelled_graph_writes_correlated_terminal(tmp_path):
    entered = None
    class Blocked(Model):
        async def select(self, *args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
    async def classify(question): return IntentDecision(intent='订单'), Usage()
    graph = build_workflow(model=Blocked(), classifier=classify, knowledge_gate=None,
        tools_factory=lambda cid:{}, limits=AgentLimits(), log=WorkflowLog(tmp_path), checkpointer=None)
    async def cancel():
        nonlocal entered
        entered = asyncio.Event()
        task = asyncio.create_task(collect(graph, turn('cancel','cancel-turn')))
        await asyncio.wait_for(entered.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
    asyncio.run(cancel())
    rows = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert any(r['event']=='turn_finished' and r['payload']['status']=='cancelled'
               and r['payload']['turn_id']=='cancel-turn' for r in rows)
    assert any(r['event']=='model_error' and r['payload']['usage']['estimated']>0 for r in rows)


from test_db import database_url
from test_workflow_repository import sessions


@pytest.mark.parametrize('initial', [False, True])
def test_tool_registry_reaches_service_sse_and_mysql_history(sessions, tmp_path, initial):
    from app.repository import Repository
    from app.workflow_repository import WorkflowRepository
    from app.workflow_service import WorkflowService
    from app.workflow_graph import open_workflow_checkpointer
    cid = Repository(sessions).create_conversation('disposable citation regression')
    class Gate:
        async def prepare(self, *args):
            data = knowledge('initial')
            return {'evidence':data['matches'], 'citations':data['citations'], 'confidence':{'sufficient':True}}
    @tool
    async def query_product(product_query: str) -> dict:
        """Read independently gated product source."""
        return knowledge('new')
    async def classify(question): return IntentDecision(intent='商品咨询' if initial else '订单'), Usage()
    class Answer(Model):
        async def stream_reply(self, messages, *, max_tokens):
            yield AIMessageChunk(content=f'Answer [{1+initial}]')
    model = Answer([AIMessage(content='',tool_calls=[{'name':'query_product','id':'read','args':{'product_query':'new'}}]), AIMessage(content='')])
    async def run():
        async with open_workflow_checkpointer(tmp_path/'citation.sqlite') as saver:
            log = WorkflowLog(tmp_path/'log')
            graph = build_workflow(model=model,classifier=classify,knowledge_gate=Gate(),
                tools_factory=lambda cid:{'query_product':query_product},limits=AgentLimits(),log=log,checkpointer=saver)
            service = WorkflowService(Repository(sessions),WorkflowRepository(sessions),graph,Settings('https://example.test','offline','offline'),log=log)
            return [event async for event in service.stream_turn(cid,'authored question')]
    events = asyncio.run(run())
    citations = next(e.data['citations'] for e in events if e.kind=='citations')
    assert [c['chunk_id'] for c in citations] == (['initial'] if initial else [])+['new']
    rows = Repository(sessions).load_messages(cid)
    assert rows[-1].citations == citations and rows[-1].content == f'Answer [{1+initial}]'
    assert events[-1].kind=='completed'
    logrows=[json.loads(line) for line in (tmp_path/'log'/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert logrows[-1]['event']=='turn_terminal' and logrows[-1]['payload']['status']=='complete'
