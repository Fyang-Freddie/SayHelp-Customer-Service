"""Fixed routing and genuine pre-output gate integration contracts."""
import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from app.agent_runtime import AgentLimits
from app.evaluation.calibration import ConfidencePolicy
from app.intent import IntentDecision, IntentClassificationError
from app.knowledge_types import EvidenceChunk, RankedChunk, RetrievalResult
from app.workflow_knowledge import KnowledgeGate
from app.workflow_log import WorkflowLog
from app.workflow_types import Usage


class Model:
    def __init__(self, decisions=None):
        self.decisions = iter(decisions or [AIMessage(content='private draft')])
        self.selected, self.finals = [], []

    async def select(self, messages, tools, *, max_tokens):
        self.selected.append(list(messages))
        return next(self.decisions)

    async def stream_reply(self, messages, *, max_tokens):
        self.finals.append(list(messages))
        yield AIMessageChunk(content='有依据的回答')


class Retrieval:
    def __init__(self, score=.8):
        self.score, self.calls = score, []

    def retrieve(self, query, strategy, filters):
        self.calls.append((query, strategy, filters))
        item = RankedChunk(EvidenceChunk(1, '问题', '完整原文证据', '商品', '饮水机', 'manual',
            '商品 / 型号', 'knowledge_db/product-specs.md', 1, 9, 'a'*64), 1, self.score)
        return RetrievalResult('hybrid_rerank', query, [item], [item], {})


def read_tools(attempts):
    @tool
    async def query_order(order_id: str) -> str:
        """Read order."""
        attempts.append(order_id)
        return '已发货'
    return {'query_order': query_order}


def selected(ident='read-1'):
    return AIMessage(content='hidden text', tool_calls=[{'name':'query_order','id':ident,'args':{'order_id':'1001'}}])


def build(tmp_path, intent='订单', model=None, score=.8, checkpointer=None, limits=None, classifier=None):
    api = importlib.import_module('app.workflow_graph')
    classifications, attempts, factories = [], [], []
    async def classify(question):
        classifications.append(question)
        return IntentDecision(intent=intent), Usage(input_tokens=7, output_tokens=3)
    retrieval = Retrieval(score)
    log = WorkflowLog(tmp_path / 'log')
    gate = KnowledgeGate(retrieval, ConfidencePolicy({'hybrid_rerank': .5}, {}),
        SimpleNamespace(context_token_budget=4096, response_token_reserve=512), log=log)
    def tools_factory(conversation_id):
        factories.append(conversation_id)
        return read_tools(attempts)
    model = model or Model()
    graph = api.build_workflow(model=model, classifier=classifier or classify, knowledge_gate=gate,
        tools_factory=tools_factory, limits=limits or AgentLimits(), log=log, checkpointer=checkpointer)
    return graph, model, retrieval, classifications, attempts, factories


def turn(question='  原始问题\n不改写  ', turn_id='t1'):
    return {'conversation_id':1,'turn_id':turn_id,'raw_question':question,
        'messages':[HumanMessage(content=question, id=f'{turn_id}:user')]}


async def collect(graph, data, config=None):
    trace, events, result = [], [], None
    async for namespace, mode, payload in graph.astream(data, config, stream_mode=['updates','custom','values'], subgraphs=True):
        if mode == 'updates':
            trace.extend(payload)
        elif mode == 'custom':
            events.append(payload)
        elif not namespace:
            result = payload
    return result, trace, events


@pytest.mark.parametrize('intent,route,tail,retrieves,calls', [
    ('物流','business',['agent_step','execute_calls','agent_step','stream_answer','agent','log_turn'],0,3),
    ('订单','business',['agent_step','execute_calls','agent_step','stream_answer','agent','log_turn'],0,3),
    ('商品咨询','knowledge',['retrieve','confidence_gate','agent_step','stream_answer','agent','log_turn'],1,2),
    ('退款退货','knowledge',['retrieve','confidence_gate','agent_step','stream_answer','agent','log_turn'],1,2),
    ('售后','business',['agent_step','execute_calls','agent_step','stream_answer','agent','log_turn'],0,3),
    ('投诉','complaint',['complaint','log_turn'],0,0),
    ('闲聊','chitchat',['chitchat','log_turn'],0,0),
])
def test_seven_paths_have_fixed_nodes_and_bounded_calls(tmp_path, intent, route, tail, retrieves, calls):
    model = Model([selected(), AIMessage(content='draft')]) if route == 'business' else Model()
    graph, model, retrieval, classifications, attempts, factories = build(tmp_path, intent, model)
    result, trace, events = asyncio.run(collect(graph, turn()))
    assert trace == ['resolve_reference','classify_intent','route'] + tail
    assert classifications == ['  原始问题\n不改写  ']
    assert result['resolved_question'] == '  原始问题\n不改写  '
    assert result['route'] == route and len(retrieval.calls) == retrieves
    assert result['model_calls'] == calls and len(model.selected)+len(model.finals) == calls
    assert result['tool_calls'] == (1 if route=='business' else 0)
    assert len(attempts) == result['tool_calls']
    assert all(x == 1 for x in factories) and bool(factories) == bool(calls)
    assert result['usage']['input_tokens'] >= 7
    assert result['status'] == 'done'
    if route == 'complaint':
        assert result['answer'] == '很抱歉给您带来不好的体验。您可以选择转人工或建立工单，也可以继续向我说明情况。'
        assert result['suggestions'] == [{'kind':'handoff'},{'kind':'create_ticket','description':'  原始问题\n不改写  ','ticket_type':'投诉'}]
    elif route == 'chitchat':
        assert result['answer'] == '您好，我是 SayHelp。可以帮您查询商品、订单、物流、退换货政策或售后问题。'
        assert result['usage'] == {'input_tokens':7,'output_tokens':3,'estimated':0}
    else:
        assert result['answer'] == '有依据的回答'
        assert all('hidden text' != m.content for m in result['messages'])
    logged = [json.loads(x) for x in (tmp_path/'log/events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert logged[-1]['event'] == 'turn_finished'
    assert '原始问题' not in json.dumps(logged, ensure_ascii=False)


def test_real_weak_gate_blocks_all_agent_events_and_records_original_once(tmp_path):
    graph, model, retrieval, classifications, attempts, factories = build(tmp_path, '商品咨询', score=.2)
    result, trace, events = asyncio.run(collect(graph, turn()))
    assert trace == ['resolve_reference','classify_intent','route','retrieve','confidence_gate','fallback','log_turn']
    assert len(retrieval.calls)==len(classifications)==1
    assert model.selected == model.finals == attempts == factories == []
    assert not any(e['event'] in ('model_start','token','tool_start') for e in events)
    assert result['model_calls']==result['tool_calls']==0
    assert result['stop_reason']=='weak_evidence' and result['citations']==[]
    assert '已记录' in result['answer']
    records=(tmp_path/'log/low-confidence.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(records)==1 and json.loads(records[0])['question']=='  原始问题\n不改写  '


def test_real_strong_gate_emits_ready_before_selection_and_first_token(tmp_path):
    graph, model, retrieval, *_ = build(tmp_path, '退款退货')
    result, trace, events = asyncio.run(collect(graph, turn()))
    names=[e['event'] for e in events]
    assert names.index('knowledge_ready') < names.index('model_start') < names.index('token')
    assert trace.index('confidence_gate') < trace.index('agent_step')
    assert len(retrieval.calls)==1 and result['citations'][0]['chunk_id']=='1'
    assert any(isinstance(m,SystemMessage) and '完整原文证据' in m.content for m in model.finals[0])
    assert not any(isinstance(m,SystemMessage) and '完整原文证据' in m.content for m in result['messages'])


def test_invalid_classification_is_controlled_and_keeps_paid_usage(tmp_path):
    async def invalid(question):
        raise IntentClassificationError(Usage(estimated=23))
    graph, model, retrieval, *_ = build(tmp_path, classifier=invalid)
    result, trace, events = asyncio.run(collect(graph, turn()))
    assert trace == ['resolve_reference','classify_intent','route','fallback','log_turn']
    assert result['stop_reason']=='invalid_intent' and result['usage']['estimated']==23
    assert not model.selected and not retrieval.calls


def test_model_budget_and_recursion_insurance_are_independent(tmp_path):
    model=Model([selected(str(i)) for i in range(9)])
    graph,*_ = build(tmp_path, model=model, limits=AgentLimits(model_calls=3))
    result, trace, _ = asyncio.run(collect(graph, turn()))
    assert result['model_calls']==3 and result['tool_calls']==2 and result['stop_reason']=='model_budget'
    assert len(model.finals)==1
    with pytest.raises(GraphRecursionError):
        asyncio.run(graph.ainvoke(turn(turn_id='t2'), {'recursion_limit':2}))


def test_public_budget_names_win_and_invalid_relationship_is_rejected(monkeypatch,tmp_path):
    from app.config import Settings
    monkeypatch.chdir(tmp_path)
    for key,value in {'CHAT_BASE_URL':'https://example.test/v1','CHAT_MODEL':'test','CHAT_API_KEY':'offline',
        'DATABASE_URL':'sqlite://','AGENT_MAX_MODEL_CALLS':'3','AGENT_MODEL_CALLS':'4',
        'AGENT_MAX_TOOL_CALLS':'2','AGENT_TOOL_CALLS':'3','AGENT_TURN_TOKEN_BUDGET':'9000',
        'AGENT_TURN_TOKENS':'8000'}.items():
        monkeypatch.setenv(key,value)
    settings=Settings.from_env()
    assert (settings.agent_model_calls,settings.agent_tool_calls,settings.agent_turn_tokens)==(3,2,9000)
    assert settings.workflow_checkpoint_path=='.runtime/ch05/checkpoints.sqlite'
    monkeypatch.setenv('AGENT_TURN_TOKEN_BUDGET','512')
    with pytest.raises(ValueError,match='AGENT_TURN_TOKEN_BUDGET'):
        Settings.from_env()
    monkeypatch.setenv('AGENT_TURN_TOKEN_BUDGET','9000')
    monkeypatch.setenv('AGENT_MAX_MODEL_CALLS','7')
    with pytest.raises(ValueError,match='AGENT_MAX_MODEL_CALLS'):
        Settings.from_env()

@pytest.mark.parametrize('tool_budget,break_tool,reason,attempt_count', [
    (1,False,'tool_budget',1), (6,True,'repeated_tool_failure',2),
])
def test_graph_tool_budget_and_failed_read_stop_before_another_selection(tmp_path,tool_budget,break_tool,reason,attempt_count):
    api=importlib.import_module('app.workflow_graph')
    attempts=[]
    @tool
    async def query_order(order_id: str) -> str:
        """Read order with a deterministic offline failure."""
        attempts.append(order_id)
        if break_tool:
            raise RuntimeError('offline read failed')
        return '已发货'
    async def classify(question):
        return IntentDecision(intent='订单'),Usage(input_tokens=7,output_tokens=3)
    model=Model([selected(str(i)) for i in range(9)])
    graph=api.build_workflow(model=model,classifier=classify,knowledge_gate=None,
        tools_factory=lambda cid:{'query_order':query_order},limits=AgentLimits(tool_calls=tool_budget),
        log=WorkflowLog(tmp_path/'log'),checkpointer=None)
    result,trace,events=asyncio.run(collect(graph,turn()))
    assert len(attempts)==result['tool_calls']==attempt_count
    assert len(model.selected)==len(model.finals)==1 and result['model_calls']==2
    assert result['stop_reason']==reason and result['answer']=='有依据的回答'
    assert trace==['resolve_reference','classify_intent','route','agent_step','execute_calls','stream_answer','agent','log_turn']
