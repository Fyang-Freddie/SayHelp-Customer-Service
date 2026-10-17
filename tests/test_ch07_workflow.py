import asyncio
import pytest
from dataclasses import replace
from types import SimpleNamespace
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.context_types import ContextSnapshot
from app.config import Settings
from test_workflow_repository import sessions
from test_db import database_url


@pytest.mark.parametrize('question,expected',[
    ('最开始那个订单后来怎么说？请复述订单号和我当时提出的诉求。',True),
    ('回顾一下之前聊的内容',True),('还记得我最初问过哪款商品吗？',True),
    ('之前那个订单现在帮我退款',False),('复述后请帮我提交退款申请',False),
    ('帮我查一下前面那个订单',False),('我想投诉',False)])
def test_recap_is_explicit_and_never_replaces_business_requests(question,expected):
    from app.context_recap import is_recap_request
    assert is_recap_request(question) is expected


def test_read_only_recap_uses_summary_without_classifier_or_tools(tmp_path,sessions):
    from app.workflow_graph import build_workflow,open_workflow_checkpointer
    from app.context_runtime import ContextRuntime
    from app.context_log import ContextLog
    from app.workflow_log import WorkflowLog
    from app.agent_runtime import AgentLimits
    from langchain_core.messages import AIMessageChunk
    class Repo:
        def load_messages(self,cid):return []
    class ContextRepo:
        def load(self,cid):return ContextSnapshot(cid,summaries=({'ordinal':1,'content':'用户订单ABC123缺件，要求核实，未处理。'},))
    class Model:
        async def select(self,*args,**kwargs):raise AssertionError('recap must never select tools')
        async def stream_reply(self,messages,**kwargs):raise AssertionError('recap must not stream unvalidated text')
        async def recap_reply(self,messages,**kwargs):
            assert messages[0].type=='system'
            assert 'ABC123' in messages[-1].content and 'JSON' in messages[-1].content
            return AIMessage(content='订单ABC123缺件，您要求核实，仍未处理。')
    async def forbidden(*args,**kwargs):raise AssertionError('recap must not classify, resolve or execute business')
    settings=Settings('x','x','x',context_token_budget=18000,response_token_reserve=2000,layered_context=True)
    runtime=ContextRuntime(Repo(),None,ContextRepo(),None,settings,ContextLog(tmp_path/'app.log'))
    graph=build_workflow(model=Model(),classifier=forbidden,knowledge_gate=None,
        tools_factory=lambda cid: (_ for _ in ()).throw(AssertionError('tools requested')),
        limits=AgentLimits(input_tokens=16000,response_tokens=2000,turn_tokens=100000),
        log=WorkflowLog(tmp_path/'workflow'),checkpointer=None,resolver=SimpleNamespace(resolve=forbidden),context_runtime=runtime)
    async def run():
        result=await graph.ainvoke({'conversation_id':1,'turn_id':'recap','raw_question':'最开始那个订单后来怎么说？',
            'messages':[HumanMessage('最开始那个订单后来怎么说？',id='3')]})
        assert result['answer'].startswith('订单ABC123') and result['tool_calls']==0 and result['jev_calls']==0
        assert result['model_calls']==1
        import json
        logs=[json.loads(line) for line in (tmp_path/'app.log').read_text(encoding='utf8').splitlines()]
        assert [row['event'] for row in logs]==['history_ctx','model_ctx']
        assert logs[-1]['tools']==[]
        from app.repository import Repository as DatabaseRepository
        from app.workflow_repository import WorkflowRepository
        from app.workflow_service import WorkflowService
        cid=DatabaseRepository(sessions).create_conversation('recap-test')
        runtime.repository=DatabaseRepository(sessions)
        async with open_workflow_checkpointer(tmp_path/'recap.sqlite') as saver:
            persisted=build_workflow(model=Model(),classifier=forbidden,knowledge_gate=None,
                tools_factory=lambda cid: (_ for _ in ()).throw(AssertionError('tools requested')),
                limits=AgentLimits(input_tokens=16000,response_tokens=2000,turn_tokens=100000),
                log=WorkflowLog(tmp_path/'workflow'),checkpointer=saver,
                resolver=SimpleNamespace(resolve=forbidden),context_runtime=runtime)
            service=WorkflowService(DatabaseRepository(sessions),WorkflowRepository(sessions),persisted,
                settings,context_runtime=runtime)
            events=[e async for e in service.stream_turn(cid,'最开始那个订单后来怎么说？')]
            assert any(e.kind=='completed' for e in events)
            assert [row.role for row in service.repository.load_messages(cid)]==['user','assistant']
            class Flaky(Model):
                calls=0
                async def recap_reply(self,messages,**kwargs):
                    self.calls+=1
                    if self.calls==1:raise ConnectionError('transient recap transport')
                    return await super().recap_reply(messages,**kwargs)
            flaky=Flaky()
            recovered=build_workflow(model=flaky,classifier=forbidden,knowledge_gate=None,
                tools_factory=lambda cid: (_ for _ in ()).throw(AssertionError('tools requested')),
                limits=AgentLimits(input_tokens=16000,response_tokens=2000,turn_tokens=100000),
                log=WorkflowLog(tmp_path/'workflow'),checkpointer=saver,
                resolver=SimpleNamespace(resolve=forbidden),context_runtime=runtime)
            second=WorkflowService(DatabaseRepository(sessions),WorkflowRepository(sessions),recovered,
                settings,context_runtime=runtime)
            next_cid=DatabaseRepository(sessions).create_conversation('recap-retry')
            events=[e async for e in second.stream_turn(next_cid,'最开始那个订单后来怎么说？')]
            assert flaky.calls==2 and any(e.kind=='completed' for e in events)
    asyncio.run(run())


def test_background_completion_does_not_mutate_running_state(tmp_path):
    from app.context_runtime import ContextRuntime
    from app.context_log import ContextLog
    class Repository:
        def load_messages(self,cid):return []
    class ContextRepo:
        def load(self,cid):return ContextSnapshot(cid)
    async def run():
        config=Settings(chat_base_url='x',chat_model='x',chat_api_key='x',context_token_budget=18000,response_token_reserve=2000)
        runtime=ContextRuntime(Repository(),None,ContextRepo(),None,config,ContextLog(tmp_path/'app.log'))
        raw=[HumanMessage('当前',id='1')]
        state={'messages':raw,'conversation_id':1,'turn_id':'turn','raw_question':'当前'}
        prepared=await runtime.model_state(state,[],phase='select')
        assert state['messages']==raw and len(state['messages'])==1
        assert prepared['messages'][0].type=='system'
        assert prepared['messages'] is not raw
        assert not (tmp_path/'app.log').exists()
        from app.agent_runtime import agent_step, AgentLimits
        class Model:
            async def select(self,messages,tools,**kwargs):
                return AIMessage('答复')
        async def emit(*args): pass
        await agent_step(prepared,model=Model(),tools={},limits=AgentLimits(input_tokens=1),emit=emit)
        assert not (tmp_path/'app.log').exists()
        await agent_step(prepared,model=Model(),tools={},limits=AgentLimits(input_tokens=18000,turn_tokens=50000),emit=emit)
        import json
        logged=json.loads((tmp_path/'app.log').read_text(encoding='utf8'))
        assert logged['event']=='model_ctx'
        assert logged['messages']==[m.model_dump(mode='json') for m in prepared['messages']]
        from app.history import InputBudgetExceeded
        import pytest
        oversized={**state,'messages':[HumanMessage('长'*2001)],'turn_id':'oversized'}
        with pytest.raises(InputBudgetExceeded):
            await runtime.history_view(oversized)
    asyncio.run(run())


def test_tool_batch_peak_is_enforced():
    from app.agent_runtime import AgentLimits,execute_calls
    from langchain_core.tools import tool
    @tool
    def large() -> str:
        """Return a large result."""
        return '大'*500
    async def run():
        emitted=[]
        async def emit(event,payload):emitted.append(event)
        state={'messages':[HumanMessage('问题'),AIMessage('',tool_calls=[dict(name='query_order',args={},id='a'),dict(name='query_order',args={},id='b')])],
               'usage':{},'tool_calls':0}
        limits=AgentLimits(turn_tokens=20000,input_tokens=10000,tool_result_tokens=200)
        result=await execute_calls(state,tools={'query_order':large},limits=limits,emit=emit)
        assert len(result['messages'])==2
        assert all(m.content!='大'*500 for m in result['messages'])
        assert result['stop_reason']=='tool_result_budget'
    asyncio.run(run())


def test_tool_selection_rejects_fallback_batch_larger_than_x():
    from app.agent_runtime import AgentLimits,agent_step,execute_calls
    async def run():
        calls=[dict(name='query_order',args={},id='a'*100),dict(name='query_order',args={},id='b'*100)]
        class Model:
            async def select(self,*args,**kwargs):return AIMessage('',tool_calls=calls)
        async def emit(*args):pass
        limits=AgentLimits(turn_tokens=100000,input_tokens=50000,tool_result_tokens=20)
        prior=[HumanMessage('查询')]
        selected=await agent_step({'messages':prior,'usage':{}},model=Model(),tools={},limits=limits,emit=emit)
        assert selected['status']=='answer' and selected['stop_reason']=='tool_result_budget'
        assert not selected.get('messages')
        direct=await execute_calls({'messages':[prior[0],AIMessage('',tool_calls=calls)]},
            tools={},limits=limits,emit=emit)
        assert direct['status']=='stopped' and not direct.get('messages')
    asyncio.run(run())


def test_react_steps_are_separate_from_query_resolution_calls():
    from app.agent_runtime import AgentLimits,agent_step
    async def run():
        class Model:
            calls=0
            async def select(self,*args,**kwargs):
                self.calls+=1
                return AIMessage('内部草稿')
        model=Model()
        async def emit(*args):pass
        state={'messages':[HumanMessage('问题')],'model_calls':1,'agent_steps':0,'usage':{}}
        limits=AgentLimits(model_calls=5,max_agent_steps=3,input_tokens=18000,turn_tokens=100000)
        for i in range(3):
            state.update(await agent_step(state,model=model,tools={},limits=limits,emit=emit))
            assert state['agent_steps']==i+1
        stopped=await agent_step(state,model=model,tools={},limits=limits,emit=emit)
        assert model.calls==3 and stopped['stop_reason']=='model_budget'
        assert state['model_calls']==4
    asyncio.run(run())


def test_default_window_initialization_is_not_logged_as_downgrade(tmp_path):
    from app.context_runtime import ContextRuntime
    from app.context_log import ContextLog
    class Repository:
        def load_messages(self,cid):
            return [SimpleNamespace(id=1,role='user',content='订单ABC123',tool_calls=None),
                    SimpleNamespace(id=2,role='assistant',content='待核实',tool_calls=None)]
    class ContextRepo:
        def load(self,cid):return ContextSnapshot(cid)
        def advance_layer1(self,*args):return True
    async def run():
        settings=Settings(chat_base_url='x',chat_model='x',chat_api_key='x',context_token_budget=65536,response_token_reserve=2000)
        runtime=ContextRuntime(Repository(),None,ContextRepo(),None,settings,ContextLog(tmp_path/'app.log'))
        view=await runtime.history_view({'conversation_id':1,'turn_id':'new','raw_question':'然后呢',
            'messages':[HumanMessage('然后呢',id='3')]})
        assert len(view.layer1)==2 and not view.layer2
        assert '层1 降级' not in (tmp_path/'app.log').read_text(encoding='utf8')
    asyncio.run(run())


def test_summary_completion_during_boundary_cas_uses_committed_snapshot(tmp_path,monkeypatch):
    from app.context_runtime import ContextRuntime
    from app.context_log import ContextLog
    from app.context_budget import calculate_budget
    from app import context_runtime as runtime_module
    budget=replace(calculate_budget(window=18000,prefix=0,evidence=0,summary=0,output=1,
        user=1,steps=1,selection=1,tool_results=1,protocol=32,target_turns=8,steady=1024),
        history=1400,layer1=980,layer2=420)
    monkeypatch.setattr(runtime_module,'settings_budget',lambda *a,**k:budget)
    rows=[]
    for n in range(4):
        rows.extend([SimpleNamespace(id=n*2+1,role='user',content='问'*600,tool_calls=None),
                     SimpleNamespace(id=n*2+2,role='assistant',content='答'*600,tool_calls=None)])
    latest=ContextSnapshot(1,version=2,summary_upto_msg_id=2,layer1_from_msg_id=7,
        summaries=({'ordinal':1,'from_msg_id':1,'upto_msg_id':2,'content':'第一轮要核实订单。'},))
    class Repo:
        def load_messages(self,cid):return rows
    class ContextRepo:
        loads=0
        def load(self,cid):
            self.loads+=1
            return ContextSnapshot(cid) if self.loads==1 else latest
        def advance_layer1(self,*args):return False
        def claim(self,cid,start,end,*args,**kwargs):
            assert start==3 and end==6
            return None
    class Worker:
        def submit(self,*args):raise AssertionError('no claim')
    async def run():
        runtime=ContextRuntime(Repo(),None,ContextRepo(),Worker(),
            Settings('x','x','x',context_token_budget=18000,response_token_reserve=2000),
            ContextLog(tmp_path/'app.log'))
        view=await runtime.history_view({'conversation_id':1,'turn_id':'current','raw_question':'然后呢',
            'messages':[HumanMessage('然后呢',id='9')]})
        assert view.snapshot==latest
        assert '第一轮要核实订单' in view.summary_text
        assert view.layer1_from_msg_id==7
    asyncio.run(run())


def test_summary_completion_between_view_and_claim_retries_uncovered_suffix(tmp_path,monkeypatch):
    from app.context_runtime import ContextRuntime
    from app.context_log import ContextLog
    from app.context_budget import calculate_budget
    from app import context_runtime as runtime_module
    budget=replace(calculate_budget(window=18000,prefix=0,evidence=0,summary=0,output=1,
        user=1,steps=1,selection=1,tool_results=1,protocol=32,target_turns=8,steady=1024),
        history=1400,layer1=980,layer2=420)
    monkeypatch.setattr(runtime_module,'settings_budget',lambda *a,**k:budget)
    rows=[]
    for n in range(4):
        rows.extend([SimpleNamespace(id=n*2+1,role='user',content='问'*600,tool_calls=None),
                     SimpleNamespace(id=n*2+2,role='assistant',content='答'*600,tool_calls=None)])
    latest=ContextSnapshot(1,version=2,summary_upto_msg_id=2,layer1_from_msg_id=7,
        summaries=({'ordinal':1,'from_msg_id':1,'upto_msg_id':2,'content':'第一轮要核实订单。'},))
    class Repo:
        def load_messages(self,cid):return rows
    class ContextRepo:
        loads=0;claims=[]
        def load(self,cid):
            self.loads+=1
            return ContextSnapshot(cid) if self.loads==1 else latest
        def advance_layer1(self,*args):return True
        def claim(self,cid,start,end,*args,**kwargs):
            self.claims.append((start,end))
            if len(self.claims)==1:raise ValueError('Summary would skip uncovered history')
            assert start==3 and end==6
            return None
    class Worker:
        def submit(self,*args):raise AssertionError('no claim')
    async def run():
        context=ContextRepo()
        runtime=ContextRuntime(Repo(),None,context,Worker(),
            Settings('x','x','x',context_token_budget=18000,response_token_reserve=2000),
            ContextLog(tmp_path/'app.log'))
        view=await runtime.history_view({'conversation_id':1,'turn_id':'current','raw_question':'然后呢',
            'messages':[HumanMessage('然后呢',id='9')]})
        assert context.claims==[(1,6),(3,6)]
        assert view.layer1_from_msg_id==7
    asyncio.run(run())


def test_recap_retry_charges_failed_attempt_and_respects_turn_limit(tmp_path):
    from app.workflow_graph import build_workflow
    from app.context_runtime import ContextRuntime
    from app.context_log import ContextLog
    from app.workflow_log import WorkflowLog
    from app.agent_runtime import AgentLimits
    from app.context_recap import recap_messages
    from app.context_budget import estimate_tokens
    class Repo:
        def load_messages(self,cid):return []
    class ContextRepo:
        def load(self,cid):return ContextSnapshot(cid)
    class Model:
        calls=0
        async def recap_reply(self,*args,**kwargs):
            self.calls+=1
            if self.calls==1:raise ValueError('invalid JSON')
            return AIMessage('此前提到订单ABC123',usage_metadata={'input_tokens':100,'output_tokens':20})
    async def forbidden(*a,**k):raise AssertionError('recap should not use business model')
    async def run():
        settings=Settings('x','x','x',context_token_budget=18000,response_token_reserve=512,layered_context=True)
        runtime=ContextRuntime(Repo(),None,ContextRepo(),None,settings,ContextLog(tmp_path/'app.log'))
        original={'conversation_id':1,'turn_id':'recap-charge','raw_question':'回顾一下之前聊的订单',
                  'messages':[HumanMessage('回顾一下之前聊的订单',id='1')]}
        prepared=await runtime.model_state(original,[],phase='recap')
        input_tokens=estimate_tokens(recap_messages(prepared['messages']))
        model=Model()
        limits=AgentLimits(input_tokens=18000,response_tokens=512,turn_tokens=input_tokens+513)
        graph=build_workflow(model=model,classifier=forbidden,knowledge_gate=None,
            tools_factory=lambda cid: (_ for _ in ()).throw(AssertionError('tools')),
            limits=limits,log=WorkflowLog(tmp_path/'workflow'),checkpointer=None,
            resolver=SimpleNamespace(resolve=forbidden),context_runtime=runtime)
        result=await graph.ainvoke(original)
        assert model.calls==1
        assert result['stop_reason']=='context_budget'
        assert result['model_calls']==1
        assert result['usage']['estimated']>=input_tokens+512
    asyncio.run(run())
