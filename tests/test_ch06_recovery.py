import asyncio
from types import SimpleNamespace
from langchain_core.messages import HumanMessage
from app.workflow_service import WorkflowService
from app.workflow_graph import open_workflow_checkpointer
from app.workflow_repository import WorkflowRepository
from app.ch06_repository import InteractionRepository
from app.repository import Repository
from test_ch06_repository import sessions
from test_db import database_url
from test_aftersales_workflow import Business,Expander,Policy,Model,emit

def graph(saver,events):
    from app.workflow_graph import build_workflow,_emit
    from app.aftersales_workflow import build_aftersales_subgraph
    from app.agent_runtime import AgentLimits
    from app.ch06_types import IntentDecision
    from app.workflow_types import Usage
    model=Model(events)
    async def classify(query):return IntentDecision(intent='退款退货',confidence=1.0),Usage()
    child=build_aftersales_subgraph(business=Business(events),expander=Expander(events),policy_retrieval=Policy(events),model=model,limits=AgentLimits(),emit=_emit)
    return build_workflow(model=model,classifier=classify,knowledge_gate=None,tools_factory=lambda cid:{},limits=AgentLimits(),log=SimpleNamespace(write=lambda *args:None),checkpointer=saver,aftersales=child)

from test_chat_api import settings


def service(sessions,compiled):
    return WorkflowService(Repository(sessions),WorkflowRepository(sessions),compiled,settings(),interactions=InteractionRepository(sessions))

async def collect(stream):return [event async for event in stream]


def test_pause_persist_resume_and_replay_without_model_repeat(sessions,tmp_path):
    async def run():
        events=[];cid=Repository(sessions).create_conversation('customer')
        async with open_workflow_checkpointer(tmp_path/'state.sqlite') as saver:
            compiled=graph(saver,events);svc=service(sessions,compiled)
            first=await collect(svc.stream_turn(cid,'我要退款'))
            assert [e.kind for e in first][-2:]==['interaction_required','paused']
            pending=await svc.get_pending_interaction(cid)
            assert pending['kind']=='customer_input'
            second=await collect(svc.resume_turn(cid,pending['turn_id'],pending['interrupt_id'],'r1',{'customer_id':'C1'}))
            assert second[-1].kind=='paused'
            pending=await svc.get_pending_interaction(cid)
            assert pending['kind']=='order_selection'
            value={'customer_id':'C1','order_id':'O1','item_id':'I1','data_digest':'data-v1'}
            third=await collect(svc.resume_turn(cid,pending['turn_id'],pending['interrupt_id'],'r2',value))
            assert third[-1].kind=='completed'
            before=list(events)
            replay=await collect(svc.resume_turn(cid,pending['turn_id'],pending['interrupt_id'],'r2',value))
            assert replay[-1].kind=='completed' and events==before
            assert await svc.get_pending_interaction(cid) is None
    asyncio.run(run())


def test_sqlite_wait_repairs_missing_mysql_display_after_restart(sessions,tmp_path,monkeypatch):
    async def run():
        events=[];cid=Repository(sessions).create_conversation('customer');path=tmp_path/'state.sqlite'
        async with open_workflow_checkpointer(path) as saver:
            svc=service(sessions,graph(saver,events));original=svc.interactions.publish_wait
            def fail(*args,**kwargs):raise RuntimeError('display failed')
            monkeypatch.setattr(svc.interactions,'publish_wait',fail)
            import pytest
            with pytest.raises(RuntimeError,match='display failed'):await collect(svc.stream_turn(cid,'我要退款'))
        async with open_workflow_checkpointer(path) as saver:
            svc=service(sessions,graph(saver,events))
            pending=await svc.get_pending_interaction(cid)
            assert pending['kind']=='customer_input' and events==['ask_customer']
            assert svc.interactions.pending(cid)['interrupt_id']==pending['interrupt_id']
    asyncio.run(run())


def test_completed_checkpoint_repairs_resume_result_without_reexecution(sessions,tmp_path,monkeypatch):
    async def run():
        import pytest
        events=[];cid=Repository(sessions).create_conversation('customer')
        async with open_workflow_checkpointer(tmp_path/'state.sqlite') as saver:
            svc=service(sessions,graph(saver,events))
            await collect(svc.stream_turn(cid,'退款'))
            p=await svc.get_pending_interaction(cid)
            await collect(svc.resume_turn(cid,p['turn_id'],p['interrupt_id'],'r1',{'customer_id':'C1'}))
            p=await svc.get_pending_interaction(cid)
            value={'customer_id':'C1','order_id':'O1','item_id':'I1','data_digest':'data-v1'}
            original=svc.interactions.finish_resume
            def fail(*args,**kwargs):raise RuntimeError('result write failed')
            monkeypatch.setattr(svc.interactions,'finish_resume',fail)
            with pytest.raises(RuntimeError,match='result write failed'):await collect(svc.resume_turn(cid,p['turn_id'],p['interrupt_id'],'r2',value))
            before=list(events)
            monkeypatch.setattr(svc.interactions,'finish_resume',original)
            replay=await collect(svc.resume_turn(cid,p['turn_id'],p['interrupt_id'],'r2',value))
            assert replay[-1].kind=='completed' and events==before
    asyncio.run(run())


def test_cancel_invalidates_sqlite_wait_and_old_card(sessions,tmp_path):
    async def run():
        import pytest
        cid=Repository(sessions).create_conversation('customer');events=[]
        async with open_workflow_checkpointer(tmp_path/'state.sqlite') as saver:
            svc=service(sessions,graph(saver,events))
            await collect(svc.stream_turn(cid,'退款'))
            pending=await svc.get_pending_interaction(cid)
            await svc.cancel_interaction(cid,pending['interrupt_id'])
            assert await svc.get_pending_interaction(cid) is None
            assert not (await svc.graph.aget_state({'configurable':{'thread_id':str(cid)}})).next
            with pytest.raises(ValueError):await collect(svc.resume_turn(cid,pending['turn_id'],pending['interrupt_id'],'r1',{'customer_id':'C1'}))
    asyncio.run(run())
