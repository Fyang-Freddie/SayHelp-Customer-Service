"""Durable human input is scoped, atomic and idempotent."""
import importlib
import pytest
from sqlalchemy import select, func
from app.db import make_session_factory, Message
from app.repository import Repository
from app.init_ch04_db import initialize_ch04_database
from app.init_ch05_db import initialize_ch05_database
from app.init_ch06_db import initialize_ch06_database
from app.ch05_db import WorkflowTurn
from app.ch06_db import WorkflowInteraction, WorkflowResumeRequest
from test_db import database_url

@pytest.fixture
def sessions(database_url):
    initialize_ch04_database(database_url);initialize_ch05_database(database_url);initialize_ch06_database(database_url)
    factory=make_session_factory(database_url)
    yield factory
    factory.kw['bind'].dispose()

def repo(sessions):return importlib.import_module('app.ch06_repository').InteractionRepository(sessions)

def publish(r,cid):return r.publish_wait(cid,'turn','interrupt','order_selection',{'message':'请选择订单','orders':[{'item_id':'I1'}]},'checkpoint')

def test_wait_is_one_message_not_a_completed_answer(sessions):
    cid=Repository(sessions).create_conversation('customer')
    r=repo(sessions);first=publish(r,cid)
    assert publish(r,cid)==first and r.pending(cid)==first
    with sessions() as s:
        assert s.scalar(select(func.count()).select_from(Message))==1
        assert s.get(WorkflowTurn,'turn').status=='waiting'
        assert s.get(WorkflowTurn,'turn').final_message_id is None
    assert first['status']=='waiting' and first['checkpoint_id']=='checkpoint'


def test_resume_replays_same_request_and_rejects_payload_conflict(sessions):
    cid=Repository(sessions).create_conversation('customer');r=repo(sessions);publish(r,cid)
    first=r.claim_resume(cid,'turn','interrupt','request','a'*64)
    assert first['claimed'] is True and first['status']=='processing'
    duplicate=r.claim_resume(cid,'turn','interrupt','request','a'*64)
    assert duplicate['claimed'] is False and duplicate['status']=='processing'
    with pytest.raises(ValueError):r.claim_resume(cid,'turn','interrupt','request','b'*64)
    with pytest.raises(ValueError):r.claim_resume(cid,'turn','interrupt','another','a'*64)
    r.finish_resume('request',{'selected':'I1'})
    result=r.claim_resume(cid,'turn','interrupt','request','a'*64)
    assert result['status']=='complete' and result['result']=={'selected':'I1'} and not result['claimed']
    assert r.pending(cid) is None
    with pytest.raises(ValueError):r.finish_resume('request',{'selected':'I2'})


def test_wait_and_resume_cannot_cross_conversations(sessions):
    base=Repository(sessions);cid=base.create_conversation('one');other=base.create_conversation('two')
    r=repo(sessions);publish(r,cid)
    assert r.pending(other) is None
    with pytest.raises((KeyError,ValueError)):r.claim_resume(other,'turn','interrupt','request','a'*64)
    with pytest.raises((KeyError,ValueError)):r.cancel_wait(other,'interrupt')
    assert r.pending(cid)['status']=='waiting'


def test_cancel_keeps_old_card_invalid_and_no_claim_allowed(sessions):
    cid=Repository(sessions).create_conversation('customer');r=repo(sessions);publish(r,cid)
    r.cancel_wait(cid,'interrupt');r.cancel_wait(cid,'interrupt')
    assert r.pending(cid) is None
    with pytest.raises(ValueError):r.claim_resume(cid,'turn','interrupt','request','a'*64)
    with sessions() as s:assert s.get(WorkflowInteraction,'interrupt').status=='cancelled'


def test_failure_rolls_back_turn_message_and_interaction(sessions,monkeypatch):
    from app.workflow_repository import WorkflowRepository
    cid=Repository(sessions).create_conversation('customer');r=repo(sessions)
    original=WorkflowRepository._message
    def failing(*args,**kwargs):
        original(*args,**kwargs)
        raise RuntimeError('injected')
    monkeypatch.setattr(WorkflowRepository,'_message',staticmethod(failing))
    with pytest.raises(RuntimeError,match='injected'):publish(r,cid)
    with sessions() as s:
        assert s.get(WorkflowTurn,'turn') is None
        assert s.scalar(select(func.count()).select_from(Message))==0
        assert s.scalar(select(func.count()).select_from(WorkflowInteraction))==0


def test_refund_action_is_exact_eligible_item_and_idempotent(sessions):
    from app.business_data import MarkdownBusinessData
    from app.ch06_types import Eligibility
    module=importlib.import_module('app.ch06_repository')
    r=module.RefundRepository(sessions)
    cid=Repository(sessions).create_conversation('customer')
    from pathlib import Path
    item=MarkdownBusinessData(Path(__file__).resolve().parents[1]/'knowledge_db').load().items[0]
    eligibility=Eligibility(status='eligible',explanation='有政策证据',evidence_ids=['1'],missing_facts=[])
    first=r.issue_action(cid,'turn',item,eligibility,{'data':'v1','policy':'v1'})
    assert r.issue_action(cid,'turn',item,eligibility,{'data':'v1','policy':'v1'})==first
    assert first['status']=='active' and first['item']['paid_amount']==str(item.paid_amount)
    with pytest.raises(ValueError):r.issue_action(cid,'turn',item,eligibility,{'data':'v2','policy':'v1'})
    with pytest.raises(ValueError):r.issue_action(cid,'turn2',item,Eligibility(status='needs_information',explanation='未知',evidence_ids=[],missing_facts=['状态']),{'data':'v1'})


def test_two_simultaneous_resume_claims_have_one_owner(sessions):
    from concurrent.futures import ThreadPoolExecutor
    cid=Repository(sessions).create_conversation('customer');r=repo(sessions);publish(r,cid)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:r.claim_resume(cid,'turn','interrupt','request','a'*64),range(2)))
    assert sorted(result['claimed'] for result in results)==[False,True]


def test_history_card_is_public_but_checkpoint_is_private(sessions):
    from app.chat_views import public_messages
    cid=Repository(sessions).create_conversation('customer');r=repo(sessions);publish(r,cid)
    messages=public_messages(Repository(sessions).load_messages(cid),interactions=r.load_interactions(cid))
    card=messages[0]['interaction']
    assert card['interrupt_id']=='interrupt' and card['status']=='waiting'
    assert 'checkpoint_id' not in card and 'conversation_id' not in card


@pytest.mark.parametrize('kind',['customer_input','clarification'])
def test_agent_clarification_waits_use_same_durable_boundary(sessions,kind):
    cid=Repository(sessions).create_conversation('customer');r=repo(sessions)
    result=r.publish_wait(cid,'turn','interrupt',kind,{'message':'请补充信息'},'checkpoint')
    assert result['kind']==kind and result['status']=='waiting'


def test_pending_turn_messages_are_excluded_from_completed_model_history(sessions):
    from app.workflow_repository import WorkflowRepository
    cid=Repository(sessions).create_conversation('customer');r=repo(sessions);wait=publish(r,cid)
    assert wait['message_id'] in WorkflowRepository(sessions).incomplete_message_ids(cid)
