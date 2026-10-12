import asyncio,importlib
from concurrent.futures import ThreadPoolExecutor
from app.ch06_repository import RefundRepository
from app.repository import Repository
from app.ch06_types import Eligibility
from app.agent_runtime import AgentLimits
from app.workflow_types import Usage
from test_ch06_repository import sessions
from test_db import database_url
from test_aftersales_workflow import ITEM,Business

class Data(Business):
    def refund_reasons(self):return [{'code':'no_longer_needed','label':'不想要了'},{'code':'quality_issue','label':'质量问题'}]
class Model:
    def __init__(self):self.calls=0;self.status='eligible'
    async def judge_eligibility(self,*args,**kwargs):
        self.calls+=1
        return Eligibility(status=self.status,explanation='复核结论',evidence_ids=['1'],missing_facts=['检测报告'] if self.status=='needs_information' else []),Usage(5,5)

def setup(sessions):
    repo=RefundRepository(sessions);model=Model();data=Data([])
    cls=importlib.import_module('app.refund_service').RefundService
    service=cls(repo,data,model,AgentLimits(),lambda:{'policy':'v1'})
    cid=Repository(sessions).create_conversation('customer')
    eligible=Eligibility(status='eligible',explanation='可申请',evidence_ids=['1'],missing_facts=[])
    action=service.prepare_action(cid,'turn',ITEM,eligible,[{'id':1,'answer':'条款'}],query='未拆封耳机退货',model_calls=3,usage={})
    return service,repo,model,cid,action

def test_refund_request_uses_server_amount_and_replays_same_number(sessions):
    service,repo,model,cid,action=setup(sessions)
    assert action['item']['paid_amount']=='99.00' and len(action['reasons'])==2
    first=asyncio.run(service.submit(cid,action['action_id'],'no_longer_needed','r1'))
    assert first['status']=='pending' and first['amount']=='99.00'
    assert asyncio.run(service.submit(cid,action['action_id'],'no_longer_needed','r1'))==first
    assert model.calls==1
    import pytest
    with pytest.raises(ValueError):asyncio.run(service.submit(cid,action['action_id'],'quality_issue','r1'))

def test_changed_reason_needs_information_does_not_create_request(sessions):
    service,repo,model,cid,action=setup(sessions);model.status='needs_information'
    import pytest
    with pytest.raises(ValueError,match='reassessment'):asyncio.run(service.submit(cid,action['action_id'],'quality_issue','r1'))
    from sqlalchemy import select,func
    from app.ch06_db import RefundRequest
    with sessions() as s:assert s.scalar(select(func.count()).select_from(RefundRequest))==0

def test_invalid_reason_and_stale_policy_do_not_call_model(sessions):
    service,repo,model,cid,action=setup(sessions)
    import pytest
    with pytest.raises(ValueError):asyncio.run(service.submit(cid,action['action_id'],'invented','r1'))
    service.fingerprints=lambda:{'policy':'v2'}
    with pytest.raises(ValueError):asyncio.run(service.submit(cid,action['action_id'],'no_longer_needed','r2'))
    assert model.calls==0


def test_last_budgeted_review_can_retry_database_write_without_model(sessions,monkeypatch):
    service,repo,model,cid,action=setup(sessions)
    from app.ch06_db import RefundAction
    with sessions.begin() as s:
        row=s.get(RefundAction,action['action_id']);value=dict(row.eligibility);value['context']={**value['context'],'model_calls':5};row.eligibility=value
    original=repo.create_request
    def fail(*args,**kwargs):raise RuntimeError('write failed')
    monkeypatch.setattr(repo,'create_request',fail)
    import pytest
    with pytest.raises(RuntimeError):asyncio.run(service.submit(cid,action['action_id'],'no_longer_needed','r1'))
    monkeypatch.setattr(repo,'create_request',original)
    result=asyncio.run(service.submit(cid,action['action_id'],'no_longer_needed','r1'))
    assert result['status']=='pending' and model.calls==1


def test_cross_conversation_concurrent_requests_create_one_application(sessions):
    service,repo,model,cid,action=setup(sessions)
    other=Repository(sessions).create_conversation('other')
    eligible=Eligibility(status='eligible',explanation='可申请',evidence_ids=['1'],missing_facts=[])
    second=service.prepare_action(other,'turn2',ITEM,eligible,[{'id':1,'answer':'条款'}],query='退货',model_calls=3)
    def submit(args):
        conversation,form,key=args
        try:return asyncio.run(service.submit(conversation,form['action_id'],'no_longer_needed',key))
        except ValueError:return None
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(submit,[(cid,action,'r1'),(other,second,'r2')]))
    assert sum(result is not None for result in results)==1
    from sqlalchemy import select,func
    from app.ch06_db import RefundRequest
    with sessions() as s:assert s.scalar(select(func.count()).select_from(RefundRequest))==1


def test_cancelled_action_never_reviews_or_submits(sessions):
    service,repo,model,cid,action=setup(sessions);repo.cancel_actions(cid)
    import pytest
    with pytest.raises(ValueError):asyncio.run(service.submit(cid,action['action_id'],'no_longer_needed','r1'))
    assert model.calls==0


def test_processing_review_blocks_a_different_request(sessions):
    service,repo,model,cid,action=setup(sessions)
    first=repo.reserve_review(cid,action['action_id'],'r1','no_longer_needed',service.limits)
    assert first['claimed']
    import pytest
    with pytest.raises(ValueError,match='processing'):
        asyncio.run(service.submit(cid,action['action_id'],'quality_issue','r2'))
    assert model.calls==0


def test_submitted_form_restores_receipt_and_reason_after_reload(sessions):
    service,repo,model,cid,action=setup(sessions)
    repo.commit_reply(cid,'turn',100,'Eligible',[],[])
    result=asyncio.run(service.submit(cid,action['action_id'],'no_longer_needed','r1'))
    form=next(iter(service.load_forms(cid).values()))
    assert form['receipt']['refund_no']==result['refund_no']
    assert form['receipt']['reason_code']=='no_longer_needed'
    assert 'eligibility' not in form and 'fingerprints' not in form
