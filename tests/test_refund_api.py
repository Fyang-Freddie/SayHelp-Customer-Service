import asyncio
from types import SimpleNamespace
from app.main import create_app
from test_ch06_repository import sessions
from test_db import database_url
from test_refund_service import setup
from test_chat_api import settings,FakeModel,request
from test_workflow_service import Gate


def test_refund_endpoint_rejects_money_and_creates_only_pending_request(sessions):
    service,repo,model,cid,action=setup(sessions)
    app=create_app(settings(),FakeModel(),sessions,knowledge_gate=Gate())
    app.state.workflow_service=SimpleNamespace(refunds=service)
    async def run():
        url=f'/v1/conversations/{cid}/refunds'
        payload={'action_id':action['action_id'],'reason_code':'no_longer_needed','request_id':'r1'}
        assert (await request(app,'POST',url,json={**payload,'amount':'0.01'})).status_code==422
        result=await request(app,'POST',url,json=payload)
        assert result.status_code==200 and result.json()['status']=='pending'
        assert result.json()['message']=='退款申请已提交，等待处理'
        assert (await request(app,'POST',url,json=payload)).json()==result.json()
        assert model.calls==1
    asyncio.run(run())
