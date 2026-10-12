import asyncio
from app.main import create_app
from app.workflow_graph import open_workflow_checkpointer
from app.repository import Repository
from test_ch06_repository import sessions
from test_db import database_url
from test_ch06_recovery import graph,service
from test_chat_api import settings,FakeModel,request,parse_events
from test_workflow_service import Gate


def test_http_pause_order_resume_and_invalid_scope(sessions,tmp_path):
    async def run():
        cid=Repository(sessions).create_conversation('customer');events=[]
        app=create_app(settings(),FakeModel(),sessions,knowledge_gate=Gate())
        async with open_workflow_checkpointer(tmp_path/'state.sqlite') as saver:
            app.state.workflow_service=service(sessions,graph(saver,events))
            first=await request(app,'POST','/v1/chat/stream',json={'conversation_id':str(cid),'message':'我要退款'})
            frames=parse_events(first.text)
            assert [kind for kind,_ in frames][-2:]==['interaction_required','paused']
            assert all(kind not in ('done','error') for kind,_ in frames)
            history=(await request(app,'GET',f'/v1/conversations/{cid}/messages')).json()
            pending=history['pending_interaction'];assert pending['kind']=='customer_input'
            url=f'/v1/conversations/{cid}/interactions/{pending["interrupt_id"]}/resume'
            response=await request(app,'POST',url,json={'turn_id':pending['turn_id'],'request_id':'r1','value':{'customer_id':'C1'}})
            assert parse_events(response.text)[-1][0]=='paused'
            pending=(await request(app,'GET',f'/v1/conversations/{cid}/messages')).json()['pending_interaction']
            url=f'/v1/conversations/{cid}/interactions/{pending["interrupt_id"]}/resume'
            payload={'turn_id':pending['turn_id'],'request_id':'r2','value':{'customer_id':'C1','order_id':'O1','item_id':'I1','data_digest':'data-v1'}}
            bad={**payload,'value':{**payload['value'],'customer_id':'C2'}}
            assert (await request(app,'POST',url,json=bad)).status_code==409
            other=Repository(sessions).create_conversation('other')
            assert (await request(app,'POST',url.replace(f'/{cid}/',f'/{other}/'),json=payload)).status_code in (404,409)
            response=await request(app,'POST',url,json=payload)
            assert parse_events(response.text)[-1][0]=='done'
            before=list(events)
            response=await request(app,'POST',url,json=payload)
            assert parse_events(response.text)[-1][0]=='done' and events==before
            assert (await request(app,'GET',f'/v1/conversations/{cid}/messages')).json()['pending_interaction'] is None
    asyncio.run(run())


def test_unknown_customer_keeps_original_wait_available(sessions,tmp_path):
    async def run():
        cid=Repository(sessions).create_conversation('customer');events=[]
        app=create_app(settings(),FakeModel(),sessions,knowledge_gate=Gate())
        async with open_workflow_checkpointer(tmp_path/'state.sqlite') as saver:
            svc=service(sessions,graph(saver,events));app.state.workflow_service=svc
            from test_aftersales_workflow import Business
            svc.business=Business(events)
            await request(app,'POST','/v1/chat/stream',json={'conversation_id':str(cid),'message':'退款'})
            pending=await svc.get_pending_interaction(cid)
            url=f'/v1/conversations/{cid}/interactions/{pending["interrupt_id"]}/resume'
            response=await request(app,'POST',url,json={'turn_id':pending['turn_id'],'request_id':'invalid','value':{'customer_id':'C2'}})
            assert response.status_code==409
            assert (await svc.get_pending_interaction(cid))['status']=='waiting'
            assert events==['ask_customer']
    asyncio.run(run())
