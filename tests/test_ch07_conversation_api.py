from types import SimpleNamespace
from hashlib import sha256
from fastapi.testclient import TestClient
from app.config import Settings
from app.repository import Repository
from test_workflow_repository import sessions
from test_db import database_url


def test_scoped_sidebar_lists_empty_and_newest_first_with_stable_pages(sessions,database_url):
    from app.init_ch07_db import initialize_ch07_database
    initialize_ch07_database(database_url)
    repo=Repository(sessions)
    old=repo.create_conversation('visitor')
    repo.append_message(old,'user','最早的问题')
    newest=repo.create_conversation('visitor')
    repo.set_pinned(old,True)
    repo.append_message(old,'assistant','旧会话的新回复')
    first=repo.list_conversations(user_id='visitor',limit=1)
    assert [row['id'] for row in first['conversations']]==[str(newest)]
    assert first['conversations'][0]['title']=='新对话'
    second=repo.list_conversations(user_id='visitor',limit=1,cursor=first['next_cursor'])
    assert [row['id'] for row in second['conversations']]==[str(old)]
    assert second['next_cursor'] is None


def test_visitors_are_scoped_on_api_and_legacy_paths(sessions,database_url):
    from app.init_ch07_db import initialize_ch07_database
    from app.main import create_app
    initialize_ch07_database(database_url)
    settings=Settings(chat_base_url='x',chat_model='x',chat_api_key='x',layered_context=True,
        context_token_budget=18000,response_token_reserve=2000)
    app=create_app(settings,SimpleNamespace(),sessions)
    async def pending(cid):return None
    app.state.workflow_service=SimpleNamespace(get_pending_interaction=pending,interactions=None,refunds=None)
    a,b=TestClient(app),TestClient(app)
    response=a.get('/api/conversations')
    assert response.status_code==200
    assert 'httponly' in response.headers['set-cookie'].lower()
    visitor=sha256(a.cookies['sayhelp_visitor'].encode()).hexdigest()
    repo=Repository(sessions)
    cid=repo.create_conversation(visitor)
    repo.append_message(cid,'user','订单ABC123的退款')
    repo.append_message(cid,'assistant','仍待核实')
    assert a.get('/api/conversations').json()['conversations'][0]['title']=='订单ABC123的退款'
    assert a.get(f'/api/conversations/{cid}/messages').json()['messages'][0]['content']=='订单ABC123的退款'
    assert b.get('/api/conversations').json()['conversations']==[]
    for prefix in ('api','v1'):
        assert b.get(f'/{prefix}/conversations/{cid}/messages').status_code==404
    assert b.post('/v1/chat/stream',json={'conversation_id':str(cid),'message':'继续'}).status_code==404
    assert b.patch(f'/v1/conversations/{cid}/pin',json={'is_pinned':True}).status_code==404
    assert b.post(f'/v1/conversations/{cid}/refunds',json={}).status_code==404
    assert b.post(f'/v1/conversations/{cid}/interactions/fake/resume',json={}).status_code==404
    assert b.delete(f'/v1/conversations/{cid}').status_code==404
    assert b.post(f'/v1/conversations/{cid}/tickets',json={}).status_code==404
    from app.context_repository import ContextRepository
    third=int(repo.append_message(cid,'user','继续查询'))
    repo.append_message(cid,'assistant','等待处理')
    contexts=ContextRepository(sessions)
    assert contexts.advance_layer1(cid,0,third)
    rows=repo.load_messages(cid)
    lease=contexts.claim(cid,rows[0].id,rows[1].id,'a'*64)
    assert contexts.complete(lease,'订单ABC123退款待核实。',20)
    assert a.get('/api/conversations').json()['conversations'][0]['has_summary'] is True
    assert a.get(f'/api/conversations/{cid}/messages').json()['messages'][0]['content']=='订单ABC123的退款'

    from app.chat_service import ChatEvent
    import json
    async def stream_turn(id,message,filters):
        repo.append_message(id,'user',message)
        answer=repo.append_message(id,'assistant','答复原文')
        yield ChatEvent('token',{'text':'答复原文'})
        yield ChatEvent('completed',{'message_id':answer})
    app.state.workflow_service.stream_turn=stream_turn
    fresh=TestClient(app)
    ids=[]
    for message in ('新对话一','新对话二'):
        response=fresh.post('/v1/chat/stream',json={'message':message})
        assert response.status_code==200
        ids.append(next(json.loads(line[6:])['conversation_id'] for line in response.text.splitlines()
                        if line.startswith('data: ') and 'conversation_id' in line))
    assert len(set(ids))==2
    assert len(fresh.get('/api/conversations').json()['conversations'])==2
    owner=sha256(fresh.cookies['sayhelp_visitor'].encode()).hexdigest()
    assert all(repo.get_conversation(int(id)).user_id==owner for id in ids)
