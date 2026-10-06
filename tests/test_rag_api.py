"""Public evidence events, stable IDs, filters and transaction failures."""
import asyncio
import json
from types import SimpleNamespace
import pytest
from sqlalchemy import select,func,event
from app.main import create_app
from app.repository import Repository
from app.db import Message
from app.ch04_db import LowConfidenceQuestion
from app.rag_generation import RagGenerator
from test_db import database_url,sessions
from test_ch04_ingest import corpus
from test_rag_chat import settings,service
from test_chat_api import post,parse_events
from test_chat_views import get


def application(sessions,corpus,**kwargs):
    chat,repo,cid,model,query,retrieval=service(sessions,corpus,**kwargs)
    app=create_app(settings(),model,sessions,query_understanding=query,rag_retrieval=retrieval,
                   rag_generator=RagGenerator(model),confidence_policy=chat.confidence_policy)
    return app,repo,cid,model,retrieval


def test_api_citation_snapshot_replay_stable_done_and_explicit_filter(sessions,corpus):
    app,repo,cid,model,retrieval=application(sessions,corpus)
    response=asyncio.run(post(app,{'message':'原始问题','conversation_id':str(cid),'filters':{'product_category':'猫砂盆'}}))
    assert response.status_code==200
    events=parse_events(response.text);kinds=[k for k,d in events]
    assert kinds.index('citations')<kinds.index('token')<kinds.index('done')
    citation=next(d for k,d in events if k=='citations');done=events[-1][1]
    assert done['message_id']==citation['message_id']
    assert retrieval.calls[0][2].product_category=='猫砂盆'
    mid=done['message_id'];snapshot=citation['citations']
    # Restoring a turn must not rescan changed documents or rewrite its evidence.
    history=get(app,f'/v1/conversations/{cid}/messages').json()['messages']
    assert history[-1]['id']==mid and history[-1]['citations']==snapshot and history[-1]['sources']==snapshot
    with sessions() as session: assert session.get(Message,int(mid)).content=='依据本轮证据[1]'


@pytest.mark.parametrize('filters',[{'product_category':None},{'product_category':''},{'unknown':'猫砂盆'}])
def test_invalid_filters_rejected_before_user_write(sessions,corpus,filters):
    app,repo,cid,*_=application(sessions,corpus)
    response=asyncio.run(post(app,{'message':'原话','conversation_id':str(cid),'filters':filters}))
    assert response.status_code==422 and repo.load_messages(cid)==[]


def test_atomic_pool_failure_yields_error_no_done_or_answer_tokens(sessions,corpus):
    app,repo,cid,*_=application(sessions,corpus,useful=False)
    def fail(connection,cursor,statement,parameters,context,executemany):
        if 'insert into low_confidence_questions' in statement.lower(): raise RuntimeError('private forced failure')
    engine=sessions.kw['bind'];event.listen(engine,'before_cursor_execute',fail)
    try: response=asyncio.run(post(app,{'message':'原话','conversation_id':str(cid)}))
    finally: event.remove(engine,'before_cursor_execute',fail)
    kinds=[k for k,d in parse_events(response.text)]
    assert kinds[-1]=='error' and not {'token','citations','done'}&set(kinds)
    assert 'private' not in response.text and [r.role for r in repo.load_messages(cid)]==['user']
    with sessions() as session: assert session.scalar(select(func.count()).select_from(LowConfidenceQuestion))==0



def test_index_service_failure_is_not_misreported_as_low_confidence(sessions,corpus):
    from app.rag_retrieval import IndexStaleError
    app,repo,cid,model,retrieval=application(sessions,corpus)
    def fail(*args): raise IndexStaleError('lost server corpus')
    retrieval.retrieve=fail
    response=asyncio.run(post(app,{'message':'原话','conversation_id':str(cid)}))
    kinds=[k for k,d in parse_events(response.text)]
    assert kinds[-1]=='error' and 'done' not in kinds and 'token' not in kinds
    assert [r.role for r in repo.load_messages(cid)]==['user'] and model.prompts==[]
    with sessions() as session: assert session.scalar(select(func.count()).select_from(LowConfidenceQuestion))==0


@pytest.mark.parametrize('phase',['generation','commit'])
def test_http_disconnect_retains_reservation_until_atomic_worker_settles(sessions,corpus,monkeypatch,phase):
    from threading import Event
    app,repo,cid,model,retrieval=application(sessions,corpus,useful=False)
    entered,write_release=Event(),Event();original=Repository.commit_knowledge_answer
    def slow(self,*args):
        if phase=='commit': entered.set();assert write_release.wait(5)
        return original(self,*args)
    monkeypatch.setattr(Repository,'commit_knowledge_answer',slow)
    async def scenario():
        disconnect=asyncio.Event();received=False;generation_entered=asyncio.Event()
        original_generate=model.generate_knowledge
        async def pending(messages):
            if phase=='generation': generation_entered.set();await asyncio.Event().wait()
            return await original_generate(messages)
        model.generate_knowledge=pending
        payload=json.dumps({'message':'断开前原话','conversation_id':str(cid)}).encode()
        scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.0'},'http_version':'1.1','method':'POST','scheme':'http',
               'path':'/v1/chat/stream','raw_path':b'/v1/chat/stream','query_string':b'',
               'headers':[(b'content-type',b'application/json')],'server':('testserver',80),'client':('test',123)}
        async def receive():
            nonlocal received
            if not received: received=True;return {'type':'http.request','body':payload,'more_body':False}
            await disconnect.wait();return {'type':'http.disconnect'}
        async def send(message): pass
        active=asyncio.create_task(app(scope,receive,send))
        try:
            if phase=='commit': assert await asyncio.to_thread(entered.wait,3)
            else: await asyncio.wait_for(generation_entered.wait(),3)
            disconnect.set();await asyncio.sleep(.03)
            if phase=='commit':
                assert not active.done()
                assert (await post(app,{'message':'重叠请求','conversation_id':str(cid)})).status_code==409
                write_release.set()
            await asyncio.wait_for(active,3)
            model.generate_knowledge=original_generate
            with sessions() as session:
                pools=list(session.scalars(select(LowConfidenceQuestion)))
                assert len(pools)==int(phase=='commit')
                assert all(p.source=='self_check' for p in pools)
            assert (await post(app,{'message':'继续问题','conversation_id':str(cid)})).status_code==200
        finally:
            disconnect.set();write_release.set()
            if not active.done(): active.cancel();await asyncio.gather(active,return_exceptions=True)
    asyncio.run(scenario())
