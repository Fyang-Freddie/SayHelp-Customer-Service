"""Persistent pin/delete contracts on disposable MySQL, no customer changes."""
import asyncio
from datetime import datetime
from sqlalchemy import select,func
from sqlalchemy.orm import undefer
import httpx,pytest
from app.db import Conversation,Message,Ticket
from app.ch04_db import LowConfidenceQuestion
from app.repository import Repository
from app.main import create_app
from test_db import database_url,sessions
from test_chat_api import settings,FakeModel,post
from test_tools import FakeKnowledgeSearch


def app(sessions,model=None): return create_app(settings(),model or FakeModel(),sessions,FakeKnowledgeSearch())
def seed(repo,count):
    ids=[repo.create_conversation(f'guest-{i}') for i in range(count)]
    for i,cid in enumerate(ids): repo.append_message(cid,'user',f'作者测试会话{i}')
    return ids
async def request(app,method,path,**kwargs):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        return await client.request(method,path,**kwargs)


def test_pin_same_second_and_mixed_pagination_has_full_sort_no_dup_or_loss(sessions):
    repo=Repository(sessions);ids=seed(repo,7)
    assert hasattr(repo,'set_pinned'),'Persistent pin port missing'
    for cid in ids[:3]: repo.set_pinned(cid,True)
    stamp=datetime(2026,10,6,8,0,0)
    with sessions.begin() as session:
        for cid in ids[:3]: session.get(Conversation,cid,options=[undefer('*')]).pinned_at=stamp
    repo.append_message(ids[0],'assistant','最近活跃不能挤掉置顶同秒id排序')
    got=[];cursor=None
    while True:
        page=repo.list_conversations(limit=2,cursor=cursor)
        got.extend(int(row['id']) for row in page['conversations']);cursor=page['next_cursor']
        if cursor is None: break
        assert not cursor.isdecimal()
    assert got==ids[:3][::-1]+ids[3:][::-1] and len(set(got))==7


def test_pin_idempotent_and_unpin_restores_activity_order(sessions):
    repo=Repository(sessions);older,newer=seed(repo,2)
    assert hasattr(repo,'set_pinned'),'Persistent pin port missing'
    first=repo.set_pinned(older,True);again=repo.set_pinned(older,True)
    assert first['is_pinned'] is True and again['pinned_at']==first['pinned_at']
    assert [int(r['id']) for r in repo.list_conversations()['conversations']]==[older,newer]
    off=repo.set_pinned(older,False)
    assert off['is_pinned'] is False and off['pinned_at'] is None
    assert [int(r['id']) for r in repo.list_conversations()['conversations']]==[newer,older]


def test_soft_delete_hides_read_resume_and_list_but_preserves_all_audit_rows(sessions):
    repo=Repository(sessions);deleted,other=seed(repo,2);repo.create_ticket(deleted,'临时数据库作者测试工单','咨询')
    with sessions.begin() as session:
        session.add(LowConfidenceQuestion(conversation_id=deleted,raw_question='测试原话',source='self_check',reason='测试不足'))
    web=app(sessions)
    first=asyncio.run(request(web,'DELETE',f'/v1/conversations/{deleted}'))
    assert first.status_code==204 and first.content==b''
    assert asyncio.run(request(web,'DELETE',f'/v1/conversations/{deleted}')).status_code==204
    assert repo.get_conversation(deleted) is None and repo.get_conversation(other) is not None
    assert [int(r['id']) for r in repo.list_conversations()['conversations']]==[other]
    assert asyncio.run(request(web,'GET',f'/v1/conversations/{deleted}/messages')).status_code==404
    assert asyncio.run(request(web,'PATCH',f'/v1/conversations/{deleted}/pin',json={'is_pinned':True})).status_code==404
    assert asyncio.run(post(web,{'message':'续聊不可见会话','conversation_id':str(deleted)})).status_code==404
    with sessions() as session:
        assert session.get(Conversation,deleted,options=[undefer('*')]).deleted_at is not None
        assert session.scalar(select(func.count()).select_from(Message))==2
        assert session.scalar(select(func.count()).select_from(Ticket))==1
        assert session.scalar(select(func.count()).select_from(LowConfidenceQuestion))==1


@pytest.mark.parametrize('body',[{}, {'is_pinned':'true'},{'is_pinned':1},{'is_pinned':None},{'is_pinned':True,'extra':'x'}])
def test_pin_body_is_strict_and_unknown_fields_rejected(sessions,body):
    cid=seed(Repository(sessions),1)[0]
    assert asyncio.run(request(app(sessions),'PATCH',f'/v1/conversations/{cid}/pin',json=body)).status_code==422


@pytest.mark.parametrize('cursor',['bad','1','0','eyJ2ZXJzaW9uIjo5fQ','a'*4096])
def test_cursor_rejects_old_numbers_corruption_or_unknown_versions(sessions,cursor):
    response=asyncio.run(request(app(sessions),'GET','/v1/conversations',params={'before':cursor}))
    assert response.status_code==422


def test_active_pin_or_delete_returns_conflict_until_turn_settles(sessions):
    cid=seed(Repository(sessions),1)[0]
    async def scenario():
        entered,release=asyncio.Event(),asyncio.Event()
        class Model(FakeModel):
            async def stream_chat(self,messages): entered.set();await release.wait();yield '测试答复'
        web=app(sessions,Model());task=asyncio.create_task(post(web,{'message':'作者测试流','conversation_id':str(cid)}))
        try:
            await asyncio.wait_for(entered.wait(),3)
            assert (await request(web,'DELETE',f'/v1/conversations/{cid}')).status_code==409
            assert (await request(web,'PATCH',f'/v1/conversations/{cid}/pin',json={'is_pinned':True})).status_code==409
            release.set();assert (await task).status_code==200
            assert (await request(web,'PATCH',f'/v1/conversations/{cid}/pin',json={'is_pinned':True})).status_code==200
            assert (await request(web,'DELETE',f'/v1/conversations/{cid}')).status_code==204
        finally:
            release.set()
            if not task.done(): task.cancel();await asyncio.gather(task,return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('method,suffix,body,status', [
    ('DELETE','999999',None,404),('PATCH','999999',{'is_pinned':True},404),
    ('DELETE','0',None,422),('PATCH','bad',{'is_pinned':True},422),
    ('DELETE','18446744073709551616',None,422)])
def test_management_missing_or_invalid_id(sessions,method,suffix,body,status):
    tail='/pin' if method=='PATCH' else ''
    assert asyncio.run(request(app(sessions),method,f'/v1/conversations/{suffix}{tail}',json=body)).status_code==status


def test_management_survives_new_app_and_repository(sessions):
    repo=Repository(sessions); pinned,deleted,other=seed(repo,3)
    assert asyncio.run(request(app(sessions),'PATCH',f'/v1/conversations/{pinned}/pin',json={'is_pinned':True})).status_code==200
    assert asyncio.run(request(app(sessions),'DELETE',f'/v1/conversations/{deleted}')).status_code==204
    payload=asyncio.run(request(app(sessions),'GET','/v1/conversations')).json()
    assert [r['id'] for r in payload['conversations']]==[str(pinned),str(other)]
    assert payload['conversations'][0]['is_pinned'] is True
    assert payload['conversations'][0]['pinned_at'] is not None


@pytest.mark.parametrize('operation',['pin','delete'])
def test_failed_update_rolls_back_and_http_never_claims_success(sessions,operation):
    from sqlalchemy import event
    from sqlalchemy.exc import SQLAlchemyError
    cid=seed(Repository(sessions),1)[0]; engine=sessions.kw['bind']
    def fail(connection,cursor,statement,parameters,context,executemany):
        if statement.startswith('UPDATE conversations'): raise SQLAlchemyError('test failure after write')
    event.listen(engine,'after_cursor_execute',fail)
    try:
        method,path,body=('PATCH',f'/v1/conversations/{cid}/pin',{'is_pinned':True}) if operation=='pin' else ('DELETE',f'/v1/conversations/{cid}',None)
        response=asyncio.run(request(app(sessions),method,path,json=body))
        assert response.status_code==503
        assert 'test failure' not in response.text
    finally: event.remove(engine,'after_cursor_execute',fail)
    row=Repository(sessions).get_conversation(cid)
    assert row is not None and not row.is_pinned and row.deleted_at is None


@pytest.mark.parametrize('winner',['chat','delete'])
def test_reservation_and_delete_share_lock_no_deleted_chat_can_resume(sessions,monkeypatch,winner):
    from threading import Event
    cid=seed(Repository(sessions),1)[0]
    assert hasattr(Repository,'soft_delete_conversation'),'Logical delete port missing'
    entered,release,loser_entered=Event(),Event(),Event()
    original_get,original_delete=Repository.get_conversation,Repository.soft_delete_conversation
    def get(repo,id):
        if winner=='chat': entered.set();assert release.wait(5)
        else: loser_entered.set()
        return original_get(repo,id)
    def delete(repo,id):
        if winner=='delete': entered.set();assert release.wait(5)
        else: loser_entered.set()
        return original_delete(repo,id)
    monkeypatch.setattr(Repository,'get_conversation',get)
    monkeypatch.setattr(Repository,'soft_delete_conversation',delete)
    async def scenario():
        model_started,model_release=asyncio.Event(),asyncio.Event()
        class Model(FakeModel):
            async def stream_chat(self,messages): model_started.set();await model_release.wait();yield '测试答复'
        web=app(sessions,Model())
        async def chat():return await post(web,{'message':'争锁作者样例','conversation_id':str(cid)})
        async def remove():return await request(web,'DELETE',f'/v1/conversations/{cid}')
        first=asyncio.create_task(chat() if winner=='chat' else remove())
        second=None
        try:
            assert await asyncio.to_thread(entered.wait,3)
            second=asyncio.create_task(remove() if winner=='chat' else chat())
            assert not await asyncio.to_thread(loser_entered.wait,.25),'Loser accessed database before reservation lock'
            release.set()
            if winner=='chat':
                assert (await asyncio.wait_for(second,4)).status_code==409
                model_release.set();assert (await asyncio.wait_for(first,4)).status_code==200
            else:
                assert (await asyncio.wait_for(first,4)).status_code==204
                assert (await asyncio.wait_for(second,4)).status_code==404
        finally:
            release.set();model_release.set()
            for task in [first,second]:
                if task is not None and not task.done():task.cancel()
            await asyncio.gather(*(t for t in [first,second] if t is not None),return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('changes',[
    {'version':2},{'pin_group':2},{'pin_group':True},{'activity_id':'1'},
    {'activity_id':0},{'id':0},{'id':18446744073709551616},
    {'pinned_at':'not-a-date'},{'pinned_at':'2026-10-06T00:00:00'},
    {'extra':1},{'pin_group':1,'pinned_at':None}])
def test_cursor_validates_complete_sort_fields(sessions,changes):
    import base64,json
    payload={'version':1,'pin_group':0,'pinned_at':None,'activity_id':1,'id':1}|changes
    encoded=base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b'=').decode()
    response=asyncio.run(request(app(sessions),'GET','/v1/conversations',params={'before':encoded}))
    assert response.status_code==422
