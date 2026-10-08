"""Explicit persisted-action confirmation against disposable MySQL."""
import asyncio
import importlib
from threading import Event
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.db import Ticket
from app.main import create_app
from app.repository import Repository
from app.workflow_repository import WorkflowRepository
from test_db import database_url
from test_workflow_repository import sessions
from test_workflow_service import Gate, Model, settings


def seed(sessions):
    repo = Repository(sessions)
    cid = repo.create_conversation('ticket-confirmation')
    _, actions = WorkflowRepository(sessions).commit_reply(cid, uuid4().hex, 1, '建议', [], [
        {'kind': 'handoff'},
        {'kind': 'create_ticket', 'description': '商品破损', 'ticket_type': '售后'}])
    return cid, actions[1]['action_id'], actions[0]['action_id']


def body(identity, **changes):
    return {'action_id': identity, 'description': '商品破损', 'ticket_type': '售后'} | changes


def count(sessions):
    with sessions() as session:
        return session.scalar(select(func.count()).select_from(Ticket))


def web(sessions):
    return create_app(settings(), Model(), sessions, knowledge_gate=Gate())


def test_confirm_calls_only_original_tool_and_retries_same_ticket(sessions, monkeypatch):
    cid, aid, _ = seed(sessions)
    module = importlib.import_module('app.ticket_actions')
    original = module.build_tools
    calls = []
    def tracked(repo, owner, search, **kwargs):
        calls.append((owner, kwargs))
        return original(repo, owner, search, **kwargs)
    monkeypatch.setattr(module, 'build_tools', tracked)
    service = module.TicketActions(Repository(sessions), WorkflowRepository(sessions))
    async def run():
        first = await service.confirm(cid, aid, '商品破损', '售后')
        assert first == await service.confirm(cid, aid, '商品破损', '售后')
        assert first == {'ticket_no': first['ticket_no'], 'status': '待处理', 'message': '已创建工单，等待处理'}
    asyncio.run(run())
    assert calls == [(cid, {'ticket_request_key': aid})] * 2
    assert count(sessions) == 1


def test_http_confirmation_retry_and_changed_payload_conflict(sessions):
    cid, aid, _ = seed(sessions)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web(sessions)), base_url='http://test') as client:
            url = f'/v1/conversations/{cid}/tickets'
            first = await client.post(url, json=body(aid))
            assert first.status_code == 200
            again = await client.post(url, json=body(aid))
            assert again.status_code == 200 and again.json() == first.json()
            for changes in ({'description': '新说明'}, {'ticket_type': '投诉'}):
                conflict = await client.post(url, json=body(aid, **changes))
                assert conflict.status_code == 409
            assert count(sessions) == 1
    asyncio.run(run())


def test_missing_cross_owner_deleted_and_handoff_cannot_create(sessions):
    cid, aid, handoff = seed(sessions)
    other = Repository(sessions).create_conversation('other')
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web(sessions)), base_url='http://test') as client:
            for owner, action, expected in ((cid, 'missing', 404), (other, aid, 404), (cid + other + 100, aid, 404), (cid, handoff, 409)):
                response = await client.post(f'/v1/conversations/{owner}/tickets', json=body(action))
                assert response.status_code == expected
            Repository(sessions).soft_delete_conversation(cid)
            assert (await client.post(f'/v1/conversations/{cid}/tickets', json=body(aid))).status_code == 404
            assert count(sessions) == 0
    asyncio.run(run())


def test_body_strictly_validates_and_forbids_identity_override(sessions):
    cid, aid, _ = seed(sessions)
    invalid = [{}, body(aid, description=' \n '), body(aid, ticket_type='其他'),
        body(aid, action_id=123), body(aid, action_id=''), body(aid, action_id='a'*65),
        body(aid, description=12), body(aid, conversation_id=str(cid)),
        body(aid, request_key='client-key'), body(aid, extra=True)]
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web(sessions)), base_url='http://test') as client:
            for payload in invalid:
                response = await client.post(f'/v1/conversations/{cid}/tickets', json=payload)
                assert response.status_code == 422, payload
            assert count(sessions) == 0
    asyncio.run(run())


def test_suggestions_ignored_or_cancelled_dialog_produce_no_write(sessions, tmp_path):
    from langchain_core.messages import AIMessage
    class ComplaintModel(Model):
        async def classify_intent(self, text): return AIMessage(content='{"intent":"投诉"}')
    app = create_app(settings(workflow_checkpoint_path=str(tmp_path/'actions.sqlite')), ComplaintModel(), sessions, knowledge_gate=Gate())
    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                from test_chat_api import parse_events
                events = parse_events((await client.post('/v1/chat/stream', json={'message': '投诉'})).text)
                cid = events[0][1]['conversation_id']
                assert any(kind == 'actions' for kind, _ in events)
                assert count(sessions) == 0
                # Dismissing the local dialog does not call the confirmation endpoint.
                again = await client.post('/v1/chat/stream', json={'message': '继续聊', 'conversation_id': cid})
                assert parse_events(again.text)[-1][0] == 'done'
                assert count(sessions) == 0
    asyncio.run(run())


def test_generation_reserves_conversation_against_confirmation(sessions, tmp_path):
    cid, aid, _ = seed(sessions)
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        class BlockingModel(Model):
            async def classify_intent(self, text):
                from langchain_core.messages import AIMessage
                entered.set(); await release.wait()
                return AIMessage(content='{"intent":"投诉"}')
        app = create_app(settings(workflow_checkpoint_path=str(tmp_path/'busy.sqlite')), BlockingModel(), sessions, knowledge_gate=Gate())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                task = asyncio.create_task(client.post('/v1/chat/stream', json={'message': '投诉', 'conversation_id': str(cid)}))
                await asyncio.wait_for(entered.wait(), 5)
                try:
                    assert (await client.post(f'/v1/conversations/{cid}/tickets', json=body(aid))).status_code == 409
                    assert count(sessions) == 0
                finally:
                    release.set(); await task
                assert (await client.post(f'/v1/conversations/{cid}/tickets', json=body(aid))).status_code == 200
    asyncio.run(run())


def test_cancelled_confirmation_holds_reservation_until_original_write_settles(sessions, monkeypatch):
    cid, aid, _ = seed(sessions)
    entered, release = Event(), Event()
    original = Repository.create_ticket
    calls = []
    def blocking(self, *args, **kwargs):
        calls.append(kwargs['request_key']); entered.set()
        if not release.wait(10): raise RuntimeError('test release timeout')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Repository, 'create_ticket', blocking)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web(sessions)), base_url='http://test') as client:
            url = f'/v1/conversations/{cid}/tickets'
            task = asyncio.create_task(client.post(url, json=body(aid)))
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel(); await asyncio.sleep(0); task.cancel(); await asyncio.sleep(0)
            try:
                assert not task.done()
                assert (await client.post(url, json=body(aid))).status_code == 409
                assert (await client.delete(f'/v1/conversations/{cid}')).status_code == 409
                assert (await client.patch(f'/v1/conversations/{cid}/pin', json={'is_pinned': True})).status_code == 409
                assert (await client.post('/v1/chat/stream', json={'message':'busy', 'conversation_id':str(cid)})).status_code == 409
            finally:
                release.set()
                with pytest.raises(asyncio.CancelledError): await task
            assert count(sessions) == 1
            assert (await client.post(url, json=body(aid))).status_code == 200
            assert count(sessions) == 1 and calls == [aid, aid]
            assert (await client.delete(f'/v1/conversations/{cid}')).status_code == 204
    asyncio.run(run())


@pytest.mark.parametrize('failure', [TimeoutError, SQLAlchemyError])
def test_unknown_result_is_safe_503_no_automatic_retry_explicit_retry_is_stable(sessions, monkeypatch, failure):
    cid, aid, _ = seed(sessions)
    original = Repository.create_ticket
    calls = []
    def uncertain(self, *args, **kwargs):
        calls.append(kwargs['request_key'])
        number = original(self, *args, **kwargs)
        if len(calls) == 1: raise failure('private connection details')
        return number
    monkeypatch.setattr(Repository, 'create_ticket', uncertain)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web(sessions)), base_url='http://test') as client:
            url = f'/v1/conversations/{cid}/tickets'
            response = await client.post(url, json=body(aid))
            assert response.status_code == 503 and 'private' not in response.text
            assert 'ticket_no' not in response.json() and calls == [aid] and count(sessions) == 1
            retry = await client.post(url, json=body(aid))
            assert retry.status_code == 200 and calls == [aid, aid] and count(sessions) == 1
    asyncio.run(run())
