"""Persisted chat HTTP behavior, public SSE payloads, and stream reservations."""
import asyncio
import json
from threading import Event

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import func, select

from app.config import Settings
from app.db import Conversation
from app.main import create_app
from app.model_service import ModelService
from app.repository import Repository
from test_db import database_url, sessions
from test_tools import FakeKnowledgeSearch


def settings(**overrides):
    values = dict(chat_base_url='https://example.invalid/v1', chat_model='test-model',
                  chat_api_key='test-secret')
    values.update(overrides)
    return Settings(**values)


def parse_events(body):
    events = []
    for block in body.replace('\r\n', '\n').strip().split('\n\n'):
        fields = dict(line.split(': ', 1) for line in block.splitlines() if not line.startswith(':'))
        if fields:
            events.append((fields['event'], json.loads(fields['data'])))
    return events


async def post(app, payload):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url='http://testserver') as client:
        return await client.post('/v1/chat/stream', json=payload)


class FakeModel:
    async def choose_tool(self, messages, tools):
        return AIMessage(content='draft')

    async def stream_chat(self, messages):
        yield '你'
        yield '好'


def test_app_requires_database_url_or_injected_storage():
    with pytest.raises(ValueError, match='DATABASE_URL'):
        create_app(settings(), FakeModel(), knowledge_search=FakeKnowledgeSearch())


def test_stream_emits_decimal_session_chunks_and_persists_final(sessions):
    app = create_app(settings(), FakeModel(), session_factory=sessions, knowledge_search=FakeKnowledgeSearch())
    response = asyncio.run(post(app, {'message': '你好'}))
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    events = parse_events(response.text)
    assert [kind for kind, _ in events] == ['session', 'token', 'token', 'done']
    cid = events[0][1]['conversation_id']
    assert isinstance(cid, str) and cid.isascii() and cid.isdecimal()
    assert events[1:3] == [('token', {'text': '你'}), ('token', {'text': '好'})]
    assert events[-1] == ('done', {'conversation_id': cid})
    repo = Repository(sessions)
    assert [(r.role, r.content) for r in repo.load_messages(int(cid))] == [('user', '你好'), ('assistant', '你好')]
    assert repo.get_conversation(int(cid)).user_id.startswith('guest-')


def test_tool_status_precedes_tokens_and_contains_only_public_badge_data(sessions):
    class ToolModel(FakeModel):
        async def choose_tool(self, messages, tools):
            return AIMessage(content='', tool_calls=[{
                'id': 'request-1', 'name': 'query_order', 'args': {'order_id': 'private-order'},
            }])
    app = create_app(settings(), ToolModel(), session_factory=sessions, knowledge_search=FakeKnowledgeSearch())
    response = asyncio.run(post(app, {'message': '查询订单'}))
    events = parse_events(response.text)
    assert [kind for kind, _ in events] == ['session', 'tool_status', 'tool_status', 'token', 'token', 'done']
    assert events[1:3] == [('tool_status', {'name': 'query_order', 'state': 'running'}),
                           ('tool_status', {'name': 'query_order', 'state': 'success'})]
    assert 'private-order' not in response.text and 'request-1' not in response.text
    rows = Repository(sessions).load_messages(int(events[0][1]['conversation_id']))
    assert [r.role for r in rows] == ['user', 'assistant', 'tool', 'assistant']
    assert rows[1].tool_calls[0]['args'] == {'order_id': 'private-order'}
    assert rows[2].tool_call_id == 'request-1'


def test_restart_uses_completed_persisted_context(sessions):
    class ContextModel(FakeModel):
        async def stream_chat(self, messages):
            yield 'saved' if [m.content for m in messages[1:-1]] == ['first', '你好'] else 'lost'
    first = parse_events(asyncio.run(post(create_app(settings(), FakeModel(), sessions, knowledge_search=FakeKnowledgeSearch()), {'message': 'first'})).text)
    cid = first[0][1]['conversation_id']
    restarted = create_app(settings(), ContextModel(), sessions, knowledge_search=FakeKnowledgeSearch())
    response = asyncio.run(post(restarted, {'message': 'second', 'conversation_id': cid}))
    assert parse_events(response.text)[1] == ('token', {'text': 'saved'})


def test_unknown_decimal_conversation_returns_404_before_stream(sessions):
    response = asyncio.run(post(create_app(settings(), FakeModel(), sessions, knowledge_search=FakeKnowledgeSearch()),
                                {'message': '你好', 'conversation_id': '18446744073709551615'}))
    assert response.status_code == 404
    assert 'text/event-stream' not in response.headers['content-type']


@pytest.mark.parametrize('cid', ['missing', '', '0', '-1', '+1', '1.0', ' 1', '١', '18446744073709551616', 1])
def test_malformed_conversation_returns_422_before_stream(cid, sessions):
    response = asyncio.run(post(create_app(settings(), FakeModel(), sessions, knowledge_search=FakeKnowledgeSearch()),
                                {'message': '你好', 'conversation_id': cid}))
    assert response.status_code == 422
    assert 'text/event-stream' not in response.headers['content-type']


def test_oversized_request_creates_no_conversation_and_preserves_existing(sessions):
    repo = Repository(sessions)
    cid = repo.create_conversation('existing')
    app = create_app(settings(), FakeModel(), sessions, knowledge_search=FakeKnowledgeSearch())
    for id in (None, str(cid)):
        response = asyncio.run(post(app, {'message': 'x' * 20000, 'conversation_id': id}))
        assert response.status_code == 413
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(Conversation)) == 1
    assert repo.load_messages(cid) == []
    assert asyncio.run(post(app, {'message': 'ok', 'conversation_id': str(cid)})).status_code == 200


@pytest.mark.parametrize('phase', ['selection', 'final'])
def test_upstream_error_is_safe_and_incomplete_audit_suffix_is_ignored(phase, sessions):
    class FailingModel(FakeModel):
        async def choose_tool(self, messages, tools):
            if phase == 'selection':
                raise RuntimeError('test-secret provider credentials')
            return await super().choose_tool(messages, tools)
        async def stream_chat(self, messages):
            yield 'partial'
            raise RuntimeError('test-secret provider credentials')
    first = asyncio.run(post(create_app(settings(), FailingModel(), sessions, knowledge_search=FakeKnowledgeSearch()), {'message': 'first'}))
    events = parse_events(first.text)
    assert [kind for kind, _ in events] == (['session', 'error'] if phase == 'selection' else ['session', 'token', 'error'])
    assert 'test-secret' not in first.text and 'credentials' not in first.text
    cid = events[0][1]['conversation_id']
    assert [(r.role, r.content) for r in Repository(sessions).load_messages(int(cid))] == [('user', 'first')]
    class RetryModel(FakeModel):
        async def stream_chat(self, messages):
            yield 'clean' if len(messages) == 2 else 'dirty'
    retry = asyncio.run(post(create_app(settings(), RetryModel(), sessions, knowledge_search=FakeKnowledgeSearch()), {'message': 'retry', 'conversation_id': cid}))
    assert parse_events(retry.text)[1] == ('token', {'text': 'clean'})


def test_final_persistence_failure_emits_safe_error_and_releases_reservation(sessions, monkeypatch):
    original = Repository.append_message
    def fail_final(self, id, role, content, **kwargs):
        if role == 'assistant':
            raise RuntimeError('test-secret commit details')
        return original(self, id, role, content, **kwargs)
    monkeypatch.setattr(Repository, 'append_message', fail_final)
    app = create_app(settings(), FakeModel(), sessions, knowledge_search=FakeKnowledgeSearch())
    response = asyncio.run(post(app, {'message': 'first'}))
    events = parse_events(response.text)
    assert [kind for kind, _ in events] == ['session', 'token', 'token', 'error']
    assert 'test-secret' not in response.text
    monkeypatch.setattr(Repository, 'append_message', original)
    assert asyncio.run(post(app, {'message': 'retry', 'conversation_id': events[0][1]['conversation_id']})).status_code == 200


def test_done_on_wire_has_already_committed_final_answer(sessions):
    app = create_app(settings(), FakeModel(), sessions, knowledge_search=FakeKnowledgeSearch())
    repo = Repository(sessions)
    async def scenario():
        payload = json.dumps({'message': 'first'}).encode()
        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'},
                 'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
                 'path': '/v1/chat/stream', 'raw_path': b'/v1/chat/stream',
                 'query_string': b'', 'headers': [(b'content-type', b'application/json')],
                 'server': ('testserver', 80), 'client': ('test', 123)}
        cid = None
        observed = False
        async def receive():
            return {'type': 'http.request', 'body': payload, 'more_body': False}
        async def send(message):
            nonlocal cid, observed
            body = message.get('body', b'').decode()
            if 'event: session' in body:
                cid = int(parse_events(body)[0][1]['conversation_id'])
            if 'event: done' in body:
                observed = True
                assert [(r.role, r.content) for r in repo.load_messages(cid)] == [('user', 'first'), ('assistant', '你好')]
        await app(scope, receive, send)
        assert observed
    asyncio.run(scenario())


@pytest.mark.parametrize('cancel_write', [False, True])
def test_overlap_rejected_until_stream_and_cancelled_write_settle_then_retry(sessions, monkeypatch, cancel_write):
    repo = Repository(sessions)
    cid = str(repo.create_conversation('guest-test'))
    write_entered, write_release = Event(), Event()
    original = Repository.append_message
    def slow_write(self, id, role, content, **kwargs):
        if cancel_write and role == 'assistant' and content == 'old-answer':
            write_entered.set()
            assert write_release.wait(5)
        return original(self, id, role, content, **kwargs)
    monkeypatch.setattr(Repository, 'append_message', slow_write)
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        class SlowModel(FakeModel):
            async def stream_chat(self, messages):
                if messages[-1].content == 'active':
                    entered.set()
                    if not cancel_write:
                        await release.wait()
                    yield 'old-answer'
                else:
                    yield 'new-answer'
        app = create_app(settings(), SlowModel(), sessions, knowledge_search=FakeKnowledgeSearch())
        active = asyncio.create_task(post(app, {'message': 'active', 'conversation_id': cid}))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            if cancel_write:
                assert await asyncio.to_thread(write_entered.wait, 2)
                active.cancel()
                await asyncio.sleep(.03)
                assert not active.done()
            overlap = await post(app, {'message': 'overlap', 'conversation_id': cid})
            assert overlap.status_code == 409
            release.set()
            write_release.set()
            if cancel_write:
                with pytest.raises(asyncio.CancelledError):
                    await active
            else:
                assert (await active).status_code == 200
            retry = await post(app, {'message': 'retry', 'conversation_id': cid})
            assert retry.status_code == 200
        finally:
            release.set()
            write_release.set()
            if not active.done():
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
    asyncio.run(scenario())
    assert [(r.role, r.content) for r in repo.load_messages(int(cid))] == [
        ('user', 'active'), ('assistant', 'old-answer'), ('user', 'retry'), ('assistant', 'new-answer')]


def test_cancelled_generation_releases_reservation_without_invented_answer(sessions):
    cid = str(Repository(sessions).create_conversation('guest-test'))
    async def scenario():
        entered = asyncio.Event()
        class SlowModel(FakeModel):
            async def stream_chat(self, messages):
                if messages[-1].content == 'active':
                    entered.set()
                    await asyncio.Event().wait()
                yield 'clean'
        app = create_app(settings(), SlowModel(), sessions, knowledge_search=FakeKnowledgeSearch())
        active = asyncio.create_task(post(app, {'message': 'active', 'conversation_id': cid}))
        await asyncio.wait_for(entered.wait(), 2)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert (await post(app, {'message': 'retry', 'conversation_id': cid})).status_code == 200
    asyncio.run(scenario())
    assert [(r.role, r.content) for r in Repository(sessions).load_messages(int(cid))] == [
        ('user', 'active'), ('user', 'retry'), ('assistant', 'clean')]


def test_model_service_forwards_nonempty_text_chunks_with_configured_model(monkeypatch):
    import app.model_service as module
    class FakeChatOpenAI:
        def __init__(self, *, base_url, model, api_key, timeout, max_retries):
            assert (timeout, max_retries) == (180, 1)
            assert (base_url, model, api_key) == ('https://example.invalid/v1', 'test-model', 'test-secret')
        async def astream(self, messages):
            yield AIMessageChunk(content='a')
            yield AIMessageChunk(content='')
            yield AIMessageChunk(content='bc')
    monkeypatch.setattr(module, 'ChatOpenAI', FakeChatOpenAI)
    async def collect():
        return [chunk async for chunk in ModelService(settings()).stream_chat([])]
    assert asyncio.run(collect()) == ['a', 'bc']


def test_http_disconnect_waits_for_pending_write_before_releasing_id(sessions, monkeypatch):
    repo = Repository(sessions)
    cid = str(repo.create_conversation('guest-disconnect'))
    write_entered, write_release = Event(), Event()
    original = Repository.append_message
    def slow_write(self, id, role, content, **kwargs):
        if role == 'assistant' and content == 'old-answer':
            write_entered.set()
            assert write_release.wait(5)
        return original(self, id, role, content, **kwargs)
    monkeypatch.setattr(Repository, 'append_message', slow_write)
    class DisconnectModel(FakeModel):
        async def stream_chat(self, messages):
            yield 'old-answer' if messages[-1].content == 'active' else 'new-answer'
    app = create_app(settings(), DisconnectModel(), sessions, knowledge_search=FakeKnowledgeSearch())
    async def scenario():
        disconnect = asyncio.Event()
        payload = json.dumps({'message': 'active', 'conversation_id': cid}).encode()
        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.0'},
                 'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
                 'path': '/v1/chat/stream', 'raw_path': b'/v1/chat/stream',
                 'query_string': b'', 'headers': [(b'content-type', b'application/json')],
                 'server': ('testserver', 80), 'client': ('test', 123)}
        received = False
        async def receive():
            nonlocal received
            if not received:
                received = True
                return {'type': 'http.request', 'body': payload, 'more_body': False}
            await disconnect.wait()
            return {'type': 'http.disconnect'}
        async def send(message):
            pass
        active = asyncio.create_task(app(scope, receive, send))
        try:
            assert await asyncio.to_thread(write_entered.wait, 2)
            disconnect.set()
            await asyncio.sleep(.03)
            assert not active.done()
            assert (await post(app, {'message': 'overlap', 'conversation_id': cid})).status_code == 409
            write_release.set()
            await asyncio.wait_for(active, 2)
            assert (await post(app, {'message': 'retry', 'conversation_id': cid})).status_code == 200
        finally:
            disconnect.set()
            write_release.set()
            if not active.done():
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
    asyncio.run(scenario())
    assert [(r.role, r.content) for r in repo.load_messages(int(cid))] == [
        ('user', 'active'), ('assistant', 'old-answer'), ('user', 'retry'), ('assistant', 'new-answer')]


def test_active_capacity_limits_streams_without_evicting_persisted_chats(sessions):
    repo = Repository(sessions)
    cid = str(repo.create_conversation('guest-capacity'))
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        class SlowModel(FakeModel):
            async def stream_chat(self, messages):
                if messages[-1].content == 'active':
                    entered.set()
                    await release.wait()
                yield 'ok'
        app = create_app(settings(max_conversations=1), SlowModel(), sessions, knowledge_search=FakeKnowledgeSearch())
        active = asyncio.create_task(post(app, {'message': 'active', 'conversation_id': cid}))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert (await post(app, {'message': 'new'})).status_code == 503
            with sessions() as session:
                assert session.scalar(select(func.count()).select_from(Conversation)) == 1
            release.set()
            assert (await active).status_code == 200
            assert (await post(app, {'message': 'new'})).status_code == 200
            assert (await post(app, {'message': 'resume', 'conversation_id': cid})).status_code == 200
        finally:
            release.set()
            if not active.done():
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
    asyncio.run(scenario())
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(Conversation)) == 2


def test_faq_adds_sources_without_changing_tool_contract(sessions):
    class PostageModel(FakeModel):
        async def choose_tool(self, messages, tools):
            assert [tool.name for tool in tools] == ['query_order', 'query_product', 'query_logistics', 'query_faq', 'create_ticket']
            return AIMessage(content='', tool_calls=[{'id': 'postage-1', 'name': 'query_faq', 'args': {'keyword': '邮费'}}])

    app = create_app(settings(), PostageModel(), sessions, knowledge_search=FakeKnowledgeSearch())
    events = parse_events(asyncio.run(post(app, {'message': '邮费是多少'})).text)
    assert [kind for kind, _ in events] == ['session', 'tool_status', 'tool_status', 'sources', 'token', 'token', 'done']
    assert events[3][1]['sources'][0]['question'] == '订单运费如何计算？'
    assert events[1:3] == [('tool_status', {'name': 'query_faq', 'state': 'running'}), ('tool_status', {'name': 'query_faq', 'state': 'success'})]
    rows = Repository(sessions).load_messages(int(events[0][1]['conversation_id']))
    payload = json.loads(rows[2].content)
    assert set(payload) == {'keyword', 'matches', 'message'}
    assert payload['matches'][0]['question'] == '订单运费如何计算？'


@pytest.mark.parametrize('initialize', [False, True])
def test_app_lifespan_closes_owned_milvus_client(monkeypatch, sessions, initialize):
    from types import SimpleNamespace
    import sys
    from app.vector_store import MilvusKnowledgeStore
    closed, constructed, stores = [], [], []
    class Client:
        def __init__(self, **kwargs): constructed.append(self)
        def close(self): closed.append(self)
    monkeypatch.setitem(sys.modules, 'pymilvus', SimpleNamespace(MilvusClient=Client))
    def factory(**kwargs):
        store = MilvusKnowledgeStore(**kwargs)
        stores.append(store)
        return store
    monkeypatch.setattr('app.main.MilvusKnowledgeStore', factory)
    app = create_app(settings(), FakeModel(), sessions)
    assert len(stores) == 1 and constructed == []
    async def lifecycle():
        async with app.router.lifespan_context(app):
            if initialize:
                stores[0].client
    asyncio.run(lifecycle())
    assert len(constructed) == int(initialize)
    assert closed == constructed
    with pytest.raises(RuntimeError, match='closed'):
        stores[0].client


def test_app_lifespan_does_not_close_injected_search(monkeypatch, sessions):
    class CallerSearch(FakeKnowledgeSearch):
        def close(self):
            raise AssertionError('Injected search remains caller owned')
    monkeypatch.setattr('app.main.MilvusKnowledgeStore', lambda **kwargs: pytest.fail('Injected search must bypass client creation'))
    app = create_app(settings(), FakeModel(), sessions, CallerSearch())
    async def lifecycle():
        async with app.router.lifespan_context(app):
            pass
    asyncio.run(lifecycle())
