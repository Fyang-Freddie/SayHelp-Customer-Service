"""Production WorkflowService evidence API; legacy ChatService tested separately."""
import asyncio
import json
from threading import Event
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import select, func
from app.main import create_app
from app.repository import Repository
from app.db import Message
from app.ch04_db import LowConfidenceQuestion
from app.evaluation.calibration import ConfidencePolicy
from app.workflow_knowledge import KnowledgeGate
from app.workflow_log import WorkflowLog
from app.workflow_repository import WorkflowRepository
from test_db import database_url
from test_workflow_repository import sessions
from test_ch04_ingest import corpus
from test_rag_chat import Retrieval
from test_chat_api import post, parse_events, settings
from test_chat_views import get


def application(sessions, corpus, tmp_path, *, weak=False):
    ingest, manifest, _ = corpus
    retrieval = Retrieval(ingest.ingest_corpus(sessions, manifest))
    repo = Repository(sessions)
    cid = repo.create_conversation('guest')
    class Model:
        def __init__(self): self.prompts = []
        async def classify_intent(self, text): return AIMessage(content='{"intent":"商品咨询"}')
        async def select(self, messages, tools, **kwargs):
            self.prompts.append(messages)
            return AIMessage(content='private draft')
        async def stream_reply(self, messages, **kwargs): yield AIMessageChunk(content='依据本轮证据[1]')
    model = Model()
    config = settings(workflow_checkpoint_path=str(tmp_path/'checkpoints.sqlite'))
    gate = KnowledgeGate(retrieval, ConfidencePolicy({'hybrid_rerank': 100 if weak else .5}, {}),
        config, log=WorkflowLog(tmp_path/'logs'))
    app = create_app(config, model, sessions, knowledge_gate=gate)
    return app, repo, cid, model, retrieval, gate


def test_api_citation_snapshot_replay_stable_done_and_explicit_filter(sessions, corpus, tmp_path):
    app, repo, cid, model, retrieval, _ = application(sessions, corpus, tmp_path)
    response = asyncio.run(post(app, {'message':'原始问题', 'conversation_id':str(cid), 'filters':{'product_category':'猫砂盆'}}))
    assert response.status_code == 200
    events = parse_events(response.text); kinds = [k for k, d in events]
    assert 'token' in kinds and 'citations' in kinds, events
    assert kinds.index('token') < kinds.index('citations') < kinds.index('done')
    citation = next(d for k, d in events if k == 'citations'); done = events[-1][1]
    assert done['message_id'] == citation['message_id']
    assert retrieval.calls[0][2].product_category == '猫砂盆'
    history = get(app, f'/v1/conversations/{cid}/messages').json()['messages']
    assert history[-1]['id'] == done['message_id']
    assert history[-1]['citations'] == citation['citations'] == history[-1]['sources']
    with sessions() as session: assert session.get(Message, int(done['message_id'])).content == '依据本轮证据[1]'


@pytest.mark.parametrize('filters', [{'product_category':None}, {'product_category':''}, {'unknown':'猫砂盆'}])
def test_invalid_filters_rejected_before_user_write(sessions, corpus, tmp_path, filters):
    app, repo, cid, *_ = application(sessions, corpus, tmp_path)
    response = asyncio.run(post(app, {'message':'原话', 'conversation_id':str(cid), 'filters':filters}))
    assert response.status_code == 422 and repo.load_messages(cid) == []


def test_weak_log_failure_never_claims_recorded_or_writes_legacy_pool(sessions, corpus, tmp_path):
    app, repo, cid, model, _, gate = application(sessions, corpus, tmp_path, weak=True)
    def fail(*args): raise OSError('private log failure')
    gate.log.write = fail
    response = asyncio.run(post(app, {'message':'原话', 'conversation_id':str(cid)}))
    events = parse_events(response.text)
    answer = ''.join(d['text'] for k, d in events if k == 'token')
    assert events[-1][0] == 'done' and '已记录' not in answer and '无法可靠回答' in answer
    assert model.prompts == [] and 'private' not in response.text
    with sessions() as session: assert session.scalar(select(func.count()).select_from(LowConfidenceQuestion)) == 0


def test_index_service_failure_is_controlled_fallback_not_weak_evidence(sessions, corpus, tmp_path):
    from app.rag_retrieval import IndexStaleError
    app, repo, cid, model, retrieval, _ = application(sessions, corpus, tmp_path)
    def fail(*args): raise IndexStaleError('lost server corpus')
    retrieval.retrieve = fail
    response = asyncio.run(post(app, {'message':'原话', 'conversation_id':str(cid)}))
    events = parse_events(response.text)
    assert events[-1][0] == 'done' and 'citations' not in [k for k, _ in events]
    assert model.prompts == [] and '已记录' not in response.text
    assert not (tmp_path/'logs/low-confidence.jsonl').exists()
    with sessions() as session: assert session.scalar(select(func.count()).select_from(LowConfidenceQuestion)) == 0


@pytest.mark.parametrize('phase', ['generation', 'commit'])
def test_http_disconnect_retains_reservation_until_atomic_worker_settles(sessions, corpus, tmp_path, monkeypatch, phase):
    app, repo, cid, model, *_ = application(sessions, corpus, tmp_path)
    entered, write_release = Event(), Event()
    original = WorkflowRepository.commit_reply
    def slow(self, *args):
        if phase == 'commit': entered.set(); assert write_release.wait(5)
        return original(self, *args)
    monkeypatch.setattr(WorkflowRepository, 'commit_reply', slow)
    async def scenario():
        disconnect = asyncio.Event(); received = False; generation_entered = asyncio.Event()
        original_stream = model.stream_reply
        async def pending(messages, **kwargs):
            if phase == 'generation': generation_entered.set(); await asyncio.Event().wait()
            async for chunk in original_stream(messages, **kwargs): yield chunk
        model.stream_reply = pending
        payload = json.dumps({'message':'断开前原话', 'conversation_id':str(cid)}).encode()
        scope = {'type':'http', 'asgi':{'version':'3.0','spec_version':'2.0'}, 'http_version':'1.1', 'method':'POST', 'scheme':'http',
            'path':'/v1/chat/stream', 'raw_path':b'/v1/chat/stream', 'query_string':b'',
            'headers':[(b'content-type',b'application/json')], 'server':('testserver',80), 'client':('test',123)}
        async def receive():
            nonlocal received
            if not received: received = True; return {'type':'http.request', 'body':payload, 'more_body':False}
            await disconnect.wait(); return {'type':'http.disconnect'}
        async def send(message): pass
        async with app.router.lifespan_context(app):
            active = asyncio.create_task(app(scope, receive, send))
            try:
                if phase == 'commit': assert await asyncio.to_thread(entered.wait, 3)
                else: await asyncio.wait_for(generation_entered.wait(), 3)
                disconnect.set(); await asyncio.sleep(.03)
                if phase == 'commit':
                    assert not active.done()
                    assert (await post(app, {'message':'重叠请求', 'conversation_id':str(cid)})).status_code == 409
                    write_release.set()
                await asyncio.wait_for(active, 3)
                assert len(repo.load_messages(cid)) == (2 if phase == 'commit' else 1)
                model.stream_reply = original_stream
                assert (await post(app, {'message':'继续问题', 'conversation_id':str(cid)})).status_code == 200
            finally:
                disconnect.set(); write_release.set()
                if not active.done(): active.cancel(); await asyncio.gather(active, return_exceptions=True)
    asyncio.run(scenario())
