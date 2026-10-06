"""Stage telemetry must be request-local, private, and survive failures."""
import asyncio
import json
import logging
import pytest
from test_rag_chat import service, collect
from test_db import database_url, sessions
from test_ch04_ingest import corpus


def test_status_precedes_understanding_and_timings_cover_validated_answer(sessions, corpus, caplog):
    chat,repo,cid,model,query,retrieval=service(sessions,corpus)
    async def run():
        stream=chat.stream_turn(cid,'私密原话')
        first=await anext(stream)
        assert first.kind=='retrieval_status' and first.data['stage']=='understanding'
        assert query.calls==[]
        return [first,*[e async for e in stream]]
    with caplog.at_level(logging.INFO,logger='app.latency'):
        events=asyncio.run(run())
    timing=next(e.data for e in events if e.kind=='timings')
    assert set(('understanding','retrieval','generation','persistence','total'))<=set(timing['timings_ms'])
    assert all(v>=0 for v in timing['timings_ms'].values())
    assert timing['outcome']=='complete' and timing['first_token_ms']<=timing['timings_ms']['total']
    assert [e.data['stage'] for e in events if e.kind=='retrieval_status']==['understanding','retrieval','generation','complete']
    logs=[r.message for r in caplog.records if r.name=='app.latency']
    assert len(logs)==1 and '私密原话' not in logs[0] and '依据本轮证据' not in logs[0]
    assert json.loads(logs[0])['outcome']=='complete'


def test_upstream_failure_records_stage_without_knowledge_pool_or_private_error(sessions,corpus,caplog):
    chat,repo,cid,model,query,retrieval=service(sessions,corpus)
    async def fail(raw): raise TimeoutError('private-key-and-question')
    query.prepare=fail
    with caplog.at_level(logging.INFO,logger='app.latency'),pytest.raises(TimeoutError):
        collect(chat,cid,'private original')
    payload=json.loads(next(r.message for r in caplog.records if r.name=='app.latency'))
    assert payload['outcome']=='error' and payload['failed_stage']=='understanding'
    assert 'understanding' in payload['timings_ms']
    assert 'private' not in json.dumps(payload)
    assert [r.role for r in repo.load_messages(cid)]==['user']


def test_cancellation_records_incomplete_stage(sessions,corpus,caplog):
    chat,repo,cid,model,query,retrieval=service(sessions,corpus)
    async def run():
        entered=asyncio.Event()
        async def pending(raw): entered.set();await asyncio.Event().wait()
        query.prepare=pending
        async def drain(): return [e async for e in chat.stream_turn(cid,'cancel raw')]
        task=asyncio.create_task(drain())
        await entered.wait();task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
    with caplog.at_level(logging.INFO,logger='app.latency'): asyncio.run(run())
    payload=json.loads(next(r.message for r in caplog.records if r.name=='app.latency'))
    assert payload['outcome']=='cancelled' and payload['failed_stage']=='understanding'
