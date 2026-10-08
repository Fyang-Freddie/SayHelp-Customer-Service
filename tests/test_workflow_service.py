"""Workflow orchestration with real MySQL history and official SQLite checkpoints."""
import asyncio
from contextlib import aclosing
from threading import Event
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from sqlalchemy import select

from app.ch05_db import WorkflowTurn
from app.config import Settings
from app.intent import IntentDecision
from app.repository import Repository
from app.workflow_graph import build_workflow, open_workflow_checkpointer
from app.workflow_repository import WorkflowRepository
from app.workflow_types import Usage
from app.agent_runtime import AgentLimits
from test_db import database_url
from test_workflow_repository import sessions


def settings(**kwargs):
    return Settings(chat_base_url='https://example.invalid/v1', chat_model='test', chat_api_key='test', **kwargs)


class Model:
    def __init__(self): self.inputs = []
    async def select(self, messages, tools, **kwargs):
        self.inputs.append(messages)
        return AIMessage(content='SECRET selection draft')
    async def stream_reply(self, messages, **kwargs):
        yield AIMessageChunk(content='first')
        yield AIMessageChunk(content='second')


class Gate:
    async def prepare(self, question, filters=None):
        return {'evidence': [{'answer': 'evidence'}], 'citations': [{'n': 1}],
                'confidence': {'sufficient': True}, 'suggestions': []}


def graph(saver, model, *, gate=None, intent='订单', log=None, tools_factory=None):
    async def classify(question): return IntentDecision(intent=intent), Usage(input_tokens=1, output_tokens=1)
    return build_workflow(model=model, classifier=classify, knowledge_gate=gate or Gate(),
        tools_factory=tools_factory or (lambda cid: {}), limits=AgentLimits(),
        log=log or SimpleNamespace(write=lambda *args: None), checkpointer=saver)


def service(sessions, compiled, **kwargs):
    from app.workflow_service import WorkflowService
    return WorkflowService(Repository(sessions), WorkflowRepository(sessions), compiled, settings(), **kwargs)


async def collect(service, cid, text='question'):
    return [event async for event in service.stream_turn(cid, text)]


def test_gate_before_first_token_and_real_early_final_stream(sessions, tmp_path):
    cid = Repository(sessions).create_conversation('stream')
    async def run():
        gate_started, allow_gate, first_received, allow_second = (asyncio.Event() for _ in range(4))
        class BlockingGate(Gate):
            async def prepare(self, *args):
                gate_started.set(); await allow_gate.wait()
                return await super().prepare(*args)
        class BlockingModel(Model):
            async def stream_reply(self, messages, **kwargs):
                yield AIMessageChunk(content='first')
                await allow_second.wait()
                yield AIMessageChunk(content='second')
        async with open_workflow_checkpointer(tmp_path/'stream.sqlite') as saver:
            model = BlockingModel()
            svc = service(sessions, graph(saver, model, gate=BlockingGate(), intent='商品咨询'))
            seen = []
            async def consume():
                async for event in svc.stream_turn(cid, 'question'):
                    seen.append(event)
                    if event.kind == 'token': first_received.set()
            task = asyncio.create_task(consume())
            try:
                await asyncio.wait_for(gate_started.wait(), 3)
                assert not seen and not model.inputs
                allow_gate.set()
                await asyncio.wait_for(first_received.wait(), 3)
                assert not task.done()
                assert [e.data['text'] for e in seen if e.kind == 'token'] == ['first']
                assert 'SECRET' not in str(seen)
                allow_second.set(); await asyncio.wait_for(task, 3)
                assert seen[-1].kind == 'completed'
            finally:
                allow_gate.set(); allow_second.set()
                if not task.done(): task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())
    assert [r.content for r in Repository(sessions).load_messages(cid)] == ['question', 'firstsecond']


def test_mysql_success_then_sqlite_failure_has_no_completed_and_reconciles(sessions, tmp_path):
    cid = Repository(sessions).create_conversation('fault')
    async def run():
        async with open_workflow_checkpointer(tmp_path/'fault.sqlite') as saver:
            original = saver.aput
            failed = False
            async def fail_after_reply(config, checkpoint, metadata, new_versions):
                nonlocal failed
                if not config['configurable'].get('checkpoint_ns') and checkpoint['channel_values'].get('status') == 'done':
                    while len(Repository(sessions).load_messages(cid)) < 2:
                        await asyncio.sleep(.001)
                    failed = True
                    raise RuntimeError('sqlite checkpoint failure')
                return await original(config, checkpoint, metadata, new_versions)
            saver.aput = fail_after_reply
            compiled = graph(saver, Model())
            seen = []
            with pytest.raises(RuntimeError, match='sqlite'):
                async for event in service(sessions, compiled).stream_turn(cid, 'first question'): seen.append(event)
            assert failed and not any(e.kind in ('actions', 'completed') for e in seen)
            assert len(Repository(sessions).load_messages(cid)) == 2
            with sessions() as session:
                assert session.scalar(select(WorkflowTurn.status)) == 'checkpoint_failed'
            saver.aput = original
            model = Model()
            events = await collect(service(sessions, graph(saver, model)), cid, 'next')
            assert events[-1].kind == 'completed'
            assert [m.content for m in model.inputs[0]][1:] == ['first question', 'firstsecond', 'next']
            assert len(Repository(sessions).load_messages(cid)) == 4
    asyncio.run(asyncio.wait_for(run(), 10))


def test_child_sqlite_completed_then_mysql_failure_never_completes(sessions, tmp_path, monkeypatch):
    cid = Repository(sessions).create_conversation('mysql-fault')
    async def run():
        async with open_workflow_checkpointer(tmp_path/'fault.sqlite') as saver:
            svc = service(sessions, graph(saver, Model()))
            checkpoint_done = Event()
            original_put = saver.aput
            async def record_put(config, checkpoint, metadata, new_versions):
                value = await original_put(config, checkpoint, metadata, new_versions)
                if config['configurable'].get('checkpoint_ns', '').startswith('agent:') and checkpoint['channel_values'].get('status') == 'done':
                    checkpoint_done.set()
                return value
            saver.aput = record_put
            def fail(*args):
                assert checkpoint_done.wait(3), 'SQLite completed state must be written before MySQL failure'
                raise RuntimeError('mysql failure')
            monkeypatch.setattr(svc.workflow_repository, 'commit_reply', fail)
            seen = []
            with pytest.raises(RuntimeError, match='mysql'):
                async for event in svc.stream_turn(cid, 'first'): seen.append(event)
            assert checkpoint_done.is_set()
            snapshots = [item async for item in saver.alist({'configurable': {'thread_id': str(cid)}})]
            assert any(item.config['configurable']['checkpoint_ns'].startswith('agent:')
                and item.checkpoint['channel_values'].get('status') == 'done' for item in snapshots)
            assert not any(e.kind in ('actions', 'completed') for e in seen)
            assert len(Repository(sessions).load_messages(cid)) == 1
            model = Model()
            await collect(service(sessions, graph(saver, model)), cid, 'retry')
            assert [m.content for m in model.inputs[0]][1:] == ['retry']
    asyncio.run(run())


def test_cancelled_commit_settles_before_exit_and_records_incomplete(sessions, tmp_path, monkeypatch):
    cid = Repository(sessions).create_conversation('cancel')
    entered, release = Event(), Event()
    async def run():
        async with open_workflow_checkpointer(tmp_path/'cancel.sqlite') as saver:
            svc = service(sessions, graph(saver, Model()))
            original = svc.workflow_repository.commit_reply
            def blocked(*args):
                entered.set(); assert release.wait(5)
                return original(*args)
            monkeypatch.setattr(svc.workflow_repository, 'commit_reply', blocked)
            task = asyncio.create_task(collect(svc, cid))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                task.cancel(); await asyncio.sleep(.03); task.cancel(); await asyncio.sleep(.03)
                assert not task.done()
                release.set()
                with pytest.raises(asyncio.CancelledError): await task
                assert len(Repository(sessions).load_messages(cid)) == 2
                with sessions() as session:
                    assert session.scalar(select(WorkflowTurn.status)) == 'incomplete'
                model = Model()
                await collect(service(sessions, graph(saver, model)), cid, 'next')
                assert [m.content for m in model.inputs[0]][1:] == ['next']
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_deleted_conversation_rejected_without_graph_execution(sessions, tmp_path):
    repo = Repository(sessions); cid = repo.create_conversation('deleted'); repo.soft_delete_conversation(cid)
    async def run():
        async with open_workflow_checkpointer(tmp_path/'deleted.sqlite') as saver:
            model = Model()
            with pytest.raises((KeyError, LookupError)):
                await collect(service(sessions, graph(saver, model)), cid)
            assert not model.inputs
    asyncio.run(run())


def test_status_transition_refreshes_turn_after_conversation_lock(sessions, monkeypatch):
    import app.workflow_repository as module
    cid = Repository(sessions).create_conversation('lock-status')
    repo = WorkflowRepository(sessions)
    repo.append_message_once(cid, 'stale-status', 0, 'user', 'question')
    original = module._active_conversation
    def changed_before_lock(session, conversation_id):
        with sessions.begin() as other:
            other.get(WorkflowTurn, 'stale-status').status = 'complete'
        return original(session, conversation_id)
    monkeypatch.setattr(module, '_active_conversation', changed_before_lock)
    repo.set_turn_status('stale-status', 'running')
    with sessions() as session:
        assert session.get(WorkflowTurn, 'stale-status').status == 'running'


def test_reconciliation_replaces_stale_checkpoint_and_preserves_bounded_multi_batch_history(sessions, tmp_path):
    from app.history import completed_turns
    from app.prompts import render_workflow_system_prompt
    repo = Repository(sessions); cid = repo.create_conversation('legacy')
    repo.append_message(cid, 'user', 'too old')
    repo.append_message(cid, 'assistant', 'old answer')
    repo.append_message(cid, 'user', 'completed question')
    for i in range(2):
        repo.append_message(cid, 'assistant', '', tool_calls=[{'id':f'old-call-{i}','name':'query_order','args':{'order_id':'100'}}])
        repo.append_message(cid, 'tool', 'saved result', tool_call_id=f'old-call-{i}')
    repo.append_message(cid, 'assistant', 'completed answer')
    repo.append_message(cid, 'user', 'unfinished user')
    expected = completed_turns(repo.load_messages(cid))[-1]
    async def run():
        async with open_workflow_checkpointer(tmp_path/'stale.sqlite') as saver:
            compiled = graph(saver, Model())
            # A separate graph invocation represents a stale SQLite-only successful answer.
            await compiled.ainvoke({'conversation_id':cid, 'turn_id':'stale',
                'raw_question':'SQLite only', 'messages':[HumanMessage(content='SQLite only')], 'filters':{}},
                {'configurable':{'thread_id':str(cid)}})
            model = Model()
            svc = service(sessions, graph(saver, model))
            svc.settings = settings(max_turns_per_conversation=1)
            await collect(svc, cid, 'new question')
            imported = model.inputs[0]
            assert imported[0].content == render_workflow_system_prompt()
            assert [m.id for m in imported[1:-1]] == [m.id for m in expected]
            assert [m.content for m in imported[1:-1]] == [m.content for m in expected]
            assert len({m.id for m in imported}) == len(imported)
            await collect(svc, cid, 'next question')
            assert [m.content for m in model.inputs[-1]][1:] == ['new question', 'firstsecond', 'next question']
            assert len(repo.load_messages(cid)) == 13
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['empty_final', 'log'])
def test_unfinished_or_log_failure_never_publishes_completion(sessions, tmp_path, failure):
    cid = Repository(sessions).create_conversation('unfinished')
    class EmptyModel(Model):
        async def stream_reply(self, messages, **kwargs):
            yield AIMessageChunk(content='')
    def fail_log(*args): raise RuntimeError('log unavailable')
    async def run():
        async with open_workflow_checkpointer(tmp_path/'unfinished.sqlite') as saver:
            svc = service(sessions, graph(saver, EmptyModel() if failure == 'empty_final' else Model(),
                log=SimpleNamespace(write=fail_log) if failure == 'log' else None))
            seen = []
            with pytest.raises(RuntimeError):
                async for event in svc.stream_turn(cid, 'question'): seen.append(event)
            assert not any(e.kind in ('actions', 'completed') for e in seen)
    asyncio.run(run())


def test_failed_checkpoint_never_replays_tool_and_final_write_is_idempotent(sessions, tmp_path):
    from app.tools import build_tools
    repo = Repository(sessions); cid = repo.create_conversation('no-replay')
    class ReadModel(Model):
        async def select(self, messages, tools, **kwargs):
            self.inputs.append(messages)
            if messages[-1].type == 'human':
                return AIMessage(content='PRIVATE draft before tool call', tool_calls=[
                    {'id':'read-once','name':'query_order','args':{'order_id':'100'}}])
            return AIMessage(content='')
    async def run():
        async with open_workflow_checkpointer(tmp_path/'no-replay.sqlite') as saver:
            model = ReadModel()
            svc = service(sessions, graph(saver, model,
                tools_factory=lambda id: build_tools(repo, id, None)))
            original_commit = svc.workflow_repository.commit_reply
            writes = []
            def repeated_commit(*args):
                first = original_commit(*args)
                assert original_commit(*args) == first
                writes.append(first)
                return first
            svc.workflow_repository.commit_reply = repeated_commit
            original_put = saver.aput
            async def fail_final(config, checkpoint, metadata, new_versions):
                if not config['configurable'].get('checkpoint_ns') and checkpoint['channel_values'].get('status') == 'done':
                    while not writes: await asyncio.sleep(.001)
                    raise RuntimeError('checkpoint failure')
                return await original_put(config, checkpoint, metadata, new_versions)
            saver.aput = fail_final
            seen = []
            with pytest.raises(RuntimeError, match='checkpoint'):
                async for event in svc.stream_turn(cid, 'read order100'): seen.append(event)
            assert len(writes) == 1
            assert len([e for e in seen if e.kind == 'tool_status' and e.data['state'] == 'running']) == 1
            assert 'PRIVATE' not in str(seen) and 'read-once' not in str(seen)
            assert [r.role for r in repo.load_messages(cid)] == ['user','assistant','tool','assistant']
            saver.aput = original_put
            fresh = Model()
            events = await collect(service(sessions, graph(saver, fresh)), cid, 'continue')
            assert not any(e.kind == 'tool_status' for e in events)
            assert [m.type for m in fresh.inputs[0]][1:] == ['human','ai','tool','ai','human']
            assert len(repo.load_messages(cid)) == 6
    asyncio.run(asyncio.wait_for(run(), 10))


def test_deleted_during_final_stream_prevents_reply_and_actions(sessions, tmp_path):
    repo = Repository(sessions); cid = repo.create_conversation('delete-during-stream')
    class DeletedModel(Model):
        async def stream_reply(self, messages, **kwargs):
            yield AIMessageChunk(content='partial')
            await asyncio.to_thread(repo.soft_delete_conversation, cid)
            yield AIMessageChunk(content='rest')
    async def run():
        async with open_workflow_checkpointer(tmp_path/'deleted.sqlite') as saver:
            seen = []
            with pytest.raises(KeyError):
                async for event in service(sessions, graph(saver, DeletedModel())).stream_turn(cid, 'question'):
                    seen.append(event)
            assert not any(e.kind in ('actions', 'completed') for e in seen)
            with sessions() as session:
                from app.db import Message
                assert len(list(session.scalars(select(Message).where(Message.conversation_id == cid)))) == 1
    asyncio.run(run())
