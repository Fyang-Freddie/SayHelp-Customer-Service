"""Chapter 5 application lifecycle and public action recovery."""
import asyncio
import httpx
from langchain_core.messages import AIMessage
from app.main import create_app
from app.workflow_graph import COMPLAINT
from test_db import database_url
from test_workflow_repository import sessions
from test_workflow_service import Gate, Model, settings


def test_lifespan_uses_workflow_and_restores_stable_actions(sessions, tmp_path):
    class ComplaintModel(Model):
        async def classify_intent(self, text): return AIMessage(content='{"intent":"投诉"}')
    app = create_app(settings(workflow_checkpoint_path=str(tmp_path/'api.sqlite')), ComplaintModel(), sessions, knowledge_gate=Gate())
    async def run():
        async with app.router.lifespan_context(app):
            from app.workflow_service import WorkflowService
            assert isinstance(app.state.workflow_service, WorkflowService)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                from test_chat_api import parse_events
                response = await client.post('/v1/chat/stream', json={'message': '我要投诉'})
                events = parse_events(response.text)
                assert [name for name, _ in events] == ['session', 'token', 'actions', 'done']
                assert events[1][1]['text'] == COMPLAINT
                cid = events[0][1]['conversation_id']
                actions = events[2][1]
                assert actions['message_id'] == events[-1][1]['message_id']
                assert [a['kind'] for a in actions['actions']] == ['handoff', 'create_ticket']
                assert [a['label'] for a in actions['actions']] == ['转人工', '建工单']
                assert set(actions['actions'][1]) == {'id', 'kind', 'label', 'description', 'ticket_type'}
                history = (await client.get(f'/v1/conversations/{cid}/messages')).json()['messages']
                assert history[-1]['actions'] == actions['actions']
                assert [a['label'] for a in history[-1]['actions']] == ['转人工', '建工单']
                assert history[-1]['id'] == actions['message_id']
    asyncio.run(run())
    (tmp_path/'api.sqlite').rename(tmp_path/'closed.sqlite')


def test_http_checkpoint_failure_has_error_no_done_and_reservation_released(sessions, tmp_path, monkeypatch):
    from app.workflow_repository import WorkflowRepository
    from test_chat_api import FakeModel, parse_events
    import app.workflow_service as module
    from contextlib import asynccontextmanager
    original_open = module.open_workflow_checkpointer
    fail = True
    @asynccontextmanager
    async def faulty(path):
        async with original_open(path) as saver:
            original_put = saver.aput
            async def put(config, checkpoint, metadata, new_versions):
                if fail and not config['configurable'].get('checkpoint_ns') and checkpoint['channel_values'].get('status') == 'done':
                    raise RuntimeError('private checkpoint details')
                return await original_put(config, checkpoint, metadata, new_versions)
            saver.aput = put
            yield saver
    monkeypatch.setattr(module, 'open_workflow_checkpointer', faulty)
    app = create_app(settings(workflow_checkpoint_path=str(tmp_path/'http-fault.sqlite')), FakeModel(), sessions, knowledge_gate=Gate())
    async def run():
        nonlocal fail
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                response = await client.post('/v1/chat/stream', json={'message':'first'})
                events = parse_events(response.text)
                assert events[-1][0] == 'error'
                assert not any(name in ('actions', 'done') for name, _ in events)
                assert 'private' not in response.text
                cid = events[0][1]['conversation_id']
                fail = False
                retry = await client.post('/v1/chat/stream', json={'message':'next', 'conversation_id':cid})
                assert parse_events(retry.text)[-1][0] == 'done'
                assert (await client.delete(f'/v1/conversations/{cid}')).status_code == 204
                assert (await client.post('/v1/chat/stream', json={'message':'deleted','conversation_id':cid})).status_code == 404
    asyncio.run(run())
