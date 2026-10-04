"""Single-tool turns, persistent audit rows, and restart context behavior."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.chat_service import ChatService
from app.config import Settings
from app.history import InputBudgetExceeded, UnknownConversation
from app.model_service import ModelService
from app.repository import Repository
from test_db import database_url, sessions
from test_tools import FakeKnowledgeSearch


def settings(**overrides):
    values = dict(chat_base_url='https://example.invalid', chat_model='test',
                  chat_api_key='test', context_token_budget=4096,
                  response_token_reserve=512)
    values.update(overrides)
    return Settings(**values)


def call(name='query_faq', args=None, id='call-1'):
    return {'name': name, 'args': args if args is not None else {'keyword': '退货'}, 'id': id}


class MemoryRepository:
    """Explicit test injection; production services always use their repository."""
    def __init__(self):
        self.rows = []
        self.keywords = []
        self.tickets = []

    def get_conversation(self, id):
        return SimpleNamespace(id=id) if id == 42 else None

    def load_messages(self, id):
        return copy.deepcopy(self.rows)

    def append_message(self, id, role, content, tool_calls=None, tool_call_id=None):
        self.rows.append(SimpleNamespace(role=role, content=content,
                                        tool_calls=copy.deepcopy(tool_calls), tool_call_id=tool_call_id))

    def find_faq(self, *args, **kwargs):
        raise AssertionError('Online chat must never use SQL keyword lookup')

    def create_ticket(self, id, description, ticket_type):
        self.tickets.append((id, description, ticket_type))
        return 'T-demo'


class FakeModel:
    def __init__(self, selection=None, chunks=('流式', '回答'), selection_error=None, final_error=None):
        self.selection = selection if selection is not None else AIMessage(content='discard this draft')
        self.chunks = chunks
        self.selection_error = selection_error
        self.final_error = final_error
        self.selection_prompts = []
        self.final_prompts = []
        self.tools = []

    async def choose_tool(self, messages, tools):
        self.selection_prompts.append(messages.copy())
        self.tools = tools
        if self.selection_error:
            raise self.selection_error
        return self.selection

    async def stream_chat(self, messages):
        self.final_prompts.append(messages.copy())
        for chunk in self.chunks:
            yield chunk
        if self.final_error:
            raise self.final_error


def run(service, message='退货政策是什么', id=42):
    async def collect():
        return [event async for event in service.stream_turn(id, message)]
    return asyncio.run(collect())


def test_no_tool_answer_streams_and_saves_only_final_answer():
    repo, model = MemoryRepository(), FakeModel()
    events = run(ChatService(repo, model, settings(), knowledge_search=FakeKnowledgeSearch(repo)))
    assert [(event.kind, event.data) for event in events] == [('token', {'text': '流式'}), ('token', {'text': '回答'})]
    assert [(row.role, row.content) for row in repo.rows] == [('user', '退货政策是什么'), ('assistant', '流式回答')]
    assert [tool.name for tool in model.tools] == ['query_order', 'query_product', 'query_logistics', 'query_faq', 'create_ticket']
    assert model.final_prompts[0] == model.selection_prompts[0]


def test_one_tool_status_pairing_and_persisted_json():
    repo = MemoryRepository()
    model = FakeModel(AIMessage(content='', tool_calls=[call()]))
    events = run(ChatService(repo, model, settings(), knowledge_search=FakeKnowledgeSearch(repo)))
    assert [(event.kind, event.data) for event in events[:2]] == [
        ('tool_status', {'name': 'query_faq', 'state': 'running'}),
        ('tool_status', {'name': 'query_faq', 'state': 'success'})]
    assert [row.role for row in repo.rows] == ['user', 'assistant', 'tool', 'assistant']
    assert repo.rows[1].content is None
    assert repo.rows[1].tool_calls == model.selection.tool_calls
    assert repo.rows[2].tool_call_id == 'call-1'
    assert json.loads(repo.rows[2].content)['matches'][0]['answer'] == '七天内申请'
    assert isinstance(model.final_prompts[0][-2], AIMessage)
    assert isinstance(model.final_prompts[0][-1], ToolMessage)
    assert model.final_prompts[0][-1].tool_call_id == 'call-1'


def test_multiple_calls_execute_only_first_and_pair_every_request():
    repo = MemoryRepository()
    model = FakeModel(AIMessage(content='', tool_calls=[call(), call('create_ticket', {'description': '咨询', 'ticket_type': '咨询'}, 'call-2')]))
    run(ChatService(repo, model, settings(), knowledge_search=FakeKnowledgeSearch(repo)))
    assert repo.keywords == ['退货'] and repo.tickets == []
    assert [row.tool_call_id for row in repo.rows if row.role == 'tool'] == ['call-1', 'call-2']
    skipped = model.final_prompts[0][-1]
    assert skipped.status == 'error' and '跳过' in skipped.content
    assert len(repo.rows[1].tool_calls) == 2


@pytest.mark.parametrize('word,keyword,has_match', [('退货政策是什么', '退货', True), ('邮费是多少', '邮费', True)])
def test_labeled_faq_wording_handoff(word, keyword, has_match):
    # Fake retrieval validates semantic result handoff; live BGE/Milvus acceptance is separate.
    repo = MemoryRepository()
    model = FakeModel(AIMessage(content='', tool_calls=[call(args={'keyword': keyword})]))
    run(ChatService(repo, model, settings(), knowledge_search=FakeKnowledgeSearch(repo)), word)
    payload = json.loads(model.final_prompts[0][-1].content)
    assert repo.keywords == [keyword]
    assert payload['keyword'] == keyword and bool(payload['matches']) is has_match


def test_restarted_service_reconstructs_completed_tool_turn():
    repo = MemoryRepository()
    first = FakeModel(AIMessage(content='', tool_calls=[call()]))
    run(ChatService(repo, first, settings(), knowledge_search=FakeKnowledgeSearch(repo)))
    second = FakeModel()
    run(ChatService(repo, second, settings(), knowledge_search=FakeKnowledgeSearch(repo)), '接下来怎么办')
    prompt = second.selection_prompts[0]
    assert [m.type for m in prompt] == ['system', 'human', 'ai', 'tool', 'ai', 'human']
    assert prompt[2].tool_calls == first.selection.tool_calls
    assert prompt[3].tool_call_id == 'call-1'
    assert prompt[4].content == '流式回答'


def test_budget_pruning_removes_whole_tool_turn_without_deleting_rows():
    repo = MemoryRepository()
    first = FakeModel(AIMessage(content='', tool_calls=[call()]), chunks=('x' * 20000,))
    run(ChatService(repo, first, settings(), knowledge_search=FakeKnowledgeSearch(repo)))
    saved = copy.deepcopy(repo.rows)
    second = FakeModel()
    run(ChatService(repo, second, settings(), knowledge_search=FakeKnowledgeSearch(repo)), '继续')
    assert [m.type for m in second.selection_prompts[0]] == ['system', 'human']
    assert repo.rows[:len(saved)] == saved


def test_max_turns_prunes_context_without_database_deletion():
    repo = MemoryRepository()
    for text in ['old', 'recent']:
        run(ChatService(repo, FakeModel(chunks=(text,)), settings(), knowledge_search=FakeKnowledgeSearch(repo)), text)
    model = FakeModel()
    run(ChatService(repo, model, settings(max_turns_per_conversation=1), knowledge_search=FakeKnowledgeSearch(repo)), 'current')
    assert [m.content for m in model.selection_prompts[0][1:]] == ['recent', 'recent', 'current']
    assert len(repo.rows) == 6


@pytest.mark.parametrize('stage', ['selection', 'final'])
def test_upstream_failure_retains_audit_trail_without_final_assistant(stage):
    repo = MemoryRepository()
    model = FakeModel(AIMessage(content='', tool_calls=[call()]), **{stage + '_error': RuntimeError('private error')})
    with pytest.raises(RuntimeError):
        run(ChatService(repo, model, settings(), knowledge_search=FakeKnowledgeSearch(repo)))
    assert [row.role for row in repo.rows] == (['user'] if stage == 'selection' else ['user', 'assistant', 'tool'])
    saved_count = len(repo.rows)
    restarted = FakeModel()
    run(ChatService(repo, restarted, settings(), knowledge_search=FakeKnowledgeSearch(repo)), 'new turn')
    assert [m.type for m in restarted.selection_prompts[0]] == ['system', 'human']
    assert len(repo.rows) == saved_count + 2


def test_unknown_conversation_and_input_budget_fail_before_writing():
    repo, model = MemoryRepository(), FakeModel()
    with pytest.raises(UnknownConversation):
        run(ChatService(repo, model, settings(), knowledge_search=FakeKnowledgeSearch(repo)), id=99)
    with pytest.raises(InputBudgetExceeded):
        run(ChatService(repo, model, settings(context_token_budget=20, response_token_reserve=10), knowledge_search=FakeKnowledgeSearch(repo)))
    assert repo.rows == [] and model.selection_prompts == []


def test_invalid_name_status_does_not_publish_raw_model_name():
    repo = MemoryRepository()
    model = FakeModel(AIMessage(content='', tool_calls=[call('private-model-name')]))
    events = run(ChatService(repo, model, settings(), knowledge_search=FakeKnowledgeSearch(repo)))
    assert events[0].data == {'name': 'unknown', 'state': 'running'}
    assert events[1].data == {'name': 'unknown', 'state': 'error'}
    assert model.final_prompts[0][-1].status == 'error'


def test_real_mysql_persists_order_json_and_reconstructs_after_restart(sessions):
    repo = Repository(sessions)
    cid = repo.create_conversation('guest-chat')
    model = FakeModel(AIMessage(content='', tool_calls=[call()]))
    run(ChatService(repo, model, settings(), knowledge_search=FakeKnowledgeSearch(repo)), id=cid)
    rows = repo.load_messages(cid)
    assert [r.role for r in rows] == ['user', 'assistant', 'tool', 'assistant']
    assert rows[1].tool_calls == model.selection.tool_calls
    assert rows[2].tool_call_id == rows[1].tool_calls[0]['id']
    next_model = FakeModel()
    run(ChatService(Repository(sessions), next_model, settings(), knowledge_search=FakeKnowledgeSearch()), '继续', cid)
    assert [m.type for m in next_model.selection_prompts[0]] == ['system', 'human', 'ai', 'tool', 'ai', 'human']


def test_model_selection_awaits_once_and_final_stream_uses_unbound_model():
    class Bound:
        async def ainvoke(self, messages):
            calls.append(('ainvoke', messages))
            await asyncio.sleep(0)
            return AIMessage(content='', tool_calls=[call()])
    class Unbound:
        def bind_tools(self, tools):
            calls.append(('bind', tools))
            return Bound()
        async def astream(self, messages):
            calls.append(('astream', messages))
            yield SimpleNamespace(text='final')
    calls = []
    service = ModelService.__new__(ModelService)
    service._model = Unbound()
    async def scenario():
        messages = [HumanMessage(content='question')]
        result = await service.choose_tool(messages, [])
        assert result.tool_calls[0]['id'] == 'call-1'
        assert [text async for text in service.stream_chat(messages)] == ['final']
    asyncio.run(scenario())
    assert [item[0] for item in calls] == ['bind', 'ainvoke', 'astream']


def test_invalid_json_call_is_paired_and_skips_later_valid_call_after_restart():
    repo = MemoryRepository()
    invalid = {'name': 'query_faq', 'args': '{bad json', 'id': 'bad-1', 'error': 'invalid JSON'}
    valid = call(id='good-2')
    selection = AIMessage(content='', tool_calls=[valid], invalid_tool_calls=[invalid],
                          additional_kwargs={'tool_calls': [
                              {'id': 'bad-1', 'type': 'function', 'function': {'name': 'query_faq', 'arguments': '{bad json'}},
                              {'id': 'good-2', 'type': 'function', 'function': {'name': 'query_faq', 'arguments': '{"keyword":"退货"}'}}]})
    model = FakeModel(selection)
    run(ChatService(repo, model, settings(), knowledge_search=FakeKnowledgeSearch(repo)))
    assert repo.keywords == []
    assert [row.tool_call_id for row in repo.rows if row.role == 'tool'] == ['bad-1', 'good-2']
    assert len(repo.rows[1].tool_calls) == 2
    assert all(m.status == 'error' for m in model.final_prompts[0] if isinstance(m, ToolMessage))
    restarted = FakeModel()
    run(ChatService(repo, restarted, settings(), knowledge_search=FakeKnowledgeSearch(repo)), '继续')
    assert restarted.selection_prompts[0][2].invalid_tool_calls[0]['id'] == 'bad-1'


@pytest.mark.parametrize('ids', [[None], ['duplicate', 'duplicate']])
def test_missing_or_duplicate_call_ids_stop_before_any_execution(ids):
    repo = MemoryRepository()
    model = FakeModel(AIMessage(content='', tool_calls=[call(id=id) for id in ids]))
    with pytest.raises(ValueError, match='IDs'):
        run(ChatService(repo, model, settings(), knowledge_search=FakeKnowledgeSearch(repo)))
    assert [row.role for row in repo.rows] == ['user', 'assistant']
    assert repo.keywords == [] and model.final_prompts == []


def test_selection_content_blocks_are_persisted_as_text_in_mysql(sessions):
    repo = Repository(sessions)
    cid = repo.create_conversation('guest-blocks')
    model = FakeModel(AIMessage(content=[{'type': 'text', 'text': '查询中'}], tool_calls=[call()]))
    run(ChatService(repo, model, settings(), knowledge_search=FakeKnowledgeSearch(repo)), id=cid)
    assert repo.load_messages(cid)[1].content == '查询中'


@pytest.mark.parametrize('phase', ['user', 'request', 'tool', 'final'])
@pytest.mark.parametrize('cancel_count', [1, 2])
def test_cancelled_persistence_settles_before_reservation_release_and_next_turn(phase, cancel_count):
    from threading import Event
    from app.history import completed_turns

    async def scenario():
        entered = asyncio.Event()
        release_worker = Event()
        settled = Event()
        loop = asyncio.get_running_loop()
        reservation = {'held': True}

        class DelayedRepository(MemoryRepository):
            def append_message(self, id, role, content, tool_calls=None, tool_call_id=None):
                actual_phase = ('user' if role == 'user' else 'tool' if role == 'tool'
                                else 'request' if tool_calls else 'final')
                if actual_phase == phase and not entered.is_set():
                    loop.call_soon_threadsafe(entered.set)
                    if not release_worker.wait(timeout=5):
                        raise RuntimeError('Test did not release its worker')
                    super().append_message(id, role, content, tool_calls, tool_call_id)
                    settled.set()
                else:
                    super().append_message(id, role, content, tool_calls, tool_call_id)

        repo = DelayedRepository()
        selection = AIMessage(content='', tool_calls=[call()]) if phase in ('request', 'tool') else None
        model = FakeModel(selection, chunks=('old-answer',))

        async def reserved_turn():
            try:
                return [event async for event in ChatService(repo, model, settings(), knowledge_search=FakeKnowledgeSearch(repo)).stream_turn(42, 'old-question')]
            finally:
                reservation['held'] = False

        task = asyncio.create_task(reserved_turn())
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            for _ in range(cancel_count):
                task.cancel()
                done, _ = await asyncio.wait({task}, timeout=0.02)
                assert task not in done, 'The turn exited while its database worker was still writing'
                assert reservation['held'] is True
                assert settled.is_set() is False
        finally:
            release_worker.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert settled.is_set() and reservation['held'] is False
        saved = copy.deepcopy(repo.rows)
        next_model = FakeModel(chunks=('new-answer',))
        await collect_next(ChatService(repo, next_model, settings(), knowledge_search=FakeKnowledgeSearch(repo)))
        assert repo.rows[:len(saved)] == saved
        expected = [('old-question', 'old-answer'), ('new-question', 'new-answer')] if phase == 'final' else [('new-question', 'new-answer')]
        assert [(turn[0].content, turn[-1].content) for turn in completed_turns(repo.rows)] == expected
        assert [m.content for m in next_model.selection_prompts[0][1:]] == (
            ['old-question', 'old-answer', 'new-question'] if phase == 'final' else ['new-question'])

    async def collect_next(service):
        return [event async for event in service.stream_turn(42, 'new-question')]

    asyncio.run(scenario())


def test_faq_timeout_chat_handoff_is_grounded_safe_json(monkeypatch):
    import time
    from app.tool_executor import ToolExecutor
    class SlowSearch:
        def search(self, keyword, *, limit=5):
            time.sleep(.08)
            return []
    monkeypatch.setattr('app.chat_service.ToolExecutor', lambda tools: ToolExecutor(tools, timeout_seconds=.01))
    repo = MemoryRepository()
    model = FakeModel(AIMessage(content='', tool_calls=[call(args={'keyword': '邮费'})]))
    events = run(ChatService(repo, model, settings(), SlowSearch()), '邮费是多少')
    assert events[1].data == {'name': 'query_faq', 'state': 'error'}
    payload = json.loads(model.final_prompts[0][-1].content)
    assert set(payload) == {'keyword', 'matches', 'message'}
    assert payload['keyword'] == '邮费' and payload['matches'] == []
    assert '暂时不可用' in payload['message']
    assert json.loads(repo.rows[2].content) == payload
