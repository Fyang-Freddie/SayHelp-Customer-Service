"""Real MySQL idempotent history, atomic reply, and ticket confirmations."""
import importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import SQLAlchemyError

from app.db import Conversation, Message, Ticket, make_session_factory
from app.repository import Repository
from test_db import database_url


@pytest.fixture
def sessions(database_url):
    from app.init_ch04_db import initialize_ch04_database
    initialize_ch04_database(database_url)
    try:
        initialize = importlib.import_module('app.init_ch05_db').initialize_ch05_database
    except ModuleNotFoundError:
        initialize = None
    if initialize is not None:
        initialize(database_url)
    factory = make_session_factory(database_url)
    yield factory
    factory.kw['bind'].dispose()


def workflow(sessions):
    try:
        return importlib.import_module('app.workflow_repository').WorkflowRepository(sessions)
    except ModuleNotFoundError:
        pytest.fail('WorkflowRepository is missing')


def test_append_once_retry_after_written_message_and_payload_conflict(sessions):
    cid = Repository(sessions).create_conversation('history')
    repo = workflow(sessions)
    mid = repo.append_message_once(cid, 'turn-1', 0, 'assistant', None, tool_calls=[{'id': 'call-1'}])
    assert repo.append_message_once(cid, 'turn-1', 0, 'assistant', None, tool_calls=[{'id': 'call-1'}]) == mid
    for changes in ({'content': 'different'}, {'tool_calls': []}, {'role': 'user'}):
        args = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call-1'}]} | changes
        with pytest.raises(ValueError, match='[Cc]onflict'):
            repo.append_message_once(cid, 'turn-1', 0, **args)
    assert len(Repository(sessions).load_messages(cid)) == 1


def test_reply_and_actions_are_stable_and_committed_together(sessions):
    cid = Repository(sessions).create_conversation('reply')
    repo = workflow(sessions)
    suggestions = [{'kind': 'handoff'}, {'kind': 'create_ticket', 'description': '商品破损', 'ticket_type': '售后'}]
    mid, actions = repo.commit_reply(cid, 'reply-turn', 1, '收到[1]', [{'n': 1}], suggestions)
    assert repo.commit_reply(cid, 'reply-turn', 1, '收到[1]', [{'n': 1}], suggestions) == (mid, actions)
    assert repo.load_actions(cid) == {mid: actions}
    assert [a['kind'] for a in actions] == ['handoff', 'create_ticket']
    assert len({a['action_id'] for a in actions}) == 2
    assert actions[1]['description'] == '商品破损' and actions[1]['ticket_type'] == '售后'
    for answer, citations, suggested in [('changed', [{'n': 1}], suggestions), ('收到[1]', [], suggestions), ('收到[1]', [{'n': 1}], [])]:
        with pytest.raises(ValueError, match='[Cc]onflict'):
            repo.commit_reply(cid, 'reply-turn', 1, answer, citations, suggested)
    with sessions() as session:
        turn = session.execute(text('SELECT status,final_message_id FROM workflow_turns WHERE turn_id=:tid'), {'tid': 'reply-turn'}).one()
        assert tuple(turn) == ('complete', int(mid))
        assert session.get(Message, int(mid)).citations == [{'n': 1}]
    repo.set_turn_status('reply-turn', 'checkpoint_failed')
    with sessions() as session:
        assert session.execute(text("SELECT status FROM workflow_turns WHERE turn_id='reply-turn'")).scalar_one() == 'checkpoint_failed'
    with pytest.raises(KeyError):
        repo.set_turn_status('missing', 'failed')


def test_database_failure_leaves_no_reply_or_half_suggestions(sessions):
    cid = Repository(sessions).create_conversation('atomic')
    repo = workflow(sessions)
    engine = sessions.kw['bind']
    def fail_action_insert(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith('INSERT INTO workflow_actions'):
            raise SQLAlchemyError('injected database failure')
    event.listen(engine, 'before_cursor_execute', fail_action_insert)
    try:
        with pytest.raises(SQLAlchemyError):
            repo.commit_reply(cid, 'atomic-turn', 0, 'answer', [], [{'kind': 'handoff'}, {'kind': 'create_ticket', 'description': 'bad', 'ticket_type': '咨询'}])
    finally:
        event.remove(engine, 'before_cursor_execute', fail_action_insert)
    assert Repository(sessions).load_messages(cid) == []
    assert repo.load_actions(cid) == {}
    repo.commit_reply(cid, 'atomic-turn', 0, 'answer', [], [{'kind': 'handoff'}])
    assert len(Repository(sessions).load_messages(cid)) == 1


def test_turn_ownership_and_deleted_conversation_reject_writes(sessions):
    base = Repository(sessions)
    cid, other = base.create_conversation('owner'), base.create_conversation('other')
    repo = workflow(sessions)
    repo.append_message_once(cid, 'owned-turn', 0, 'user', 'first')
    with pytest.raises(ValueError, match='[Cc]onflict'):
        repo.append_message_once(other, 'owned-turn', 1, 'user', 'intruder')
    base.soft_delete_conversation(cid)
    for write in (lambda: repo.append_message_once(cid, 'owned-turn', 0, 'user', 'first'), lambda: repo.commit_reply(cid, 'owned-turn', 1, 'answer', [], []), lambda: base.create_ticket(cid, 'desc', '咨询', request_key='deleted'), lambda: repo.set_turn_status('owned-turn', 'complete')):
        with pytest.raises(KeyError):
            write()
    assert base.load_messages(other) == []


def test_repeated_and_concurrent_confirmations_return_one_ticket(sessions):
    repo = Repository(sessions)
    cid = repo.create_conversation('concurrent')
    barrier = Barrier(6)
    def confirm(_):
        barrier.wait(timeout=10)
        return repo.create_ticket(cid, '重复确认', '投诉', request_key='confirm-one')
    with ThreadPoolExecutor(max_workers=6) as pool:
        numbers = list(pool.map(confirm, range(6)))
    assert len(set(numbers)) == 1
    assert repo.create_ticket(cid, '重复确认', '投诉', request_key='confirm-one') == numbers[0]
    assert repo.get_conversation(cid).status == '进行中'
    for description, kind in [('改描述', '投诉'), ('重复确认', '咨询')]:
        with pytest.raises(ValueError, match='[Cc]onflict'):
            repo.create_ticket(cid, description, kind, request_key='confirm-one')
    other = repo.create_conversation('other')
    with pytest.raises(ValueError, match='[Cc]onflict'):
        repo.create_ticket(other, '重复确认', '投诉', request_key='confirm-one')
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(Ticket)) == 1
        assert session.execute(text('SELECT COUNT(*) FROM workflow_ticket_requests')).scalar_one() == 1


def test_ticket_failure_rolls_back_request_and_ticket(sessions):
    repo = Repository(sessions)
    cid = repo.create_conversation('ticket-atomic')
    engine = sessions.kw['bind']
    def fail_request_insert(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith('INSERT INTO workflow_ticket_requests'):
            raise SQLAlchemyError('injected request failure')
    event.listen(engine, 'before_cursor_execute', fail_request_insert)
    try:
        with pytest.raises(SQLAlchemyError):
            repo.create_ticket(cid, 'retry', '咨询', request_key='retry-request')
    finally:
        event.remove(engine, 'before_cursor_execute', fail_request_insert)
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(Ticket)) == 0
    number = repo.create_ticket(cid, 'retry', '咨询', request_key='retry-request')
    assert repo.create_ticket(cid, 'retry', '咨询', request_key='retry-request') == number



def test_cross_conversation_request_key_race_rolls_back_loser(sessions):
    """Two distinct conversation locks contend on the real request-key constraint."""
    from threading import local
    repo = Repository(sessions)
    ids = [repo.create_conversation('race-a'), repo.create_conversation('race-b')]
    barrier = Barrier(2)
    seen = local()
    duplicates = []
    engine = sessions.kw['bind']
    def synchronize_missing_key(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith('SELECT workflow_ticket_requests.') and not getattr(seen, 'read', False):
            seen.read = True
            barrier.wait(timeout=10)
    def capture_duplicate(context):
        if getattr(context.original_exception, 'args', (None,))[0] == 1062:
            duplicates.append(True)
    def confirm(cid):
        try:
            return repo.create_ticket(cid, 'same description', '咨询', request_key='shared-key')
        except ValueError as error:
            assert 'conflict' in str(error)
            return 'conflict'
    event.listen(engine, 'after_cursor_execute', synchronize_missing_key)
    event.listen(engine, 'handle_error', capture_duplicate)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(confirm, ids))
    finally:
        event.remove(engine, 'after_cursor_execute', synchronize_missing_key)
        event.remove(engine, 'handle_error', capture_duplicate)
    assert results.count('conflict') == 1
    assert duplicates == [True]
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(Ticket)) == 1
        assert session.execute(text('SELECT COUNT(*) FROM workflow_ticket_requests')).scalar_one() == 1


def test_concurrent_history_position_returns_same_message(sessions):
    cid = Repository(sessions).create_conversation('concurrent-history')
    repo = workflow(sessions)
    barrier = Barrier(4)
    def append(_):
        barrier.wait(timeout=10)
        return repo.append_message_once(cid, 'same-position', 0, 'user', 'same content')
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(append, range(4)))
    assert len(set(ids)) == 1
    assert len(Repository(sessions).load_messages(cid)) == 1
