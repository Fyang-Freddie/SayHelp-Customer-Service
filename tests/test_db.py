import os
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from app.config import Settings
from app.db import Base, Conversation, Faq, Message, Ticket, make_session_factory
from app.init_db import initialize_database
from app.repository import Repository


@pytest.fixture
def database_url():
    url = os.environ.get('TEST_DATABASE_URL')
    if not url:
        pytest.skip('Set TEST_DATABASE_URL to a disposable Docker MySQL admin connection')
    admin_url = make_url(url)
    name = 'sayhelp_test_' + uuid4().hex
    admin = create_engine(admin_url)
    with admin.begin() as connection:
        connection.exec_driver_sql(f'CREATE DATABASE `{name}` CHARACTER SET utf8mb4')
    target = admin_url.set(database=name).render_as_string(hide_password=False)
    try:
        yield target
    finally:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'DROP DATABASE `{name}`')
        admin.dispose()


@pytest.fixture
def sessions(database_url):
    initialize_database(database_url)
    factory = make_session_factory(database_url)
    yield factory
    factory.kw['bind'].dispose()


def test_database_url_required_and_hidden(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key, value in {'CHAT_BASE_URL': 'https://example.test/v1', 'CHAT_MODEL': 'test', 'CHAT_API_KEY': 'secret'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('DATABASE_URL', raising=False)
    with pytest.raises(ValueError, match='DATABASE_URL'):
        Settings.from_env()
    monkeypatch.setenv('DATABASE_URL', 'mysql+pymysql://u:private-marker@localhost/db')
    settings = Settings.from_env()
    assert settings.database_url.endswith('/db')
    assert 'private-marker' not in repr(settings)


def test_mysql_schema_all_columns_types_indexes_and_foreign_keys(sessions):
    engine = sessions.kw['bind']
    expected = {
        'conversations': {'id': ('bigint unsigned', 'NO'), 'user_id': ('varchar(64)', 'NO'), 'status': ("enum('进行中','已转人工','已结束')", 'NO'), 'created_at': ('datetime', 'NO'), 'updated_at': ('datetime', 'NO')},
        'messages': {'id': ('bigint unsigned', 'NO'), 'conversation_id': ('bigint unsigned', 'NO'), 'role': ("enum('user','assistant','tool')", 'NO'), 'content': ('text', 'YES'), 'tool_calls': ('json', 'YES'), 'tool_call_id': ('varchar(64)', 'YES'), 'created_at': ('datetime', 'NO')},
        'faq': {'id': ('bigint unsigned', 'NO'), 'question': ('varchar(512)', 'NO'), 'answer': ('text', 'NO'), 'category': ('varchar(64)', 'NO'), 'created_at': ('datetime', 'NO'), 'updated_at': ('datetime', 'NO')},
        'tickets': {'ticket_no': ('varchar(32)', 'NO'), 'conversation_id': ('bigint unsigned', 'NO'), 'description': ('text', 'NO'), 'ticket_type': ("enum('售后','投诉','咨询')", 'NO'), 'status': ("enum('待处理','已处理')", 'NO'), 'created_at': ('datetime', 'NO')},
    }
    with engine.connect() as connection:
        for table, columns in expected.items():
            actual = connection.execute(text('SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT, EXTRA FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:table'), {'table': table}).all()
            assert {r[0]: (r[1], r[2]) for r in actual} == columns
            mapped = Base.metadata.tables[table]
            assert {col.name: (col.type.compile(dialect=engine.dialect).lower(), 'YES' if col.nullable else 'NO') for col in mapped.columns} == columns
            assert {index.name for index in mapped.indexes} == {item['name'] for item in inspect(engine).get_indexes(table)}
            details = {r[0]: r for r in actual}
            if table != 'tickets':
                assert 'auto_increment' in details['id'][4]
            assert details['created_at'][3].lower() == 'current_timestamp'
            if 'updated_at' in columns:
                assert 'on update CURRENT_TIMESTAMP'.lower() in details['updated_at'][4].lower()
            shape = connection.execute(text('SELECT ENGINE,TABLE_COLLATION FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:table'), {'table': table}).one()
            assert shape[0] == 'InnoDB'
            assert shape[1].startswith('utf8mb4')
    inspector = inspect(engine)
    for table, index in [('conversations', 'idx_user_id'), ('messages', 'idx_conversation_id'), ('faq', 'idx_category'), ('tickets', 'idx_conversation_id')]:
        assert index in {item['name'] for item in inspector.get_indexes(table)}
    for table in ('messages', 'tickets'):
        fk = inspector.get_foreign_keys(table)[0]
        assert fk['constrained_columns'] == ['conversation_id']
        assert fk['referred_table'] == 'conversations'
        assert fk['referred_columns'] == ['id']
    assert inspector.get_pk_constraint('tickets')['constrained_columns'] == ['ticket_no']
    with engine.connect() as connection:
        defaults = connection.execute(text("SELECT TABLE_NAME,COLUMN_DEFAULT FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='status'")).all()
        assert dict(defaults) == {'conversations': '进行中', 'tickets': '待处理'}


def test_conversation_and_protocol_message_round_trip(sessions):
    repo = Repository(sessions)
    cid = repo.create_conversation('guest-demo')
    conversation = repo.get_conversation(cid)
    assert conversation.user_id == 'guest-demo'
    assert conversation.status == '进行中'
    assert conversation.created_at is not None
    assert repo.get_conversation(cid + 1000) is None
    calls = [{'id': 'call_1', 'name': 'query_faq', 'args': {'keyword': '退货'}}]
    repo.append_message(cid, 'user', '退货政策是什么')
    repo.append_message(cid, 'assistant', None, calls)
    repo.append_message(cid, 'tool', '可以申请退货', tool_call_id='call_1')
    rows = repo.load_messages(cid)
    assert [row.role for row in rows] == ['user', 'assistant', 'tool']
    assert rows[1].content is None
    assert rows[1].tool_calls == calls
    assert rows[2].tool_call_id == 'call_1'
    assert all(row.created_at is not None for row in rows)


def test_faq_query_is_literal_question_only_and_bounded(sessions):
    with sessions.begin() as session:
        session.add_all([Faq(question=f'测试退货 {i}', answer='answer', category='测试') for i in range(8)])
        session.add_all([Faq(question='100%_折扣', answer='answer', category='测试'), Faq(question='其他问题', answer='仅答案含秘密词', category='测试')])
    repo = Repository(sessions)
    assert len(repo.find_faq('退货')) == 5
    assert len(repo.find_faq('退货', limit=2)) == 2
    assert repo.find_faq('邮费') == []
    assert repo.find_faq('秘密词') == []
    assert [row.question for row in repo.find_faq('%_')] == ['100%_折扣']
    assert repo.find_faq("' OR 1=1 --") == []
    assert repo.find_faq('') == []
    with pytest.raises(ValueError):
        repo.find_faq('退货', limit=0)


def test_ticket_insert_updates_conversation_atomically(sessions):
    repo = Repository(sessions)
    cid = repo.create_conversation('guest-ticket')
    numbers = [repo.create_ticket(cid, '商品破损，请联系我', '售后') for _ in range(2)]
    assert len(set(numbers)) == 2
    assert all(len(number) <= 32 for number in numbers)
    with sessions() as session:
        ticket = session.get(Ticket, numbers[0])
        assert (ticket.conversation_id, ticket.description, ticket.ticket_type, ticket.status) == (cid, '商品破损，请联系我', '售后', '待处理')
    assert repo.get_conversation(cid).status == '已转人工'
    other = repo.create_conversation('guest-rollback')
    with pytest.raises((ValueError, IntegrityError)):
        repo.create_ticket(other, 'bad type', '退货')
    assert repo.get_conversation(other).status == '进行中'
    with pytest.raises(ValueError):
        repo.create_ticket(cid + 1000, 'missing conversation', '咨询')


def test_mysql_rejects_orphan_messages_and_tickets(sessions):
    with pytest.raises(IntegrityError):
        with sessions.begin() as session:
            session.add(Message(conversation_id=999999, role='user', content='orphan'))
    with pytest.raises(IntegrityError):
        with sessions.begin() as session:
            session.add(Ticket(ticket_no='orphan', conversation_id=999999, description='orphan', ticket_type='投诉'))


def test_schema_and_faq_seed_rerun_are_idempotent(database_url):
    initialize_database(database_url)
    initialize_database(database_url)
    factory = make_session_factory(database_url)
    try:
        with factory() as session:
            rows = session.scalars(select(Faq)).all()
            assert rows
            assert len(rows) == len({row.question for row in rows})
            assert any('退货' in row.question for row in rows)
            assert any('运费' in row.question for row in rows)
            assert all('邮费' not in row.question for row in rows)
    finally:
        factory.kw['bind'].dispose()


def test_partial_schema_is_rejected_without_changing_existing_table(database_url):
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.exec_driver_sql('CREATE TABLE conversations (id BIGINT PRIMARY KEY)')
        connection.exec_driver_sql('INSERT INTO conversations VALUES (123)')
    try:
        with pytest.raises(ValueError, match='[Pp]artial schema'):
            initialize_database(database_url)
        with engine.connect() as connection:
            assert connection.exec_driver_sql('SELECT id FROM conversations').scalar_one() == 123
        assert inspect(engine).get_table_names() == ['conversations']
    finally:
        engine.dispose()
