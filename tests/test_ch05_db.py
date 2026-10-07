"""Additive chapter 5 schema contract on isolated real MySQL databases."""
import importlib

import pytest
from sqlalchemy import create_engine, inspect, text

from test_db import database_url
from app.init_ch04_db import initialize_ch04_database

TABLES = {'workflow_turns', 'workflow_message_keys', 'workflow_actions', 'workflow_ticket_requests'}


def initializer():
    try:
        return importlib.import_module('app.init_ch05_db').initialize_ch05_database
    except ModuleNotFoundError:
        pytest.fail('Chapter 5 initializer is missing')


def test_additive_schema_repeats_preserving_existing_messages(database_url):
    initialize_ch04_database(database_url)
    engine = create_engine(database_url)
    try:
        before = set(inspect(engine).get_table_names())
        with engine.begin() as connection:
            cid = connection.exec_driver_sql("INSERT INTO conversations(user_id) VALUES ('preserved')").lastrowid
            connection.execute(text("INSERT INTO messages(conversation_id,role,content) VALUES (:cid,'user','旧消息')"), {'cid': cid})
        initializer()(database_url)
        initializer()(database_url)
        assert set(inspect(engine).get_table_names()) - before == TABLES
        with engine.connect() as connection:
            assert connection.exec_driver_sql('SELECT content FROM messages').scalar_one() == '旧消息'
        from app.ch05_db import WorkflowTurn, WorkflowMessageKey, WorkflowAction, WorkflowTicketRequest
        from app.init_ch05_db import _validate_workflow_table
        with engine.connect() as connection:
            for model in (WorkflowTurn, WorkflowMessageKey, WorkflowAction, WorkflowTicketRequest):
                _validate_workflow_table(connection, model)
    finally:
        engine.dispose()


@pytest.mark.parametrize('damage', ['partial', 'column', 'fk', 'index', 'options', 'ch04', 'staging', 'collation'])
def test_wrong_or_partial_schema_is_rejected_without_repair(database_url, damage):
    initialize_ch04_database(database_url)
    initializer()(database_url)
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("INSERT INTO conversations(user_id) VALUES ('preserved')")
            if damage == 'partial':
                connection.exec_driver_sql('DROP TABLE workflow_actions')
            elif damage == 'column':
                connection.exec_driver_sql('ALTER TABLE workflow_turns MODIFY status VARCHAR(99) NOT NULL')
            elif damage == 'fk':
                connection.exec_driver_sql('ALTER TABLE workflow_actions DROP FOREIGN KEY fk_workflow_action_message')
            elif damage == 'index':
                connection.exec_driver_sql('CREATE INDEX unexpected ON workflow_turns(status)')
            elif damage == 'options':
                connection.exec_driver_sql("ALTER TABLE workflow_turns COMMENT='wrong'")
            elif damage == 'staging':
                connection.exec_driver_sql('DROP TABLE qa_extraction_staging')
            elif damage == 'collation':
                connection.exec_driver_sql('ALTER TABLE workflow_ticket_requests MODIFY request_key VARCHAR(128) COLLATE utf8mb4_general_ci NOT NULL')
            else:
                connection.exec_driver_sql('ALTER TABLE messages MODIFY citations TEXT NULL')
        tables_before = set(inspect(engine).get_table_names())
        with pytest.raises(ValueError, match='[Pp]artial|[Ii]ncompatible'):
            initializer()(database_url)
        assert set(inspect(engine).get_table_names()) == tables_before
        with engine.connect() as connection:
            assert connection.exec_driver_sql('SELECT user_id FROM conversations').scalar_one() == 'preserved'
    finally:
        engine.dispose()


def test_ch04_required_before_any_ch05_ddl(database_url):
    engine = create_engine(database_url)
    try:
        with pytest.raises(ValueError, match='Chapter 4'):
            initializer()(database_url)
        assert not set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_non_mysql_rejected():
    with pytest.raises(ValueError, match='MySQL'):
        initializer()('sqlite://')


def test_real_schema_has_only_the_declared_metadata_fields(database_url):
    initialize_ch04_database(database_url)
    initializer()(database_url)
    engine = create_engine(database_url)
    expected = {
        'workflow_turns': ['turn_id', 'conversation_id', 'status', 'final_message_id'],
        'workflow_message_keys': ['turn_id', 'position', 'message_id'],
        'workflow_actions': ['action_id', 'conversation_id', 'turn_id', 'message_id', 'kind', 'description', 'ticket_type'],
        'workflow_ticket_requests': ['request_key', 'conversation_id', 'payload_digest', 'ticket_no'],
    }
    try:
        inspector = inspect(engine)
        for name, fields in expected.items():
            assert [column['name'] for column in inspector.get_columns(name)] == fields
        assert inspector.get_pk_constraint('workflow_message_keys')['constrained_columns'] == ['turn_id', 'position']
        assert inspector.get_pk_constraint('workflow_ticket_requests')['constrained_columns'] == ['request_key']
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT COLLATION_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='workflow_ticket_requests' AND COLUMN_NAME='request_key'").scalar_one() == 'utf8mb4_bin'
    finally:
        engine.dispose()


def test_initializer_module_command_repeats_on_disposable_mysql(database_url):
    import os
    import subprocess
    import sys
    initialize_ch04_database(database_url)
    environment = dict(os.environ, DATABASE_URL=database_url)
    for _ in range(2):
        result = subprocess.run([sys.executable, '-m', 'app.init_ch05_db'], env=environment,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == 'Chapter 5 additive workflow metadata schema verified.'
