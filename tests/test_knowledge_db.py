import os
import re
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.dialects import mysql
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.db import Conversation, Faq, Message
from app.init_db import initialize_database
from app.knowledge_db import KnowledgeChunk, QaExtractionStaging
from app.init_knowledge_db import initialize_knowledge_database

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'docs/superpowers/specs/2026-10-04-sayhelp-ch03-dense-rag-design.md'
DDL = SPEC.read_text(encoding='utf-8').split('```sql\n', 1)[1].split('```', 1)[0].strip()


class PrivateDatabaseUrl(str):
    def __repr__(self):
        return '<disposable MySQL test database>'


@pytest.fixture
def database_url():
    url = os.environ.get('TEST_DATABASE_URL')
    if not url:
        pytest.skip('Set TEST_DATABASE_URL to a disposable MySQL admin connection')
    admin_url = make_url(url)
    name = 'sayhelp_ch03_test_' + uuid4().hex
    admin = create_engine(admin_url)
    with admin.begin() as connection:
        connection.exec_driver_sql(f'CREATE DATABASE `{name}` CHARACTER SET utf8mb4')
    try:
        yield PrivateDatabaseUrl(admin_url.set(database=name).render_as_string(hide_password=False))
    finally:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'DROP DATABASE `{name}`')
        admin.dispose()


def supplied_columns(table_name):
    body = DDL.split(f'CREATE TABLE {table_name} (', 1)[1].split('  PRIMARY KEY', 1)[0]
    columns = {}
    for line in body.strip().splitlines():
        match = re.fullmatch(r"\s*(\w+)\s+(.+?)\s+(NOT NULL|NULL)\s*(.*?)\s+COMMENT '([^']*)',?", line)
        assert match, line
        name, type_name, nullable, extra, comment = match.groups()
        default = re.search(r"DEFAULT ('[^']*'|\w+)", extra)
        columns[name] = (type_name.lower(), 'YES' if nullable == 'NULL' else 'NO',
                         default.group(1).strip("'").lower() if default else None, extra, comment)
    return columns


def test_ch03_ddl_and_mappings_match_supplied_definitions():
    assert (ROOT / 'db/ch03_schema.sql').read_text(encoding='utf-8').strip() == DDL
    assert DDL.startswith('SET NAMES utf8mb4;')
    dialect = mysql.dialect()
    for model in (KnowledgeChunk, QaExtractionStaging):
        table = model.__table__
        columns = supplied_columns(table.name)
        assert set(table.columns.keys()) - {'source_file','source_start_line','source_end_line','product_category','source_digest'} == set(columns)
        for column in (table.c[name] for name in columns):
            expected_type, nullable, default, extra, comment = columns[column.name]
            assert column.type.compile(dialect=dialect).lower().replace(', ', ',') == expected_type
            assert ('YES' if column.nullable else 'NO') == nullable
            assert column.comment == comment
            actual_default = str(column.server_default.arg).strip("'").lower() if column.server_default else None
            expected_default = default
            if 'ON UPDATE' in extra:
                expected_default += ' on update current_timestamp'
            assert actual_default == expected_default
        assert table.dialect_options['mysql']['engine'] == 'InnoDB'
        assert table.dialect_options['mysql']['charset'] == 'utf8mb4'
        assert table.columns.id.primary_key and table.columns.id.autoincrement
    assert {(fk.name, fk.parent.name, fk.target_fullname, fk.ondelete)
            for fk in KnowledgeChunk.__table__.foreign_keys} == {
        ('fk_chunks_prev', 'prev_chunk_id', 'knowledge_chunks.id', 'SET NULL'),
        ('fk_chunks_next', 'next_chunk_id', 'knowledge_chunks.id', 'SET NULL'),
    }


def test_ch03_schema_exactly_matches_supplied_ddl(database_url):
    initialize_knowledge_database(database_url)
    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        with engine.connect() as connection:
            for model in (KnowledgeChunk, QaExtractionStaging):
                table = model.__tablename__
                columns = supplied_columns(table)
                actual = connection.execute(text('SELECT COLUMN_NAME,COLUMN_TYPE,IS_NULLABLE,COLUMN_DEFAULT,EXTRA,COLUMN_COMMENT FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:table'), {'table': table}).all()
                assert set(row[0] for row in actual) == set(columns)
                for name, kind, nullable, default, extra, comment in actual:
                    expected_kind, expected_nullable, expected_default, expected_extra, expected_comment = columns[name]
                    assert (kind.lower(), nullable, str(default).lower() if default is not None else None, comment) == (expected_kind, expected_nullable, expected_default, expected_comment)
                    assert ('auto_increment' in extra.lower()) == ('AUTO_INCREMENT' in expected_extra)
                    assert ('on update current_timestamp' in extra.lower()) == ('ON UPDATE CURRENT_TIMESTAMP' in expected_extra)
                shape = connection.execute(text('SELECT ENGINE,TABLE_COLLATION,TABLE_COMMENT FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:table'), {'table': table}).one()
                assert shape[0] == 'InnoDB' and shape[1].startswith('utf8mb4')
                assert shape[2] == model.__table__.comment
                expected_indexes = {(index.name, tuple(col.name for col in index.columns)) for index in model.__table__.indexes}
                implicit_fk_indexes = {('fk_chunks_prev', ('prev_chunk_id',)), ('fk_chunks_next', ('next_chunk_id',))} if table == 'knowledge_chunks' else set()
                assert expected_indexes | implicit_fk_indexes == {(index['name'], tuple(index['column_names'])) for index in inspector.get_indexes(table)}
                assert expected_indexes == ({('idx_category', ('category',)), ('idx_vectorize_status', ('vectorize_status',))} if table == 'knowledge_chunks' else {('idx_batch_no', ('batch_no',)), ('idx_status', ('status',))})
                assert inspector.get_pk_constraint(table)['constrained_columns'] == ['id']
        assert {(fk['name'], tuple(fk['constrained_columns']), fk['referred_table'], tuple(fk['referred_columns']), fk['options']['ondelete']) for fk in inspector.get_foreign_keys('knowledge_chunks')} == {
            ('fk_chunks_prev', ('prev_chunk_id',), 'knowledge_chunks', ('id',), 'SET NULL'),
            ('fk_chunks_next', ('next_chunk_id',), 'knowledge_chunks', ('id',), 'SET NULL'),
        }
        assert inspector.get_foreign_keys('qa_extraction_staging') == []
    finally:
        engine.dispose()


def test_ch03_initializer_is_additive_and_repeatable(database_url):
    initialize_knowledge_database(database_url)
    engine = create_engine(database_url)
    try:
        with Session(engine) as session, session.begin():
            conversation = Conversation(user_id='preserved')
            session.add(conversation)
            session.flush()
            session.add(Message(conversation_id=conversation.id, role='user', content='preserved message'))
            session.add(KnowledgeChunk(category='配送', questions='邮费是多少', answer='按结算页计算'))
            session.add(QaExtractionStaging(batch_no='batch-1', question='问题', answer='答案'))
        initialize_knowledge_database(database_url)
        initialize_database(database_url)
        with Session(engine) as session:
            assert session.scalar(select(Conversation.user_id)) == 'preserved'
            assert session.scalar(select(Message.content)) == 'preserved message'
            faq = session.scalars(select(Faq)).all()
            assert len(faq) == 3
            chunk = session.scalar(select(KnowledgeChunk))
            assert chunk.vectorize_status == 'pending' and chunk.is_key_clause == 0
            assert chunk.created_at and chunk.updated_at
            assert session.scalar(select(QaExtractionStaging.status)) == 'extracted'
        with engine.begin() as connection:
            connection.exec_driver_sql('DROP TABLE qa_extraction_staging')
        with pytest.raises(ValueError, match='[Pp]artial.*schema'):
            initialize_knowledge_database(database_url)
        assert set(inspect(engine).get_table_names()) == {'conversations', 'messages', 'faq', 'tickets', 'knowledge_chunks'}
        with Session(engine) as session:
            assert session.scalar(select(KnowledgeChunk.answer)) == '按结算页计算'
    finally:
        engine.dispose()


def test_ch03_initializer_rejects_non_mysql():
    with pytest.raises(ValueError, match='MySQL'):
        initialize_knowledge_database('sqlite://')
