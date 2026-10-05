"""Chapter 4 contract tests use the user DDL and a disposable real MySQL database."""
import importlib
import re
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import Session

from test_db import database_url

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'docs/superpowers/specs/2026-10-05-sayhelp-ch04-hybrid-rag-design.md'
DDL = SPEC.read_text(encoding='utf-8').split('```SQL\n', 1)[1].split('```', 1)[0].strip()


def initializer():
    try:
        module = importlib.import_module('app.init_ch04_db')
    except ModuleNotFoundError:
        pytest.fail('Chapter 4 initializer is missing')
    assert callable(getattr(module, 'initialize_ch04_database', None))
    return module.initialize_ch04_database


def supplied_columns(name):
    body = DDL.split('CREATE TABLE ' + name + ' (', 1)[1].split('  PRIMARY KEY', 1)[0]
    result = {}
    for line in body.strip().splitlines():
        match = re.fullmatch(r"\s*(\w+)\s+(.+?)\s+(NOT NULL|NULL)\s*(.*?)\s+COMMENT '([^']*)',?", line)
        assert match, line
        column, kind, nullable, extra, comment = match.groups()
        default = re.search(r"DEFAULT ('[^']*'|\w+)", extra)
        result[column] = (kind.lower(), 'YES' if nullable == 'NULL' else 'NO',
                          default.group(1).strip("'") if default else None, comment,
                          'AUTO_INCREMENT' in extra)
    return result


def test_ch04_sql_retains_the_exact_user_contract():
    path = ROOT / 'db/ch04_schema.sql'
    assert path.is_file(), 'Supplied Chapter 4 DDL has not been saved'
    assert path.read_text(encoding='utf-8').strip() == DDL


def test_ch04_mappings_match_every_supplied_column():
    initializer()
    from app.ch04_db import LowConfidenceQuestion, FaithCase
    for model in (LowConfidenceQuestion, FaithCase):
        expected = supplied_columns(model.__tablename__)
        assert set(model.__table__.columns.keys()) == set(expected)
        for column in model.__table__.columns:
            kind, nullable, default, comment, auto = expected[column.name]
            assert column.type.compile(dialect=mysql.dialect()).lower().replace(', ', ',').replace('integer', 'int') == kind
            assert ('YES' if column.nullable else 'NO') == nullable
            actual = str(column.server_default.arg).strip("'") if column.server_default else None
            assert (actual.lower() if actual else None) == (default.lower() if default else None)
            assert column.comment == comment
            if auto:
                assert column.primary_key and column.autoincrement


def test_real_mysql_schema_and_unicode_enum_json(database_url):
    initializer()(database_url)
    from app.ch04_db import FaithCase, LowConfidenceQuestion
    from app.db import Conversation
    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        with engine.connect() as connection:
            for name in ('low_confidence_questions', 'faith_cases'):
                rows = connection.execute(text('SELECT COLUMN_NAME,COLUMN_TYPE,IS_NULLABLE,COLUMN_DEFAULT,COLUMN_COMMENT,EXTRA FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:name'), {'name': name}).all()
                actual = {name: (kind, nullable, default, comment, 'auto_increment' in extra) for name,kind,nullable,default,comment,extra in rows}
                assert actual == supplied_columns(name)
                shape = connection.execute(text('SELECT ENGINE,TABLE_COLLATION FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:name'), {'name': name}).one()
                assert shape[0] == 'InnoDB' and shape[1].startswith('utf8mb4')
            assert {(i['name'], tuple(i['column_names']), bool(i['unique'])) for i in inspector.get_indexes('faith_cases')} == {('uk_eval_id', ('eval_id',), True), ('idx_status', ('status',), False), ('idx_last_seen_at', ('last_seen_at',), False)}
            fk = inspector.get_foreign_keys('low_confidence_questions')
            assert len(fk) == 1 and fk[0]['name'] == 'fk_lcq_conversation' and fk[0]['referred_table'] == 'conversations'
        cid = 9007199254740997
        with Session(engine) as session, session.begin():
            session.add(Conversation(id=cid, user_id='ch04-unicode'))
            session.flush()
            session.add(LowConfidenceQuestion(conversation_id=cid, raw_question='这东西咋弄啊？', source='self_check', reason='证据未覆盖型号'))
            session.add(FaithCase(eval_id='E01', bucket='E_multi', query='问题', answer='答案[1]', reason='无证据', citations=[{'n': 1, 'chunk_id': str(cid), 'answer': '中文原文'}]))
        with Session(engine) as session:
            low = session.scalar(select(LowConfidenceQuestion))
            case = session.scalar(select(FaithCase))
            assert (low.raw_question, low.source, low.conversation_id) == ('这东西咋弄啊？', 'self_check', cid)
            assert case.status == '未解决' and case.seen_count == 1
            assert case.citations == [{'n': 1, 'chunk_id': str(cid), 'answer': '中文原文'}]
            assert low.created_at and case.first_seen_at and case.last_seen_at
    finally:
        engine.dispose()


def test_additive_migration_preserves_old_rows_and_can_repeat(database_url):
    from app.init_knowledge_db import initialize_knowledge_database
    initialize_knowledge_database(database_url)
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            cid = connection.exec_driver_sql("INSERT INTO conversations (user_id) VALUES ('old-row')").lastrowid
            connection.execute(text("INSERT INTO messages (conversation_id,role,content) VALUES (:cid,'user','old question')"), {'cid': cid})
            connection.exec_driver_sql("INSERT INTO knowledge_chunks (category,questions,answer,vectorize_status) VALUES ('old','q','a','done')")
        initialize = initializer()
        initialize(database_url)
        initialize(database_url)
        with engine.connect() as connection:
            old = connection.execute(text('SELECT is_pinned,pinned_at,deleted_at FROM conversations WHERE id=:cid'), {'cid': cid}).one()
            assert tuple(old) == (0, None, None)
            assert connection.exec_driver_sql('SELECT citations FROM messages').scalar() is None
            chunk = connection.exec_driver_sql('SELECT source_file,source_start_line,source_end_line,product_category,source_digest,vectorize_status FROM knowledge_chunks').one()
            assert tuple(chunk) == (None,None,None,None,None,'done')
            assert connection.exec_driver_sql('SELECT COUNT(*) FROM messages').scalar() == 1
            assert connection.exec_driver_sql('SELECT COUNT(*) FROM knowledge_index_states').scalar() == 0
    finally:
        engine.dispose()


@pytest.mark.parametrize('damage', ['partial', 'incompatible', 'migration', 'missing_both', 'enum_case', 'default_case', 'fk_cascade'])
def test_incompatible_schema_is_rejected_before_modifying_data(database_url, damage):
    initialize = initializer()
    initialize(database_url)
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("INSERT INTO conversations (user_id) VALUES ('preserved')")
            if damage == 'partial':
                connection.exec_driver_sql('DROP TABLE faith_cases')
            elif damage == 'incompatible':
                connection.exec_driver_sql('ALTER TABLE faith_cases MODIFY eval_id VARCHAR(32) NOT NULL')
            elif damage == 'migration':
                connection.exec_driver_sql('ALTER TABLE knowledge_chunks MODIFY source_digest VARCHAR(12) NULL')
            elif damage == 'enum_case':
                connection.exec_driver_sql("ALTER TABLE low_confidence_questions MODIFY source ENUM('RETRIEVAL_LOW_CONF','SELF_CHECK','USER_FEEDBACK') NOT NULL COMMENT '入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决'")
            elif damage == 'fk_cascade':
                connection.exec_driver_sql('ALTER TABLE low_confidence_questions DROP FOREIGN KEY fk_lcq_conversation')
                connection.exec_driver_sql('ALTER TABLE low_confidence_questions ADD CONSTRAINT fk_lcq_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE')
            elif damage == 'default_case':
                connection.exec_driver_sql("ALTER TABLE faith_cases ALTER COLUMN strategy SET DEFAULT 'HYBRID_RERANK'")
            else:
                connection.exec_driver_sql('DROP TABLE low_confidence_questions')
                connection.exec_driver_sql('DROP TABLE faith_cases')
        with pytest.raises(ValueError, match='[Pp]artial|[Ii]ncompatible'):
            initialize(database_url)
        with engine.connect() as connection:
            assert connection.exec_driver_sql('SELECT user_id FROM conversations').scalar_one() == 'preserved'
    finally:
        engine.dispose()


def test_non_mysql_rejected():
    with pytest.raises(ValueError, match='MySQL'):
        initializer()('sqlite://')


@pytest.mark.parametrize('payload', [{'unknown': 'x'}, {'product_category': ''}, {'category': 123}, {'content_type': None}, {'product_category': 'x' * 65}])
def test_metadata_filters_reject_invalid_or_unbounded_fields(payload):
    try:
        from app.knowledge_types import KnowledgeFilters
    except ModuleNotFoundError:
        pytest.fail('Validated metadata filters are missing')
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        KnowledgeFilters.model_validate(payload)


def test_metadata_filters_keep_safe_parameter_values():
    try:
        from app.knowledge_types import KnowledgeFilters
    except ModuleNotFoundError:
        pytest.fail('Validated metadata filters are missing')
    filters = KnowledgeFilters.model_validate({'product_category': ' 猫砂盆 ', 'category': 'a" OR true'})
    assert filters.model_dump(exclude_none=True) == {'product_category': '猫砂盆', 'category': 'a" OR true'}
