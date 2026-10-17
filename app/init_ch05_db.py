"""Explicit, repeatable Chapter 5 initialization; never rebuild partial schemas."""
import os
import re
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.ch04_db import FaithCase, KnowledgeIndexState, LowConfidenceQuestion
from app.ch05_db import WorkflowAction, WorkflowMessageKey, WorkflowTicketRequest, WorkflowTurn
from app.db import Conversation, Faq, Message, Ticket
from app.init_ch04_db import (_preflight as ch04_preflight, _validate_table, sql_statements,
                              _columns, _column_shape, _fk_action)
from app.knowledge_db import KnowledgeChunk, QaExtractionStaging

ROOT = Path(__file__).resolve().parents[1]
MODELS = (WorkflowTurn, WorkflowMessageKey, WorkflowAction, WorkflowTicketRequest)


def _validate_workflow_table(connection, model):
    table = model.__table__
    expected = {}
    for column in table.columns:
        shape = _column_shape(column, connection.dialect)
        expected[column.name] = (re.sub(r' collate \"?[^ ]+\"?$', '', shape[0]), shape[1], shape[2], shape[3] or '', *shape[4:])
    if _columns(connection, table.name) != expected:
        raise ValueError(f'Incompatible Chapter 5 schema: {table.name} columns; repair explicitly')
    collations = dict(connection.execute(text('SELECT COLUMN_NAME,COLLATION_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:name'), {'name': table.name}).all())
    for column in table.columns:
        collation = getattr(column.type, 'collation', None)
        if collation is not None and collations[column.name] != collation:
            raise ValueError(f'Incompatible Chapter 5 schema: {table.name} identity collation')
    inspector = inspect(connection)
    expected_indexes = {(i.name, tuple(c.name for c in i.columns), i.unique) for i in table.indexes}
    pk = tuple(c.name for c in table.primary_key.columns)
    for constraint in table.foreign_key_constraints:
        cols = tuple(c.name for c in constraint.columns)
        if not any(idxcols[:len(cols)] == cols for _, idxcols, _ in expected_indexes) and pk[:len(cols)] != cols:
            expected_indexes.add((constraint.name, cols, False))
    indexes = {(i['name'], tuple(i['column_names']), bool(i['unique'])) for i in inspector.get_indexes(table.name)}
    expected_fks = {(c.name, tuple(col.name for col in c.columns), c.elements[0].column.table.name,
                     tuple(f.column.name for f in c.elements), _fk_action(c.ondelete), _fk_action(c.onupdate)) for c in table.foreign_key_constraints}
    fks = {(f['name'], tuple(f['constrained_columns']), f['referred_table'], tuple(f['referred_columns']),
            _fk_action(f.get('options', {}).get('ondelete')), _fk_action(f.get('options', {}).get('onupdate'))) for f in inspector.get_foreign_keys(table.name)}
    if indexes != expected_indexes or fks != expected_fks or inspector.get_pk_constraint(table.name)['constrained_columns'] != list(pk):
        raise ValueError(f'Incompatible Chapter 5 schema: {table.name} indexes or constraints')
    engine, collation, comment = connection.execute(text('SELECT ENGINE,TABLE_COLLATION,TABLE_COMMENT FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:name'), {'name': table.name}).one()
    if engine != 'InnoDB' or not collation.startswith('utf8mb4') or comment != table.comment:
        raise ValueError(f'Incompatible Chapter 5 schema: {table.name} table options')


def _preflight(connection):
    present = set(inspect(connection).get_table_names())
    required = (Conversation, Message, Faq, Ticket, KnowledgeChunk, QaExtractionStaging,
                LowConfidenceQuestion, FaithCase, KnowledgeIndexState)
    if not {m.__tablename__ for m in required} <= present:
        raise ValueError('Incompatible Chapter 4 schema: complete schema is required before Chapter 5 initialization')
    supplied, migrated = ch04_preflight(connection)
    if not supplied or not migrated:
        raise ValueError('Incompatible Chapter 4 schema: complete schema is required before Chapter 5 initialization')
    for model in required:
        _validate_table(connection, model)
    names = {model.__tablename__ for model in MODELS}
    if present & names and not names <= present:
        raise ValueError('Partial Chapter 5 schema; repair explicitly')
    for model in MODELS:
        if model.__tablename__ in present:
            _validate_workflow_table(connection, model)
    return names <= present


def initialize_ch05_database(database_url: str) -> None:
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        if engine.dialect.name != 'mysql':
            raise ValueError('Chapter 5 initialization requires MySQL')
        with engine.connect() as connection:
            lock_sql = "CONCAT('sayhelp_ch05_', MD5(DATABASE()))"
            locked = connection.execute(text(f'SELECT GET_LOCK({lock_sql}, 30)')).scalar()
            if locked != 1:
                raise ValueError('Chapter 5 initialization is already running')
            try:
                initialized = _preflight(connection)
                connection.commit()
                if not initialized:
                    for statement in sql_statements((ROOT / 'db/ch05_schema.sql').read_text(encoding='utf-8')):
                        connection.exec_driver_sql(statement)
                    connection.commit()
                if not _preflight(connection):
                    raise ValueError('Partial Chapter 5 initialization; repair explicitly')
            finally:
                try:
                    connection.rollback()
                    connection.execute(text(f'SELECT RELEASE_LOCK({lock_sql})'))
                    connection.commit()
                except Exception:
                    connection.invalidate()
                    raise
    finally:
        engine.dispose()


def main():
    values = dotenv_values(Path.cwd() / '.env')
    url = os.environ.get('DATABASE_URL', values.get('DATABASE_URL'))
    if not url or not url.strip():
        raise SystemExit('DATABASE_URL is required')
    try:
        initialize_ch05_database(url.strip())
    except ValueError as error:
        raise SystemExit(str(error)) from None
    except SQLAlchemyError:
        raise SystemExit('Chapter 5 initialization failed; inspect schema compatibility and MySQL availability') from None
    print('Chapter 5 additive workflow metadata schema verified.')


if __name__ == '__main__':
    main()
