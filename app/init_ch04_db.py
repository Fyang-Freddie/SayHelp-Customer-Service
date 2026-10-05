"""Explicit additive Chapter 4 initialization with strict schema preflight.

MySQL DDL commits implicitly. Partial migrations are rejected for explicit repair,
never masked by CREATE IF NOT EXISTS or destructive reconstruction.
"""
import os
import re
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.ch04_db import FaithCase, KnowledgeIndexState, LowConfidenceQuestion
from app.db import Conversation, Message
from app.knowledge_db import KnowledgeChunk
from app.init_knowledge_db import initialize_knowledge_database

ROOT = Path(__file__).resolve().parents[1]
MODELS = (LowConfidenceQuestion, FaithCase, KnowledgeIndexState)
ADDITIONS = {
    Conversation.__table__: ('is_pinned', 'pinned_at', 'deleted_at'),
    Message.__table__: ('citations',),
    KnowledgeChunk.__table__: ('source_file', 'source_start_line', 'source_end_line', 'product_category', 'source_digest'),
}


def sql_statements(script: str) -> list[str]:
    """Split only unquoted semicolons; discard comments outside literals."""
    result, current = [], []
    quote = None
    i = 0
    while i < len(script):
        ch = script[i]
        if quote:
            current.append(ch)
            if ch == '\\' and i + 1 < len(script):
                i += 1
                current.append(script[i])
            elif ch == quote:
                if i + 1 < len(script) and script[i + 1] == quote:
                    i += 1
                    current.append(script[i])
                else:
                    quote = None
        elif ch in "'\"`":
            quote = ch
            current.append(ch)
        elif script.startswith('--', i) and (i + 2 == len(script) or script[i + 2].isspace()):
            end = script.find('\n', i)
            i = len(script) if end == -1 else end
            current.append('\n')
        elif script.startswith('/*', i):
            end = script.find('*/', i + 2)
            if end == -1:
                raise ValueError('Unclosed SQL comment')
            i = end + 1
            current.append(' ')
        elif ch == ';':
            if ''.join(current).strip():
                result.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    if quote:
        raise ValueError('Unclosed SQL string')
    if ''.join(current).strip():
        result.append(''.join(current).strip())
    return result



def _canonical_type(value: str) -> str:
    # SQL keywords and INT/INTEGER aliases are case-insensitive; ENUM literals aren't.
    parts = re.split(r"('(?:''|[^'])*')", value)
    return ''.join(part if part.startswith("'") else part.lower().replace(', ', ',').replace('integer', 'int')
                   for part in parts)


def _canonical_default(value):
    if value is None:
        return None
    value = str(value).strip("'")
    if value.lower() in ('null', 'current_timestamp'):
        return None if value.lower() == 'null' else 'current_timestamp'
    return value


def _column_shape(column, dialect):
    default = str(column.server_default.arg).strip("'") if column.server_default else None
    updated = bool(default and ' on update ' in default.lower())
    if updated:
        default = re.split(' on update ', default, flags=re.IGNORECASE)[0]
    default = _canonical_default(default)
    return (_canonical_type(column.type.compile(dialect=dialect)),
            'YES' if column.nullable else 'NO', default, column.comment,
            column.primary_key and column.autoincrement is True, updated)


def _columns(connection, table_name):
    rows = connection.execute(text('SELECT COLUMN_NAME,COLUMN_TYPE,IS_NULLABLE,COLUMN_DEFAULT,COLUMN_COMMENT,EXTRA FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:name'), {'name': table_name})
    return {name: (_canonical_type(kind), nullable, _canonical_default(default),
                   comment, 'auto_increment' in extra, 'on update current_timestamp' in extra.lower())
            for name, kind, nullable, default, comment, extra in rows}


def _fk_action(action):
    return None if action in (None, 'RESTRICT', 'NO ACTION') else action


def _validate_table(connection, model):
    table = model.__table__
    actual = _columns(connection, table.name)
    expected = {c.name: _column_shape(c, connection.dialect) for c in table.columns}
    if actual != expected:
        raise ValueError(f'Incompatible Chapter 4 schema: {table.name} columns; repair explicitly')
    inspector = inspect(connection)
    expected_indexes = {(i.name, tuple(c.name for c in i.columns), i.unique) for i in table.indexes}
    # MySQL creates an index for an uncovered FK automatically.
    actual_indexes = {(i['name'], tuple(i['column_names']), bool(i['unique'])) for i in inspector.get_indexes(table.name)}
    for constraint in table.foreign_key_constraints:
        cols = tuple(c.name for c in constraint.columns)
        pk = tuple(c.name for c in table.primary_key.columns)
        if not any(idxcols[:len(cols)] == cols for _,idxcols,_ in expected_indexes) and pk[:len(cols)] != cols:
            expected_indexes.add((constraint.name, cols, False))
    expected_fks = {(c.name, tuple(col.name for col in c.columns), c.elements[0].column.table.name,
                     tuple(f.column.name for f in c.elements), _fk_action(c.ondelete), _fk_action(c.onupdate)) for c in table.foreign_key_constraints}
    actual_fks = {(f['name'], tuple(f['constrained_columns']), f['referred_table'], tuple(f['referred_columns']), _fk_action(f.get('options', {}).get('ondelete')), _fk_action(f.get('options', {}).get('onupdate')))
                  for f in inspector.get_foreign_keys(table.name)}
    if (actual_indexes != expected_indexes or actual_fks != expected_fks
            or inspector.get_pk_constraint(table.name)['constrained_columns'] != [c.name for c in table.primary_key.columns]):
        raise ValueError(f'Incompatible Chapter 4 schema: {table.name} indexes or constraints')
    shape = connection.execute(text('SELECT ENGINE,TABLE_COLLATION,TABLE_COMMENT FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:name'), {'name': table.name}).one()
    if shape[0] != 'InnoDB' or not shape[1].startswith('utf8mb4') or shape[2] != table.comment:
        raise ValueError(f'Incompatible Chapter 4 schema: {table.name} table options')


def _preflight(connection):
    present = set(inspect(connection).get_table_names())
    supplied = {'low_confidence_questions', 'faith_cases'}
    if present & supplied and not supplied <= present:
        raise ValueError('Partial Chapter 4 schema; repair missing tables explicitly')
    for model in MODELS:
        if model.__tablename__ in present:
            _validate_table(connection, model)
    flags = []
    for table, names in ADDITIONS.items():
        actual = _columns(connection, table.name) if table.name in present else {}
        for name in names:
            flags.append(name in actual)
            if name in actual and actual[name] != _column_shape(table.c[name], connection.dialect):
                raise ValueError(f'Incompatible migration column {table.name}.{name}; repair explicitly')
    if any(flags) and not all(flags):
        raise ValueError('Partial Chapter 4 migration; MySQL DDL may have committed, repair explicitly')
    if all(flags):
        if not supplied <= present:
            raise ValueError('Partial Chapter 4 schema: supplied tables missing after migration')
        indexes = inspect(connection).get_indexes('conversations')
        history = next((i for i in indexes if i['name'] == 'idx_history_order'), None)
        if history is None or history['column_names'] != ['deleted_at', 'is_pinned', 'pinned_at', 'id'] or history['unique']:
            raise ValueError('Incompatible history sorting index')
        if 'knowledge_index_states' not in present:
            raise ValueError('Partial Chapter 4 migration: index state missing')
    elif 'knowledge_index_states' in present:
        raise ValueError('Partial Chapter 4 migration: provenance fields missing')
    return bool(present & supplied), all(flags)


def initialize_ch04_database(database_url: str) -> None:
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        if engine.dialect.name != 'mysql':
            raise ValueError('Chapter 4 initialization requires MySQL')
        with engine.connect() as connection:
            lock_sql = "CONCAT('sayhelp_ch04_', MD5(DATABASE()))"
            locked = connection.execute(text(f'SELECT GET_LOCK({lock_sql}, 30)')).scalar()
            if locked != 1:
                raise ValueError('Chapter 4 initialization is already running')
            try:
                supplied, migrated = _preflight(connection)
                connection.commit()
                initialize_knowledge_database(database_url)
                if not supplied:
                    for statement in sql_statements((ROOT / 'db/ch04_schema.sql').read_text(encoding='utf-8')):
                        connection.exec_driver_sql(statement)
                    connection.commit()
                if not migrated:
                    for statement in sql_statements((ROOT / 'db/ch04_migration.sql').read_text(encoding='utf-8')):
                        connection.exec_driver_sql(statement)
                    connection.commit()
                supplied, migrated = _preflight(connection)
                if not supplied or not migrated:
                    raise ValueError('Partial Chapter 4 initialization; repair explicitly')
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
        initialize_ch04_database(url.strip())
    except ValueError as error:
        raise SystemExit(str(error)) from None
    except SQLAlchemyError:
        raise SystemExit('Chapter 4 initialization failed; inspect schema compatibility and MySQL availability') from None
    print('Chapter 4 schema and additive migrations verified.')


if __name__ == '__main__':
    main()
