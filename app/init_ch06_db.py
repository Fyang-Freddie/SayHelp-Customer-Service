"""Explicit, repeatable Chapter 6 initialization; never rebuild partial schemas."""
import os
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.init_ch04_db import sql_statements
from app.ch06_db import WorkflowInteraction, WorkflowResumeRequest, RefundAction, RefundRequest
from app.init_ch05_db import _preflight as ch05_preflight, _validate_workflow_table

ROOT = Path(__file__).resolve().parents[1]
MODELS = (WorkflowInteraction, WorkflowResumeRequest, RefundAction, RefundRequest)


def _preflight(connection):
    if not ch05_preflight(connection):
        raise ValueError('Incompatible Chapter 5 schema: initialize Chapter 5 first')
    present=set(inspect(connection).get_table_names())
    names={m.__tablename__ for m in MODELS}
    if present & names and not names <= present:
        raise ValueError('Partial Chapter 6 schema; repair explicitly')
    for model in MODELS:
        if model.__tablename__ in present:
            try: _validate_workflow_table(connection,model)
            except ValueError as error: raise ValueError(str(error).replace('Chapter 5','Chapter 6')) from None
    return names <= present


def initialize_ch06_database(database_url: str) -> None:
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        if engine.dialect.name != 'mysql':
            raise ValueError('Chapter 6 initialization requires MySQL')
        with engine.connect() as connection:
            lock_sql = "CONCAT('sayhelp_ch06_', MD5(DATABASE()))"
            locked = connection.execute(text(f'SELECT GET_LOCK({lock_sql}, 30)')).scalar()
            if locked != 1:
                raise ValueError('Chapter 6 initialization is already running')
            try:
                initialized = _preflight(connection)
                connection.commit()
                if not initialized:
                    for statement in sql_statements((ROOT / 'db/ch06_schema.sql').read_text(encoding='utf-8')):
                        connection.exec_driver_sql(statement)
                    connection.commit()
                if not _preflight(connection):
                    raise ValueError('Partial Chapter 6 initialization; repair explicitly')
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
        initialize_ch06_database(url.strip())
    except ValueError as error:
        raise SystemExit(str(error)) from None
    except SQLAlchemyError:
        raise SystemExit('Chapter 6 initialization failed; inspect schema compatibility and MySQL availability') from None
    print('Chapter 6 additive interaction metadata schema verified.')


if __name__ == '__main__':
    main()
