"""Add the exact Chapter 3 knowledge schema without rewriting existing data."""

import os
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.init_db import initialize_database

SCHEMA_PATH = Path(__file__).resolve().parents[1] / 'db' / 'ch03_schema.sql'
TABLES = {'knowledge_chunks', 'qa_extraction_staging'}


def initialize_knowledge_database(database_url: str) -> None:
    initialize_database(database_url)
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            locked = connection.execute(text("SELECT GET_LOCK(CONCAT('sayhelp_ch03_', DATABASE()), 30)")).scalar()
            if locked != 1:
                raise ValueError('Knowledge database initialization is already running')
            try:
                present = set(inspect(connection).get_table_names()) & TABLES
                if present and present != TABLES:
                    raise ValueError('Partial Chapter 3 schema detected; repair missing tables explicitly before initialization')
                if not present:
                    for statement in SCHEMA_PATH.read_text(encoding='utf-8').split(';'):
                        if statement.strip():
                            connection.exec_driver_sql(statement)
                    connection.commit()
            finally:
                connection.execute(text("SELECT RELEASE_LOCK(CONCAT('sayhelp_ch03_', DATABASE()))"))
    finally:
        engine.dispose()


def main() -> None:
    values = dotenv_values(Path.cwd() / '.env')
    url = os.environ.get('DATABASE_URL', values.get('DATABASE_URL'))
    if not url or not url.strip():
        raise SystemExit('DATABASE_URL is required')
    try:
        initialize_knowledge_database(url.strip())
    except ValueError as error:
        raise SystemExit(str(error)) from None
    except SQLAlchemyError:
        raise SystemExit('Knowledge database initialization failed; check MySQL availability, credentials, and schema') from None
    print('Chapter 2 schema and FAQ seed ready; Chapter 3 knowledge schema ready.')


if __name__ == '__main__':
    main()
