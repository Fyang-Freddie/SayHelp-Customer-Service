"""Explicit additive migration; never recreate or delete existing chat data."""
import os
from pathlib import Path
from dotenv import dotenv_values
from sqlalchemy import create_engine, inspect, text
from app.ch07_db import ConversationContext, ConversationSummary


def initialize_ch07_database(database_url: str) -> None:
    engine=create_engine(database_url)
    try:
        if engine.dialect.name!='mysql':
            raise ValueError('Chapter 7 initialization requires MySQL')
        with engine.connect() as conn:
            if conn.execute(text("SELECT GET_LOCK(CONCAT('sayhelp_ch07_',DATABASE()),30)")).scalar()!=1:
                raise ValueError('Context migration already running')
            try:
                names=set(inspect(conn).get_table_names())
                if not {'conversations','messages'}<=names:
                    raise ValueError('Initialize conversation schema first')
                needed=set(ConversationContext.__table__.columns.keys())-{'id'}
                present={c['name'] for c in inspect(conn).get_columns('conversations')}
                if present & needed and not needed<=present:
                    raise ValueError('Partial Chapter 7 schema; repair explicitly')
                if not needed<=present:
                    conn.exec_driver_sql((Path(__file__).resolve().parents[1]/'db/ch07_migration.sql').read_text(encoding='utf8').strip().rstrip(';'))
                ConversationSummary.__table__.create(conn,checkfirst=True)
                conn.commit()
            finally:
                conn.rollback()
                conn.execute(text("SELECT RELEASE_LOCK(CONCAT('sayhelp_ch07_',DATABASE()))"))
                conn.commit()
    finally:
        engine.dispose()


def main():
    values=dotenv_values(Path.cwd()/'.env')
    url=os.environ.get('DATABASE_URL') or values.get('DATABASE_URL')
    if not url: raise SystemExit('DATABASE_URL is required')
    initialize_ch07_database(url)
    print('Chapter 7 context schema ready')


if __name__=='__main__': main()
