"""Apply the exact SQL schema and seed demonstration FAQ data; safe to rerun."""

import os
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db import Faq

SCHEMA_PATH = Path(__file__).resolve().parents[1] / 'db' / 'schema.sql'
TABLES = {'conversations', 'messages', 'faq', 'tickets'}
FAQ_SEED = (
    ('退货政策是什么？', '收到商品后七天内可申请退货；请保留商品及包装，联系客服核实条件。', '售后'),
    ('订单运费如何计算？', '运费根据收货地区与订单金额计算，请以结算页显示为准。', '配送'),
    ('如何联系人工客服？', '请说明问题，客服可以为您创建工单并转交人工处理。', '咨询'),
)


def initialize_database(database_url: str) -> None:
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        if engine.dialect.name != 'mysql':
            raise ValueError('Database initialization requires MySQL')
        # Serialize initialization and seed checks without changing the supplied DDL.
        with engine.connect() as connection:
            locked = connection.execute(text("SELECT GET_LOCK(CONCAT('sayhelp_init_', DATABASE()), 30)")).scalar()
            if locked != 1:
                raise ValueError('Database initialization is already running')
            try:
                present = set(inspect(connection).get_table_names()) & TABLES
                if present and present != TABLES:
                    raise ValueError('Partial schema detected; repair missing tables explicitly before initialization')
                if not present:
                    for statement in SCHEMA_PATH.read_text(encoding='utf-8').split(';'):
                        if statement.strip():
                            connection.exec_driver_sql(statement)
                    connection.commit()
                with Session(engine) as session, session.begin():
                    for question, answer, category in FAQ_SEED:
                        exists = session.scalar(select(Faq.id).where(Faq.question == question).limit(1))
                        if exists is None:
                            session.add(Faq(question=question, answer=answer, category=category))
            finally:
                connection.execute(text("SELECT RELEASE_LOCK(CONCAT('sayhelp_init_', DATABASE()))"))
    finally:
        engine.dispose()


def main() -> None:
    values = dotenv_values(Path.cwd() / '.env')
    url = os.environ.get('DATABASE_URL', values.get('DATABASE_URL'))
    if not url or not url.strip():
        raise SystemExit('DATABASE_URL is required')
    try:
        initialize_database(url.strip())
    except ValueError as error:
        raise SystemExit(str(error)) from None
    except SQLAlchemyError:
        raise SystemExit('Database initialization failed; check MySQL availability, credentials, and schema') from None
    print('Database schema ready; FAQ seed ready.')


if __name__ == '__main__':
    main()
