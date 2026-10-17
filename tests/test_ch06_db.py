"""Additive chapter 6 schema on disposable real MySQL."""
import importlib
import pytest
from sqlalchemy import create_engine, inspect
from test_db import database_url
from app.init_ch04_db import initialize_ch04_database
from app.init_ch05_db import initialize_ch05_database


def initialize(url):
    return importlib.import_module('app.init_ch06_db').initialize_ch06_database(url)


def test_schema_repeats_and_money_is_decimal(database_url):
    initialize_ch04_database(database_url)
    initialize_ch05_database(database_url)
    engine=create_engine(database_url)
    try:
        before=set(inspect(engine).get_table_names())
        initialize(database_url)
        initialize(database_url)
        assert set(inspect(engine).get_table_names())-before=={'workflow_interactions','workflow_resume_requests','refund_actions','refund_requests'}
        columns={c['name']:c for c in inspect(engine).get_columns('refund_requests')}
        assert str(columns['amount']['type'])=='DECIMAL(12, 2)'
        assert columns['amount']['nullable'] is False
    finally: engine.dispose()


@pytest.mark.parametrize('damage',['partial','column'])
def test_partial_or_incompatible_schema_is_not_repaired(database_url,damage):
    initialize_ch04_database(database_url)
    initialize_ch05_database(database_url)
    initialize(database_url)
    engine=create_engine(database_url)
    try:
        with engine.begin() as c:
            c.exec_driver_sql('DROP TABLE refund_requests' if damage=='partial' else 'ALTER TABLE workflow_interactions MODIFY status VARCHAR(90) NOT NULL')
        before=set(inspect(engine).get_table_names())
        with pytest.raises(ValueError,match='[Pp]artial|[Ii]ncompatible'): initialize(database_url)
        assert set(inspect(engine).get_table_names())==before
    finally: engine.dispose()
