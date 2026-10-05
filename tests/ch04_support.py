"""Disposable migrated session helper; never connected to the business database."""
from contextlib import contextmanager
from app.db import make_session_factory
from app.init_ch04_db import initialize_ch04_database


@contextmanager
def migrated_sessions(database_url):
    initialize_ch04_database(database_url)
    factory = make_session_factory(database_url)
    try:
        yield factory
    finally:
        factory.kw['bind'].dispose()
