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


LEGACY_DOCUMENTS = {
    'knowledge_db/after-sales-manual.md':'manual',
    'knowledge_db/returns-policy.md':'policy',
    'knowledge_db/product-faq.md':'faq',
    'knowledge_db/product-specs.md':'manual',
}

def activate_legacy_corpus(monkeypatch,root=None):
    from pathlib import Path
    from app import ch04_ingest
    from app.evaluation import dataset
    fixture=Path(__file__).resolve().parent/'fixtures/ch04_corpus'
    monkeypatch.setattr(ch04_ingest,'ROOT',root or fixture)
    monkeypatch.setattr(ch04_ingest,'DOCUMENTS',LEGACY_DOCUMENTS)
    monkeypatch.setattr(dataset,'DOCUMENTS',LEGACY_DOCUMENTS)
    return fixture
