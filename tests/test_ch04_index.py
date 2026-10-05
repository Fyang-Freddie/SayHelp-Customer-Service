"""Recoverable independent indexing verifies real MySQL state; I/O adapter is explicit."""
import copy
import math

import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import Session

from app.ch04_db import KnowledgeIndexState
from app.knowledge_db import KnowledgeChunk
from test_db import database_url, sessions
from test_ch04_ingest import corpus, module


class Store:
    collection_name = 'sayhelp_ch04_test_unit'
    def __init__(self):
        self.rows = {}
        self.upserts = []
        self.deleted = []
    def ensure_collection(self):
        pass
    def upsert(self, chunk_id, vector, text, metadata):
        self.rows[chunk_id] = {'id': chunk_id, 'text': text, **metadata}
        self.upserts.append(chunk_id)
        return chunk_id
    def read_entities(self, ids=None):
        return copy.deepcopy({key:value for key,value in self.rows.items() if ids is None or key in ids})
    def delete_ids(self, ids):
        self.deleted.extend(ids)
        for key in ids:
            self.rows.pop(key,None)


class Embedder:
    def __init__(self, callback=None):
        self.callback = callback
        self.inputs = []
    def encode(self, texts):
        self.inputs.extend(texts)
        if self.callback:
            self.callback()
        return [[1/math.sqrt(1024)]*1024 for _ in texts]


def test_new_collection_indexes_existing_done_rows_and_verifies_on_repeat(sessions, corpus):
    ingest, manifest, _ = corpus
    snapshot = ingest.ingest_corpus(sessions, manifest)
    with sessions.begin() as session:
        for row in session.scalars(select(KnowledgeChunk)):
            row.vectorize_status = 'done'
    index = module('ch04_index')
    store, embedder = Store(), Embedder()
    first = index.build_index(sessions, embedder, store, snapshot)
    second = index.build_index(sessions, embedder, store, snapshot)
    assert (first.indexed, first.verified) == (5,5)
    assert (second.indexed, second.reused, second.verified) == (0,5,5)
    assert len(embedder.inputs) == 5 and set(store.rows) == set(snapshot.chunks)
    with sessions() as session:
        states = session.scalars(select(KnowledgeIndexState)).all()
        assert len(states)==5 and all(row.status=='done' for row in states)
        assert all(row.payload_digest==snapshot.payload_digests[row.chunk_id] for row in states)
        assert all(row.vectorize_status=='done' for row in session.scalars(select(KnowledgeChunk)))


def test_crash_after_vector_upsert_is_repaired_by_verified_retry(sessions, corpus):
    ingest, manifest, _ = corpus
    snapshot = ingest.ingest_corpus(sessions, manifest)
    index = module('ch04_index')
    store, embedder = Store(), Embedder()
    failed = False
    def crash(session):
        nonlocal failed
        if session.get_bind() is sessions.kw['bind'] and store.upserts and not failed:
            failed = True
            raise RuntimeError('simulated commit interruption')
    event.listen(Session,'before_commit',crash)
    try:
        with pytest.raises(RuntimeError):
            index.build_index(sessions, embedder, store, snapshot)
    finally:
        event.remove(Session,'before_commit',crash)
    assert len(store.rows)==1
    result = index.build_index(sessions, embedder, store, snapshot)
    assert result.verified==5 and set(store.rows)==set(snapshot.chunks)
    with sessions() as session:
        assert all(row.status=='done' for row in session.scalars(select(KnowledgeIndexState)))


def test_collection_recreated_or_entity_changed_is_not_skipped_by_done_state(sessions, corpus):
    ingest, manifest, _ = corpus
    snapshot = ingest.ingest_corpus(sessions, manifest)
    index = module('ch04_index')
    store, embedder = Store(), Embedder()
    index.build_index(sessions, embedder, store, snapshot)
    store.rows.clear()
    result = index.build_index(sessions, embedder, store, snapshot)
    assert result.indexed==5 and result.verified==5
    first = next(iter(store.rows))
    store.rows[first]['product_category']='错误品类'
    result = index.build_index(sessions, embedder, store, snapshot)
    assert result.indexed==1 and result.verified==5


def test_edit_during_embedding_stays_unacknowledged(sessions, corpus):
    ingest, manifest, _ = corpus
    snapshot = ingest.ingest_corpus(sessions, manifest)
    index = module('ch04_index')
    first = next(iter(snapshot.chunks))
    changed = False
    def edit():
        nonlocal changed
        if not changed:
            changed=True
            with sessions.begin() as session:
                session.get(KnowledgeChunk,first).answer='并发修改的正文。'
    with pytest.raises(RuntimeError, match='[Ss]tale|[Cc]hanged|[Ii]ndex'):
        index.build_index(sessions, Embedder(edit), Store(), snapshot)
    with sessions() as session:
        row=session.get(KnowledgeIndexState, ('sayhelp_ch04_test_unit',first))
        assert row is None or row.status!='done'


def test_old_dense_collection_can_never_be_cleaned_or_rebuilt(sessions, corpus):
    ingest, manifest, _ = corpus
    snapshot=ingest.ingest_corpus(sessions, manifest)
    index=module('ch04_index')
    store=Store()
    store.collection_name='knowledge'
    store.rows[999]={'id':999,'text':'old corpus'}
    with pytest.raises(ValueError, match='[Oo]ld|[Ll]egacy|knowledge'):
        index.build_index(sessions, Embedder(), store, snapshot)
    assert store.upserts==[] and store.deleted==[] and 999 in store.rows


def test_real_file_dry_run_needs_no_database_or_models(monkeypatch, capsys):
    import json
    from pathlib import Path
    build = module('build_ch04_knowledge')
    monkeypatch.delenv('DATABASE_URL', raising=False)
    manifest = Path(__file__).resolve().parents[1]/'eval/ch04/corpus.json'
    build.main(['--manifest',str(manifest),'--dry-run'])
    result=json.loads(capsys.readouterr().out)
    assert result['mode']=='dry_run' and result['documents']==4 and result['chunks']>4
    assert {s['source_file'] for s in result['sources']}=={'knowledge_db/after-sales-manual.md','knowledge_db/product-faq.md','knowledge_db/product-specs.md','knowledge_db/returns-policy.md'}
    assert all(s['chunks']>0 and len(s['sha256'])==64 and s['sample_range'][0]<=s['sample_range'][1] for s in result['sources'])


def test_new_version_removes_only_obsolete_vector_rows(sessions, corpus):
    ingest, manifest, root = corpus
    initial=ingest.ingest_corpus(sessions,manifest)
    store=Store()
    index=module('ch04_index')
    index.build_index(sessions,Embedder(),store,initial)
    source=root/'product-specs.md'
    source.write_text(source.read_text(encoding='utf-8')+'\n新增规范：重量 5kg。\n',encoding='utf-8')
    current=ingest.ingest_corpus(sessions,manifest)
    obsolete=set(initial.chunks)-set(current.chunks)
    result=index.build_index(sessions,Embedder(),store,current)
    assert obsolete and set(store.deleted)==obsolete
    assert set(store.rows)==set(current.chunks) and result.removed==len(obsolete)
    with sessions() as session:
        assert all(session.get(KnowledgeChunk,id_) is not None for id_ in obsolete)


def test_index_external_io_never_holds_database_transaction(sessions, corpus):
    ingest,manifest,_=corpus
    snapshot=ingest.ingest_corpus(sessions,manifest)
    engine=sessions.kw['bind']
    active=set()
    def begin(connection): active.add(connection)
    def end(connection): active.discard(connection)
    def check(): assert not active, 'Database transaction held across external I/O'
    class CheckingStore(Store):
        def ensure_collection(self): check(); super().ensure_collection()
        def read_entities(self,ids=None): check(); return super().read_entities(ids)
        def upsert(self,*args): check(); return super().upsert(*args)
        def delete_ids(self,ids): check(); super().delete_ids(ids)
    event.listen(engine,'begin',begin)
    event.listen(engine,'commit',end)
    event.listen(engine,'rollback',end)
    try:
        result=module('ch04_index').build_index(sessions,Embedder(check),CheckingStore(),snapshot)
        assert result.verified==len(snapshot.chunks)
    finally:
        event.remove(engine,'begin',begin)
        event.remove(engine,'commit',end)
        event.remove(engine,'rollback',end)
