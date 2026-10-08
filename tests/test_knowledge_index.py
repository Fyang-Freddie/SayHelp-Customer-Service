import math
from types import SimpleNamespace

import pytest
from sqlalchemy import event

from tests.test_knowledge_db import database_url
from tests.test_knowledge_ingest import sessions
from app.knowledge_db import KnowledgeChunk


def test_vector_text_has_only_three_labeled_fields():
    from app.knowledge_index import knowledge_text
    row = SimpleNamespace(category='配送', questions='运费？', answer='结算页显示。',
                          section_path='secret-path', source_ref='private', content_type='faq')
    assert knowledge_text(row) == 'category: 配送\nquestions: 运费？\nanswer: 结算页显示。'


class FakeStore:
    def __init__(self):
        self.vectors = {}
    def upsert(self, id, vector):
        self.vectors[id] = vector
        return id


class FakeEmbedder:
    def encode(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts]


def add_pending(sessions):
    with sessions() as session:
        row = KnowledgeChunk(category='配送', questions='运费？', answer='结算页显示。', vectorize_status='pending')
        session.add(row)
        session.commit()
        return row.id


def test_failed_vector_write_stays_pending(sessions):
    from app.knowledge_index import index_pending
    id = add_pending(sessions)
    class BrokenStore(FakeStore):
        def upsert(self, id, vector):
            with sessions() as s:
                assert s.get(KnowledgeChunk, id).vectorize_status == 'pending'
            raise RuntimeError('vector write failed')
    with pytest.raises(RuntimeError, match='vector write failed'):
        index_pending(sessions, FakeEmbedder(), BrokenStore(), 10)
    with sessions() as s:
        row = s.get(KnowledgeChunk, id)
        assert (row.vectorize_status, row.vector_id) == ('pending', None)


def test_rerun_repairs_crash_after_upsert(sessions):
    from app.knowledge_index import index_pending
    id = add_pending(sessions)
    store = FakeStore()
    engine = sessions.kw['bind']
    def crash(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith('UPDATE KNOWLEDGE_CHUNKS'):
            raise RuntimeError('crash after upsert')
    event.listen(engine, 'before_cursor_execute', crash)
    try:
        with pytest.raises(RuntimeError, match='crash after upsert'):
            index_pending(sessions, FakeEmbedder(), store, 10)
    finally:
        event.remove(engine, 'before_cursor_execute', crash)
    assert list(store.vectors) == [id]
    assert index_pending(sessions, FakeEmbedder(), store, 10) == 1
    assert index_pending(sessions, FakeEmbedder(), store, 10) == 0
    assert list(store.vectors) == [id]
    with sessions() as s:
        row = s.get(KnowledgeChunk, id)
        assert (row.vectorize_status, row.vector_id) == ('done', str(id))


@pytest.mark.parametrize('vector', [[0.0]*1023, [float('nan')]*1024, [float('inf')]*1024])
def test_invalid_embedding_stays_pending(sessions, vector):
    from app.knowledge_index import index_pending
    id = add_pending(sessions)
    embedder = SimpleNamespace(encode=lambda texts: [vector])
    store = FakeStore()
    with pytest.raises(ValueError):
        index_pending(sessions, embedder, store, 10)
    assert store.vectors == {}
    with sessions() as s:
        assert s.get(KnowledgeChunk, id).vectorize_status == 'pending'


def test_model_is_lazy_fixed_and_dense(monkeypatch):
    import sys
    from app.embedding import BgeM3Embedder
    calls = []
    class Model:
        max_seq_length = 8192
        tokenizer = staticmethod(lambda texts, **kwargs: {"input_ids": [[0] * (len(text) + 2) for text in texts]})
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))
        def encode(self, texts, **kwargs):
            calls.append(kwargs)
            return [[1.0]*1024 for _ in texts]
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Model))
    embedder = BgeM3Embedder(cache_dir='cache')
    assert calls == []
    assert embedder.encode([]) == []
    assert len(embedder.encode(['hello'])[0]) == 1024
    assert calls[0] == ('BAAI/bge-m3', {'cache_folder': 'cache'})
    assert calls[1]['normalize_embeddings'] is True


def test_collection_is_1024d_cosine_with_mysql_primary_key():
    from app.vector_store import MilvusKnowledgeStore
    from pymilvus import DataType
    class Client:
        def has_collection(self, name): return False
        def create_schema(self, **kwargs):
            from pymilvus import MilvusClient
            return MilvusClient.create_schema(**kwargs)
        def prepare_index_params(self):
            from pymilvus import MilvusClient
            return MilvusClient.prepare_index_params()
        def create_collection(self, **kwargs): self.created = kwargs
        def upsert(self, **kwargs):
            self.row = kwargs['data'][0]
            return {'upsert_count': 1, 'ids': [self.row['id']]}
        def search(self, **kwargs): return [[{'id': 17, 'distance': 0.9}]]
    client = Client()
    store = MilvusKnowledgeStore(client=client)
    store.ensure_collection()
    fields = client.created['schema'].fields
    assert [(f.name, f.dtype) for f in fields] == [('id', DataType.INT64), ('vector', DataType.FLOAT_VECTOR)]
    assert fields[0].is_primary and not fields[0].auto_id
    assert fields[1].params['dim'] == 1024
    assert client.created['collection_name'] == 'knowledge'
    assert client.created['index_params'][0].to_dict()['metric_type'] == 'COSINE'
    assert store.upsert(17, [1.0]*1024) == 17
    assert set(client.row) == {'id', 'vector'}
    assert store.search([1.0]*1024, 1)[0].id == 17


def test_no_mysql_transaction_during_external_io(sessions):
    from app.knowledge_index import index_pending
    add_pending(sessions)
    engine = sessions.kw['bind']
    active = set()
    def begin(connection): active.add(id(connection))
    def end(connection): active.discard(id(connection))
    event.listen(engine, 'begin', begin)
    event.listen(engine, 'commit', end)
    event.listen(engine, 'rollback', end)
    class CheckEmbedder(FakeEmbedder):
        def encode(self, texts):
            assert engine.pool.checkedout() == 1  # Dedicated advisory-lock connection.
            assert not active
            return super().encode(texts)
    class CheckStore(FakeStore):
        def upsert(self, id, vector):
            assert engine.pool.checkedout() == 1  # Dedicated advisory-lock connection.
            assert not active
            return super().upsert(id, vector)
    try:
        assert index_pending(sessions, CheckEmbedder(), CheckStore(), 10) == 1
    finally:
        event.remove(engine, 'begin', begin)
        event.remove(engine, 'commit', end)
        event.remove(engine, 'rollback', end)


def test_wrong_upsert_primary_key_stays_pending(sessions):
    from app.knowledge_index import index_pending
    id = add_pending(sessions)
    store = SimpleNamespace(upsert=lambda id, vector: id + 1)
    with pytest.raises(ValueError, match='primary key'):
        index_pending(sessions, FakeEmbedder(), store, 10)
    with sessions() as s:
        assert s.get(KnowledgeChunk, id).vectorize_status == 'pending'


def test_real_milvus_mysql_recovery(sessions):
    import os
    uri = os.environ.get('TEST_MILVUS_URI')
    if not uri:
        pytest.skip('Set TEST_MILVUS_URI for a disposable Milvus instance')
    from pymilvus import MilvusClient
    from app.vector_store import MilvusKnowledgeStore
    from app.knowledge_index import index_pending
    database = os.environ.get('TEST_MILVUS_DATABASE')
    if database is not None:
        import re
        if not re.fullmatch(r'sayhelp_test_[0-9a-f]{32}', database):
            pytest.fail('TEST_MILVUS_DATABASE must be a fresh sayhelp_test_<32hex> database')
    client = MilvusClient(uri=uri, **({'db_name': database} if database else {}))
    if client.has_collection('knowledge'):
        client.close()
        pytest.fail('Real integration requires a disposable Milvus without knowledge collection')
    try:
        store = MilvusKnowledgeStore(client=client)
        store.ensure_collection()
        # Validate the existing schema path too.
        store.ensure_collection()
        id = add_pending(sessions)
        engine = sessions.kw['bind']
        def crash(connection, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith('UPDATE KNOWLEDGE_CHUNKS'):
                raise RuntimeError('real crash after upsert')
        event.listen(engine, 'before_cursor_execute', crash)
        try:
            with pytest.raises(RuntimeError, match='real crash'):
                index_pending(sessions, FakeEmbedder(), store, 10)
        finally:
            event.remove(engine, 'before_cursor_execute', crash)
        assert index_pending(sessions, FakeEmbedder(), store, 10) == 1
        assert index_pending(sessions, FakeEmbedder(), store, 10) == 0
        assert client.query('knowledge', filter=f'id == {id}', output_fields=['id']) == [{'id': id}]
        assert store.search([1.0] + [0.0]*1023, 5)[0].id == id
        with sessions() as s:
            row = s.get(KnowledgeChunk, id)
            assert (row.vectorize_status, row.vector_id) == ('done', str(id))
    finally:
        client.drop_collection('knowledge')
        client.close()


def test_real_bge_m3_dense():
    import os
    if os.environ.get('TEST_BGE_M3') != '1':
        pytest.skip('Set TEST_BGE_M3=1 to load the real public BGE-M3 model')
    from app.embedding import BgeM3Embedder
    vectors = BgeM3Embedder(cache_dir=os.environ.get('BGE_CACHE_DIR')).encode(['邮费是多少？'])
    assert len(vectors) == 1 and len(vectors[0]) == 1024
    assert all(math.isfinite(value) for value in vectors[0])


def test_build_cli_ingests_then_drains_pending(monkeypatch, capsys):
    from app import build_knowledge as build
    events = []
    monkeypatch.setenv('DATABASE_URL', 'private-test-placeholder')
    monkeypatch.setattr(build, 'initialize_knowledge_database', lambda url: events.append('init'))
    engine = SimpleNamespace(dispose=lambda: events.append('dispose'))
    monkeypatch.setattr(build, 'create_engine', lambda *args, **kwargs: engine)
    monkeypatch.setattr(build, 'sessionmaker', lambda engine: 'sessions')
    monkeypatch.setattr(build, 'ingest_faq', lambda sessions: events.append('faq'))
    monkeypatch.setattr(build, 'ingest_markdown', lambda sessions, path, kind: events.append((path, kind)))
    monkeypatch.setattr(build, 'BgeM3Embedder', lambda **kwargs: 'embedder')
    store = SimpleNamespace(ensure_collection=lambda: events.append('milvus'))
    monkeypatch.setattr(build, 'MilvusKnowledgeStore', lambda **kwargs: store)
    counts = iter([2, 1, 0])
    monkeypatch.setattr(build, 'index_pending', lambda *args: next(counts))
    monkeypatch.setattr(build, 'count_pending', lambda sessions: 0)
    build.main(['--policy', 'policy.md', '--manual', 'manual.md', '--batch-size', '2'])
    assert events == ['init', 'faq', ('policy.md', 'policy'), ('manual.md', 'manual'), 'milvus', 'dispose']
    assert '3 pending rows marked done' in capsys.readouterr().out


def test_build_cli_failure_hides_connection_details(monkeypatch):
    from app import build_knowledge as build
    monkeypatch.setenv('DATABASE_URL', 'credential-sentinel')
    def fail(url):
        raise RuntimeError(url)
    monkeypatch.setattr(build, 'initialize_knowledge_database', fail)
    with pytest.raises(SystemExit) as error:
        build.main([])
    assert 'credential-sentinel' not in str(error.value)
    assert 'pending rows can be retried' in str(error.value)


def test_concurrent_new_content_cannot_leave_done_with_old_vector(sessions):
    from app.knowledge_index import index_pending
    id = add_pending(sessions)
    store = FakeStore()
    class ContentEmbedder:
        def encode(self, texts):
            return [[2.0 if '新答案' in text else 1.0] + [0.0]*1023 for text in texts]
    new_embedder = ContentEmbedder()
    class InterleavingEmbedder(ContentEmbedder):
        def encode(self, texts):
            with sessions() as s, s.begin():
                row = s.get(KnowledgeChunk, id)
                row.answer = '新答案'
                row.vectorize_status = 'pending'
            try:
                index_pending(sessions, new_embedder, store, 10)
            except RuntimeError as error:
                assert 'indexing' in str(error)
            return super().encode(texts)
    index_pending(sessions, InterleavingEmbedder(), store, 10)
    # Retry any pending content after worker A releases ownership.
    index_pending(sessions, new_embedder, store, 10)
    with sessions() as s:
        row = s.get(KnowledgeChunk, id)
        assert row.answer == '新答案' and row.vectorize_status == 'done'
    assert store.vectors[id][0] == 2.0


def test_build_cli_zero_progress_with_pending_is_failure(sessions, monkeypatch, capsys):
    from app import build_knowledge as build
    add_pending(sessions)
    monkeypatch.setenv('DATABASE_URL', 'private-test-placeholder')
    monkeypatch.setattr(build, 'initialize_knowledge_database', lambda url: None)
    monkeypatch.setattr(build, 'create_engine', lambda *args, **kwargs: SimpleNamespace(dispose=lambda: None))
    monkeypatch.setattr(build, 'sessionmaker', lambda engine: sessions)
    monkeypatch.setattr(build, 'ingest_faq', lambda sessions: None)
    monkeypatch.setattr(build, 'BgeM3Embedder', lambda **kwargs: FakeEmbedder())
    monkeypatch.setattr(build, 'MilvusKnowledgeStore', lambda **kwargs: SimpleNamespace(ensure_collection=lambda: None))
    monkeypatch.setattr(build, 'index_pending', lambda *args: 0)
    with pytest.raises(SystemExit, match='pending rows can be retried'):
        build.main([])
    assert 'completed' not in capsys.readouterr().out


class CharacterTokenizerModel:
    """Cheap inference double; production adapter still owns all validation."""
    max_seq_length = 100

    def tokenizer(self, texts, *, truncation, add_special_tokens, padding):
        assert truncation is False and add_special_tokens is True and padding is False
        return {'input_ids': [[0] * (len(text) + 2) for text in texts]}

    def encode(self, texts, **kwargs):
        # Any truncated inference on an oversized input must fail this regression.
        assert all(len(text) + 2 <= self.max_seq_length for text in texts)
        return [[1.0] + [0.0] * 1023 for _ in texts]


def bounded_embedder():
    from app.embedding import BgeM3Embedder
    embedder = BgeM3Embedder()
    embedder._model = CharacterTokenizerModel()
    return embedder


def test_embedding_token_limit_includes_special_tokens_and_checks_entire_batch():
    embedder = bounded_embedder()
    assert len(embedder.encode(['x' * 98])[0]) == 1024
    with pytest.raises(ValueError, match=r'input 2.*101 tokens.*100'):
        embedder.encode(['short', 'x' * 99])


@pytest.mark.parametrize('source', ['faq', 'manual'])
def test_oversized_complete_vector_text_stays_pending_with_source_context(sessions, source):
    from app.knowledge_ingest import ingest_faq
    from app.knowledge_index import index_pending
    from app.knowledge_chunking import chunk_markdown
    from app.db import Faq
    with sessions() as session, session.begin():
        # Isolate the oversized input from seeded FAQs.
        for row in session.query(Faq).all():
            session.delete(row)
        if source == 'faq':
            session.add(Faq(category='FAQ source', question='x' * 75, answer='safe.'))
        else:
            draft = chunk_markdown('# Manual source\n' + 'x' * 75 + '.',
                                   content_type='manual', max_chars=200)[0]
            session.add(KnowledgeChunk(category=draft.category, questions=draft.questions,
                answer=draft.answer, section_path=draft.section_path, content_type='manual',
                vectorize_status='pending'))
    if source == 'faq':
        ids = ingest_faq(sessions)
    else:
        with sessions() as session:
            ids = [row.id for row in session.query(KnowledgeChunk).all()]
    store = FakeStore()
    with pytest.raises(ValueError, match=rf'chunk {ids[0]}.*{source}.*tokens'):
        index_pending(sessions, bounded_embedder(), store)
    assert store.vectors == {}
    with sessions() as session:
        row = session.get(KnowledgeChunk, ids[0])
        assert (row.vectorize_status, row.vector_id) == ('pending', None)
        assert ('x' * 75) in (row.questions + row.answer)


def test_real_bge_tokenizer_rejects_8193_tokens_without_inference():
    import os
    if os.environ.get('TEST_BGE_M3') != '1':
        pytest.skip('Set TEST_BGE_M3=1 for cached public BGE-M3 tokenizer boundary')
    from transformers import AutoTokenizer
    from app.embedding import BgeM3Embedder, MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=os.environ.get('BGE_CACHE_DIR'),
                                              local_files_only=True)
    class TokenizerModel:
        max_seq_length = 8192
        def __init__(self):
            self.tokenizer = tokenizer
        def encode(self, texts, **kwargs):
            # Cheap inference double prevents quadratic 8192-token transformer work.
            return [[1.0] + [0.0] * 1023 for _ in texts]
    embedder = BgeM3Embedder()
    embedder._model = TokenizerModel()
    boundary = 'the ' * 8190
    assert len(tokenizer(boundary, truncation=False, add_special_tokens=True)['input_ids']) == 8192
    assert len(embedder.encode([boundary])[0]) == 1024
    with pytest.raises(ValueError, match='8193 tokens.*8192'):
        embedder.encode([boundary + 'the'])


def test_build_cli_reports_safe_token_limit_source_and_leaves_pending(sessions, monkeypatch):
    from app import build_knowledge as build
    with sessions() as session, session.begin():
        row = KnowledgeChunk(category='Manual', questions='Restart', answer='private-body-' * 10,
                             content_type='manual', section_path='Manual / Restart',
                             vectorize_status='pending')
        session.add(row)
        session.flush()
        id = row.id
    monkeypatch.setenv('DATABASE_URL', 'credential-sentinel')
    monkeypatch.setattr(build, 'initialize_knowledge_database', lambda url: None)
    monkeypatch.setattr(build, 'create_engine', lambda *args, **kwargs: SimpleNamespace(dispose=lambda: None))
    monkeypatch.setattr(build, 'sessionmaker', lambda engine: sessions)
    monkeypatch.setattr(build, 'ingest_faq', lambda sessions: None)
    monkeypatch.setattr(build, 'BgeM3Embedder', lambda **kwargs: bounded_embedder())
    store = FakeStore()
    store.ensure_collection = lambda: None
    monkeypatch.setattr(build, 'MilvusKnowledgeStore', lambda **kwargs: store)
    with pytest.raises(SystemExit, match=rf'chunk {id}.*manual.*Manual / Restart.*tokens.*100') as error:
        build.main([])
    assert 'credential-sentinel' not in str(error.value)
    assert 'private-body' not in str(error.value)
    assert 'pending rows can be retried' in str(error.value)
    with sessions() as session:
        assert session.get(KnowledgeChunk, id).vectorize_status == 'pending'
    assert store.vectors == {}
