"""Real MySQL authority with fake external embedding/vector adapters."""
import importlib.util
import json
from pathlib import Path

import pytest
from sqlalchemy import event

from app.knowledge_db import KnowledgeChunk
from app.vector_store import SearchHit
from tests.test_knowledge_ingest import database_url, sessions
from test_tools import FakeRepository
from app.tools import build_tools


def search_class():
    assert importlib.util.find_spec('app.knowledge_search') is not None, 'Dense retrieval adapter is required'
    from app.knowledge_search import KnowledgeSearch
    return KnowledgeSearch


class Embedder:
    def __init__(self, error=None):
        self.texts = []
        self.error = error
    def encode(self, texts):
        self.texts.extend(texts)
        if self.error:
            raise self.error
        return [[1.0] + [0.0] * 1023]


class Store:
    def __init__(self, hits=(), error=None):
        self.hits = list(hits)
        self.error = error
        self.limits = []
    def search(self, vector, limit):
        assert len(vector) == 1024
        self.limits.append(limit)
        if self.error:
            raise self.error
        return self.hits[:limit]


def insert(sessions, question, *, status='done'):
    with sessions() as session:
        row = KnowledgeChunk(category='配送', questions=question, answer='MySQL权威答案', vectorize_status=status)
        session.add(row)
        session.flush()
        row.vector_id = str(row.id) if status == 'done' else None
        id_ = row.id
        session.commit()
        return id_


def test_search_preserves_rank_and_skips_stale_rows(sessions):
    search_type = search_class()
    first = insert(sessions, '数据库第一条')
    second = insert(sessions, '向量第一名')
    pending = insert(sessions, '待索引', status='pending')
    deleted = insert(sessions, '已删除')
    with sessions.begin() as session:
        session.delete(session.get(KnowledgeChunk, deleted))
    embedder = Embedder()
    store = Store([SearchHit(second, .96), SearchHit(pending, .94), SearchHit(deleted, .92), SearchHit(999999, .90), SearchHit(first, .88)])
    result = search_type(sessions, embedder, store).search('邮费', limit=5)
    assert [(match.question, match.answer, match.category) for match in result] == [('向量第一名', 'MySQL权威答案', '配送'), ('数据库第一条', 'MySQL权威答案', '配送')]
    assert embedder.texts == ['邮费'] and store.limits == [5]
    assert [match.question for match in search_type(sessions, embedder, store).search('邮费', limit=1)] == ['向量第一名']


@pytest.mark.parametrize('failure', ['low', 'embedding', 'milvus', 'mysql'])
def test_low_score_or_backend_error_is_safe_miss(sessions, failure):
    search_type = search_class()
    id_ = insert(sessions, '订单运费如何计算？')
    private = RuntimeError('private dependency details')
    embedder = Embedder(private if failure == 'embedding' else None)
    store = Store([SearchHit(id_, .54 if failure == 'low' else .95)], private if failure == 'milvus' else None)
    def broken_sessions():
        raise private
    adapter = search_type(broken_sessions if failure == 'mysql' else sessions, embedder, store)
    payload = build_tools(FakeRepository(), 42, knowledge_search=adapter)['query_faq'].invoke({'keyword': '邮费'})
    assert set(payload) == {'keyword', 'matches', 'message'}
    assert payload['keyword'] == '邮费' and payload['matches'] == []
    assert ('未找到' if failure == 'low' else '暂时不可用') in payload['message']
    assert 'private' not in json.dumps(payload)


def test_labeled_cases_use_real_mysql_hydration_with_fake_scores(sessions):
    search_type = search_class()
    id_ = insert(sessions, '订单运费如何计算？')
    cases = json.loads(Path('tests/fixtures/ch03_cases.json').read_text(encoding='utf-8'))
    for case in cases:
        adapter = search_type(sessions, Embedder(), Store([SearchHit(id_, case['fake_score'])]))
        payload = build_tools(FakeRepository(), 42, knowledge_search=adapter)['query_faq'].invoke({'keyword': case['keyword']})
        assert bool(payload['matches']) is case['expected_match'], case['id']
        assert payload['keyword'] == case['keyword']


@pytest.mark.parametrize('score, hit', [(0.55, True), (0.5499, False), (float('nan'), False), (float('inf'), False)])
def test_score_boundary_and_nonfinite_scores(sessions, score, hit):
    id_ = insert(sessions, '运费')
    result = search_class()(sessions, Embedder(), Store([SearchHit(id_, score)])).search('邮费')
    assert bool(result) is hit


@pytest.mark.parametrize('limit', [0, -1, 101, True])
def test_search_rejects_unbounded_or_invalid_limit(sessions, limit):
    with pytest.raises(ValueError, match='limit'):
        search_class()(sessions, Embedder(), Store()).search('邮费', limit=limit)


def test_embedding_and_vector_io_happen_before_mysql_transaction(sessions):
    active = set()
    engine = sessions.kw['bind']
    event.listen(engine, 'begin', lambda connection: active.add(connection))
    event.listen(engine, 'commit', lambda connection: active.discard(connection))
    event.listen(engine, 'rollback', lambda connection: active.discard(connection))
    class ObservedEmbedder(Embedder):
        def encode(self, texts):
            assert not active
            return super().encode(texts)
    class ObservedStore(Store):
        def search(self, vector, limit):
            assert not active
            return super().search(vector, limit)
    id_ = insert(sessions, '运费')
    assert search_class()(sessions, ObservedEmbedder(), ObservedStore([SearchHit(id_, .95)])).search('邮费')[0].question == '运费'


def test_configured_score_threshold_changes_online_matches(sessions, monkeypatch, tmp_path):
    from app.config import Settings
    monkeypatch.chdir(tmp_path)
    for name, value in {'CHAT_BASE_URL': 'https://example.invalid', 'CHAT_MODEL': 'test', 'CHAT_API_KEY': 'test', 'DATABASE_URL': 'mysql+pymysql://test:test@localhost/test', 'KNOWLEDGE_MIN_SCORE': '0.8'}.items():
        monkeypatch.setenv(name, value)
    config = Settings.from_env()
    assert config.knowledge_min_score == .8
    id_ = insert(sessions, '运费')
    assert search_class()(sessions, Embedder(), Store([SearchHit(id_, .7)]), min_score=config.knowledge_min_score).search('邮费') == []


@pytest.mark.parametrize('raw', ['nan', 'inf', '-1.1', '1.1', 'invalid'])
def test_invalid_threshold_configuration_rejected(monkeypatch, tmp_path, raw):
    from app.config import Settings
    monkeypatch.chdir(tmp_path)
    for name, value in {'CHAT_BASE_URL': 'https://example.invalid', 'CHAT_MODEL': 'test', 'CHAT_API_KEY': 'test', 'DATABASE_URL': 'mysql+pymysql://test:test@localhost/test', 'KNOWLEDGE_MIN_SCORE': raw}.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match='KNOWLEDGE_MIN_SCORE'):
        Settings.from_env()
