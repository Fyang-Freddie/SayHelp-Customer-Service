"""Workflow knowledge gate and privacy-safe local recording contracts."""
import asyncio
from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

import pytest

from app.evaluation.calibration import ConfidencePolicy
from app.knowledge_types import EvidenceChunk, KnowledgeFilters, PreparedQuery, RankedChunk, RetrievalResult
from app.rag_retrieval import IndexStaleError


def settings(budget=12000):
    return SimpleNamespace(context_token_budget=budget, response_token_reserve=512)


def retrieved(score=.8, count=1, answer='原文完整证据'):
    query = PreparedQuery('原话', '原话', '原话')
    ranked = [RankedChunk(EvidenceChunk(i, '问题', answer, '商品', '饮水机', 'manual',
              '商品 / 型号', 'knowledge_db/product-specs.md', 1, 9, 'a'*64), i, score)
              for i in range(1, count+1)]
    return RetrievalResult('hybrid_rerank', query, ranked, ranked, {})


class Retrieval:
    def __init__(self, result=None, error=None):
        self.result = result or retrieved()
        self.error = error
        self.calls = []

    def retrieve(self, query, strategy, filters=None):
        self.calls.append((query, strategy, filters))
        if self.error:
            raise self.error
        return replace(self.result, query=query)


def gate(tmp_path, retrieval=None, budget=12000, log=None):
    api = importlib.import_module('app.workflow_knowledge')
    logger = importlib.import_module('app.workflow_log').WorkflowLog(tmp_path)
    return api.KnowledgeGate(retrieval or Retrieval(),
        ConfidencePolicy({'hybrid_rerank': .5}, {}), settings(budget), log=log or logger)


def test_retrieves_raw_query_hybrid_and_preserves_filters(tmp_path):
    retrieval = Retrieval()
    question = '  MH-W20 不支持什么？\n原话  '
    filters = KnowledgeFilters(content_type='manual')
    result = asyncio.run(gate(tmp_path, retrieval).prepare(question, filters))
    query, strategy, actual_filters = retrieval.calls[0]
    assert strategy == 'hybrid_rerank' and actual_filters is filters
    assert query.raw_question == query.standard_question == query.search_text == question
    assert query.synonyms == [] and query.intent == 'knowledge'
    assert result['stop_reason'] is None and result['confidence']['sufficient']
    assert result['evidence'][0]['answer'] == '原文完整证据'
    citation = result['citations'][0]
    assert citation['chunk_id'] == str(result['evidence'][0]['id'])
    assert citation['source_digest'] == 'a'*64 and citation['source_start_line'] == 1
    json.dumps(result)


@pytest.mark.parametrize('score,passed', [(.499999, False), (.5, True), (.500001, True)])
def test_confidence_exact_boundary(tmp_path, score, passed):
    result = asyncio.run(gate(tmp_path, Retrieval(retrieved(score))).prepare('问题'))
    assert result['confidence']['sufficient'] is passed
    assert (result['stop_reason'] is None) is passed
    assert bool(result['evidence']) is passed
    assert bool(result['citations']) is passed


@pytest.mark.parametrize('result,budget', [(retrieved(count=0), 12000),
    (retrieved(answer='长'*16000), 2000)])
def test_empty_or_budget_excluded_evidence_cannot_pass(tmp_path, result, budget):
    outcome = asyncio.run(gate(tmp_path, Retrieval(result), budget).prepare('原始问题'))
    assert outcome['stop_reason'] == 'weak_evidence'
    assert not outcome['confidence']['sufficient']
    assert outcome['evidence'] == outcome['citations'] == []
    assert outcome['low_confidence_recorded'] is True
    lines = [json.loads(line) for line in (tmp_path/'low-confidence.jsonl').read_text(encoding='utf-8').splitlines()]
    assert lines[0]['question'] == '原始问题'


@pytest.mark.parametrize('error,reason', [(IndexStaleError('sensitive'), 'index_stale'),
    (RuntimeError('secret api_key=x'), 'retrieval_error')])
def test_retrieval_errors_are_not_weak_evidence(tmp_path, error, reason):
    outcome = asyncio.run(gate(tmp_path, Retrieval(error=error)).prepare('私密原问题'))
    assert outcome['stop_reason'] == reason and not outcome['confidence']['sufficient']
    assert outcome['evidence'] == outcome['citations'] == []
    assert not (tmp_path/'low-confidence.jsonl').exists()
    event = json.loads((tmp_path/'events.jsonl').read_text(encoding='utf-8').splitlines()[0])
    assert event['event'] == 'retrieval_error' and event['payload']['stop_reason'] == reason
    assert 'sensitive' not in json.dumps(event) and 'secret' not in json.dumps(event)
    assert '私密原问题' not in json.dumps(event, ensure_ascii=False)


def test_weak_record_failure_is_reported_without_claiming_recorded(tmp_path):
    class BrokenLog:
        def write(self, event, payload):
            if event == 'weak_evidence':
                raise OSError('disk failure')
    outcome = asyncio.run(gate(tmp_path, Retrieval(retrieved(.1)), log=BrokenLog()).prepare('问题'))
    assert outcome['stop_reason'] == 'low_confidence_log_error'
    assert outcome['low_confidence_recorded'] is False
    assert not outcome['confidence']['sufficient']


def test_log_allowlists_safe_fields_and_only_weak_file_has_question(tmp_path):
    logger = importlib.import_module('app.workflow_log').WorkflowLog(tmp_path)
    payload = {'question': '原始问题', 'score': .1, 'threshold': .5,
               'stop_reason': 'weak_evidence', 'api_key': 'secret',
               'messages': [{'content': 'hidden'}], 'error_message': 'private'}
    logger.write('weak_evidence', payload)
    logger.write('retrieval_error', payload)
    events = (tmp_path/'events.jsonl').read_text(encoding='utf-8')
    weak = (tmp_path/'low-confidence.jsonl').read_text(encoding='utf-8')
    assert '原始问题' not in events and 'secret' not in events+weak and 'hidden' not in events+weak
    assert json.loads(weak)['question'] == '原始问题'
    assert len(events.splitlines()) == 2


def test_citations_only_cover_whole_selected_evidence(tmp_path):
    outcome = asyncio.run(gate(tmp_path, Retrieval(retrieved(count=10, answer='整段'*100)), 2200).prepare('问题'))
    assert outcome['confidence']['sufficient']
    assert 0 < len(outcome['evidence']) < 10
    assert {str(c['id']) for c in outcome['evidence']} == {c['chunk_id'] for c in outcome['citations']}
    assert all(c['answer'] == '整段'*100 for c in outcome['evidence'])


@pytest.mark.parametrize('score', [float('inf'), float('nan')])
def test_invalid_scores_do_not_pass(tmp_path, score):
    outcome = asyncio.run(gate(tmp_path, Retrieval(retrieved(score))).prepare('问题'))
    assert not outcome['confidence']['sufficient'] and outcome['stop_reason'] == 'retrieval_error'


def test_cancelled_task_is_propagated(tmp_path):
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(gate(tmp_path, Retrieval(error=asyncio.CancelledError())).prepare('问题'))



def test_live_retrieval_factory_closes_resources_on_constructor_error(monkeypatch):
    api = importlib.import_module('app.workflow_knowledge')
    from app import db, ch04_ingest, hybrid_store, embedding
    closed = []
    sessions = SimpleNamespace(kw={'bind': SimpleNamespace(dispose=lambda: closed.append('engine'))})
    monkeypatch.setattr(db, 'make_session_factory', lambda url: sessions)
    monkeypatch.setattr(ch04_ingest, 'read_current_corpus', lambda *args: 'corpus')
    class Store:
        def __init__(self, **kwargs): pass
        def close(self): closed.append('store')
    monkeypatch.setattr(hybrid_store, 'HybridMilvusStore', Store)
    def broken(**kwargs): raise RuntimeError('model construction failed')
    monkeypatch.setattr(embedding, 'BgeM3Embedder', broken)
    config = SimpleNamespace(database_url='unused', knowledge_collection='knowledge_ch04', milvus_uri='unused', bge_cache_dir=None)
    with pytest.raises(RuntimeError, match='model construction failed'):
        with api.create_live_retrieval(config):
            pass
    assert closed == ['store', 'engine']


def test_live_gate_factory_closes_retrieval_on_policy_error(monkeypatch):
    from contextlib import contextmanager
    api = importlib.import_module('app.workflow_knowledge')
    closed = []
    @contextmanager
    def resources(settings):
        try: yield SimpleNamespace(corpus='corpus')
        finally: closed.append(True)
    monkeypatch.setattr(api, 'create_live_retrieval', resources)
    monkeypatch.setattr(api, 'knowledge_fingerprints', lambda *args: {})
    with pytest.raises(Exception, match='calibration missing'):
        with api.create_live_knowledge_gate(settings(), confidence_path='missing-policy.json'):
            pass
    assert closed == [True]


def test_knowledge_evaluation_never_accepts_all_refusal_or_weak_false_accept():
    from pathlib import Path
    from scripts.evaluate_ch05 import evaluate_knowledge
    cases = json.loads(Path('eval/ch05/knowledge_cases.json').read_text(encoding='utf-8-sig'))
    class Gate:
        def __init__(self, mode): self.mode = mode
        async def prepare(self, question, filters=None):
            case = next(c for c in cases if c['question'] == question)
            passed = self.mode == 'all-pass' or self.mode == 'correct' and case['answerable']
            citations = [{'chunk_id': '1', **case.get('evidence', {}),
                'answer': case.get('evidence', {}).get('quote', '')}] if passed else []
            return {'confidence': {'sufficient': passed, 'score': .9, 'threshold': .5},
                    'stop_reason': None if passed else 'weak_evidence', 'citations': citations,
                    'low_confidence_recorded': not passed, 'logging_error': False}
    for mode, expected in [('all-refuse', False), ('all-pass', False), ('correct', True)]:
        report = asyncio.run(evaluate_knowledge(cases, gate=Gate(mode)))
        assert report['accepted'] is expected
        assert report['acceptance']['answerable_total'] == report['acceptance']['unanswerable_total'] == 6
        assert report['acceptance']['refusal_rate'] == {'all-refuse': 1, 'all-pass': 0, 'correct': .5}[mode]


def test_knowledge_evaluation_data_rejected_before_retrieval():
    from pathlib import Path
    from scripts.evaluate_ch05 import validate_knowledge_cases
    cases = json.loads(Path('eval/ch05/knowledge_cases.json').read_text(encoding='utf-8-sig'))
    validate_knowledge_cases(cases)
    for changed in [[dict(cases[0], split='acceptance'), *cases[1:]],
                    [dict(cases[0], evidence={}), *cases[1:]],
                    [dict(cases[0], answerable='true'), *cases[1:]]]:
        with pytest.raises(ValueError): validate_knowledge_cases(changed)


def test_calibration_reads_only_calibration_split_and_preserves_first_artifact(tmp_path, monkeypatch):
    from pathlib import Path
    from scripts.evaluate_ch05 import calibrate_knowledge
    api = importlib.import_module('app.workflow_knowledge')
    cases_path = Path('eval/ch05/knowledge_cases.json')
    cases = json.loads(cases_path.read_text(encoding='utf-8-sig'))
    expected_questions = {c['question'] for c in cases if c['split'] == 'calibration'}
    class CalibrationRetrieval:
        corpus = None
        def __init__(self): self.questions = []
        def retrieve(self, query, strategy, filters):
            assert query.raw_question in expected_questions  # Held-out requests are forbidden here.
            self.questions.append(query.raw_question)
            case = next(c for c in cases if c['question'] == query.raw_question)
            return retrieved(.9 if case['answerable'] else .1)
    monkeypatch.setattr(api, 'knowledge_fingerprints', lambda *args: {'query_mode': api.QUERY_MODE})
    retrieval = CalibrationRetrieval()
    path = tmp_path/'confidence.json'
    policy = asyncio.run(calibrate_knowledge(cases, retrieval=retrieval, settings=settings(),
        cases_path=cases_path, path=path))
    assert .1 < policy.thresholds['hybrid_rerank'] <= .9
    assert set(retrieval.questions) == expected_questions and len(retrieval.questions) == 12
    initial = path.with_name('confidence.initial.json').read_bytes()
    asyncio.run(calibrate_knowledge(cases, retrieval=retrieval, settings=settings(), cases_path=cases_path, path=path))
    assert path.with_name('confidence.initial.json').read_bytes() == initial
    assert len(json.loads(path.read_text(encoding='utf-8'))['strategies']['hybrid_rerank']['samples']) == 12


def test_raw_query_policy_rejects_wrong_fingerprint_and_nonfinite_threshold(tmp_path):
    api = importlib.import_module('app.workflow_knowledge')
    artifact = {'version': 1, 'fingerprints': {'query_mode': api.QUERY_MODE},
                'strategies': {'hybrid_rerank': {'threshold': .5}}}
    path = tmp_path/'confidence.json'
    path.write_text(json.dumps(artifact), encoding='utf-8')
    assert api.load_knowledge_policy(path, artifact['fingerprints']).thresholds == {'hybrid_rerank': .5}
    with pytest.raises(Exception, match='incompatible'):
        api.load_knowledge_policy(path, {'query_mode': 'rewritten'})
    artifact['strategies']['hybrid_rerank']['threshold'] = float('nan')
    path.write_text(json.dumps(artifact), encoding='utf-8')
    with pytest.raises(Exception, match='incompatible'):
        api.load_knowledge_policy(path, artifact['fingerprints'])


def test_calibration_uses_midpoint_margin_without_changing_calibration_error_optimum(tmp_path, monkeypatch):
    from pathlib import Path
    from scripts.evaluate_ch05 import calibrate_knowledge
    api = importlib.import_module('app.workflow_knowledge')
    cases_path = Path('eval/ch05/knowledge_cases.json')
    cases = json.loads(cases_path.read_text(encoding='utf-8-sig'))
    class RetrievalScores:
        corpus = None
        def retrieve(self, query, strategy, filters):
            case = next(c for c in cases if c['question'] == query.raw_question)
            assert case['split'] == 'calibration'
            return retrieved(8 if case['answerable'] else 2)
    monkeypatch.setattr(api, 'knowledge_fingerprints', lambda *args: {})
    policy = asyncio.run(calibrate_knowledge(cases, retrieval=RetrievalScores(), settings=settings(),
        cases_path=cases_path, path=tmp_path/'confidence.json'))
    assert policy.thresholds['hybrid_rerank'] == 5
    strategy = json.loads((tmp_path/'confidence.json').read_text(encoding='utf-8'))['strategies']['hybrid_rerank']
    assert strategy['false_accept'] == strategy['false_reject'] == 0


@pytest.mark.parametrize('positive,negative', [(2, 2), (1, 2)])
def test_midpoint_calibration_fails_closed_on_overlapping_labels(tmp_path, monkeypatch, positive, negative):
    from pathlib import Path
    from scripts.evaluate_ch05 import calibrate_knowledge
    api = importlib.import_module('app.workflow_knowledge')
    cases_path = Path('eval/ch05/knowledge_cases.json')
    cases = json.loads(cases_path.read_text(encoding='utf-8-sig'))
    class RetrievalScores:
        corpus = None
        def retrieve(self, query, strategy, filters):
            case = next(c for c in cases if c['question'] == query.raw_question)
            return retrieved(positive if case['answerable'] else negative)
    monkeypatch.setattr(api, 'knowledge_fingerprints', lambda *args: {})
    with pytest.raises(ValueError, match='not separable'):
        asyncio.run(calibrate_knowledge(cases, retrieval=RetrievalScores(), settings=settings(),
            cases_path=cases_path, path=tmp_path/'confidence.json'))
    assert not (tmp_path/'confidence.json').exists()


def test_confirmatory_evaluation_never_requests_development_acceptance_cases():
    from pathlib import Path
    from scripts.evaluate_ch05 import evaluate_knowledge
    cases = json.loads(Path('eval/ch05/knowledge_cases.json').read_text(encoding='utf-8-sig'))
    confirmatory = json.loads(Path('eval/ch05/knowledge_confirmation_cases.json').read_text(encoding='utf-8-sig'))
    class Gate:
        async def prepare(self, question, filters=None):
            case = next(c for c in confirmatory if c['question'] == question)
            return {'confidence': {'sufficient': case['answerable']},
                'stop_reason': None if case['answerable'] else 'weak_evidence',
                'citations': [{'chunk_id': '1', **case.get('evidence', {}),
                    'answer': case.get('evidence', {}).get('quote', '')}] if case['answerable'] else [],
                'low_confidence_recorded': not case['answerable']}
    report = asyncio.run(evaluate_knowledge(cases + confirmatory, gate=Gate(), split='confirmatory'))
    assert report['accepted'] and len(report['cases']) == 12
    assert all(c['split'] == 'confirmatory' for c in report['cases'])
