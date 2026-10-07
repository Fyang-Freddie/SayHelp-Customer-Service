"""Thin pre-output adapter over chapter 4 retrieval, confidence and whole evidence."""
import asyncio
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from app.evaluation.calibration import ConfidenceConfigurationError, ConfidencePolicy, fingerprints
from app.evidence import select_prompt_evidence
from app.knowledge_types import KnowledgeFilters, PreparedQuery
from app.rag_retrieval import IndexStaleError, RagRetrieval
from app.workflow_log import WorkflowLog

QUERY_MODE = 'raw-question-v1: PreparedQuery(question, question, question); synonyms=[]'
CALIBRATION_METHOD = 'separable-label-gap-midpoint-v1'


def knowledge_fingerprints(corpus, settings, cases_path):
    """Only calibration labels bind the threshold; acceptance labels never tune it."""
    cases = json.loads(Path(cases_path).read_text(encoding='utf-8-sig'))
    calibration = [case for case in cases if case['split'] == 'calibration']
    expected = fingerprints(corpus, settings, cases_path)
    expected.update(query_mode=QUERY_MODE, query_model='none', calibration_method=CALIBRATION_METHOD, calibration_digest=hashlib.sha256(
        json.dumps(calibration, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    return expected


def load_knowledge_policy(path, expected):
    try:
        artifact = json.loads(Path(path).read_text(encoding='utf-8'))
        threshold = artifact['strategies']['hybrid_rerank']['threshold']
        if (artifact['version'] != 1 or artifact['fingerprints'] != expected
                or set(artifact['strategies']) != {'hybrid_rerank'}
                or type(threshold) not in (int, float) or not math.isfinite(threshold)):
            raise ValueError()
        return ConfidencePolicy({'hybrid_rerank': threshold}, expected)
    except (OSError, ValueError, KeyError, TypeError):
        raise ConfidenceConfigurationError('Chapter 5 raw-query calibration missing or incompatible') from None


class KnowledgeGate:
    def __init__(self, retrieval: RagRetrieval, confidence_policy: ConfidencePolicy, settings, *, log=None):
        self.retrieval = retrieval
        self.confidence_policy = confidence_policy
        self.settings = settings
        self.log = log if log is not None else WorkflowLog()

    async def prepare(self, question: str, filters: KnowledgeFilters | None = None) -> dict:
        """Retrieve then assess before any Agent output. Graph gate consumes this decision.

        Results contain only whole selected chunks. Blocked paths expose no evidence.
        Cancellation propagates rather than becoming a normal refusal.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Knowledge question must be nonempty')
        query = PreparedQuery(question, question, question)
        score = None
        threshold = self.confidence_policy.thresholds['hybrid_rerank']
        try:
            result = await asyncio.to_thread(self.retrieval.retrieve, query, 'hybrid_rerank', filters)
            if any(not math.isfinite(item.score) for item in result.ranked):
                raise ValueError('Invalid retrieval score')
            score = result.ranked[0].score if result.ranked else None
            decision = self.confidence_policy.assess(result)
            selected = select_prompt_evidence(query, result, self.settings) if decision.sufficient else None
            sufficient = decision.sufficient and bool(selected.citations)
            reason = decision.reason if not decision.sufficient or sufficient else '上下文预算内没有完整证据'
        except Exception as error:
            stop = 'index_stale' if isinstance(error, IndexStaleError) else 'retrieval_error'
            logging_error = False
            try:
                self.log.write('retrieval_error', {'stop_reason': stop, 'error_type': type(error).__name__})
            except Exception:
                logging_error = True
            return {'evidence': [], 'citations': [], 'confidence': {'sufficient': False,
                    'reason': stop, 'score': None, 'threshold': threshold}, 'stop_reason': stop,
                    'low_confidence_recorded': False, 'logging_error': logging_error}
        confidence = {'sufficient': sufficient, 'reason': reason, 'score': score, 'threshold': threshold}
        if not sufficient:
            try:
                self.log.write('weak_evidence', {'question': question, 'score': score,
                    'threshold': threshold, 'stop_reason': 'weak_evidence'})
            except Exception:
                return {'evidence': [], 'citations': [], 'confidence': confidence,
                        'stop_reason': 'low_confidence_log_error', 'low_confidence_recorded': False,
                        'logging_error': True}
            return {'evidence': [], 'citations': [], 'confidence': confidence,
                    'stop_reason': 'weak_evidence', 'low_confidence_recorded': True, 'logging_error': False}
        by_id = {str(item.chunk.id): item for item in result.ranked}
        evidence = [{**asdict(by_id[c['chunk_id']].chunk), 'n': c['n'],
                     'rank': by_id[c['chunk_id']].rank, 'score': by_id[c['chunk_id']].score}
                    for c in selected.citations]
        return {'evidence': evidence, 'citations': selected.citations, 'confidence': confidence,
                'stop_reason': None, 'low_confidence_recorded': False, 'logging_error': False}


@contextmanager
def create_live_retrieval(settings):
    """Read existing SQL/Milvus corpus without ingestion, FastAPI or model warmup."""
    from app.ch04_ingest import ROOT, read_current_corpus
    from app.db import make_session_factory
    from app.embedding import BgeM3Embedder
    from app.hybrid_store import HybridMilvusStore
    from app.reranking import BgeReranker

    sessions = make_session_factory(settings.database_url)
    store = None
    try:
        corpus = read_current_corpus(sessions, ROOT / 'eval/ch04/corpus.json', settings.knowledge_collection)
        store = HybridMilvusStore(uri=settings.milvus_uri, collection_name=settings.knowledge_collection)
        retrieval = RagRetrieval(sessions, BgeM3Embedder(cache_dir=settings.bge_cache_dir),
                                 store, BgeReranker(cache_dir=settings.bge_cache_dir), corpus, diagnostics=False)
        yield retrieval
    finally:
        try:
            if store is not None:
                store.close()
        finally:
            sessions.kw['bind'].dispose()


@contextmanager
def create_live_knowledge_gate(settings, *, log=None, confidence_path=None):
    """Own read-only retrieval resources and validate raw-query calibration."""
    from app.ch04_ingest import ROOT
    with create_live_retrieval(settings) as retrieval:
        expected = knowledge_fingerprints(retrieval.corpus, settings, ROOT / 'eval/ch05/knowledge_cases.json')
        policy = load_knowledge_policy(confidence_path or ROOT / 'eval/ch05/confidence.json', expected)
        yield KnowledgeGate(retrieval, policy, settings, log=log)
