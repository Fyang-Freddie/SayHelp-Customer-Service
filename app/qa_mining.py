"""Mine complete turns, commit staging, then globally deduplicate into pending QA."""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr
from sqlalchemy import select, text

from app.db import Message
from app.embedding import BgeM3Embedder, validate_vector
from app.history import completed_turns
from app.knowledge_db import KnowledgeChunk, QaExtractionStaging
from app.knowledge_ingest import _LOCK_NAME as INGEST_LOCK
from app.qa_prompts import extraction_messages, equivalence_messages

# Candidate gates only; never sufficient for deletion. Provisional until local
# BGE-M3 labeled evaluation is available. Both meanings need model confirmation.
QUESTION_CANDIDATE_SCORE = 0.85
ANSWER_CANDIDATE_SCORE = 0.90
MAX_REQUEST_CHARS = 12000
MAX_TURN_CHARS = 4000
_CREDENTIAL = re.compile(r'(?i)(api[_ -]?key|password|passwd|secret|token)\s*(?:[:=：]|\bis\b)|(?:密码|口令|验证码)\s*(?:是|为|[:=：])|\bsk-[A-Za-z0-9_-]{12,}|Bearer\s+\S+')
_LOCK = "CONCAT('sayhelp_qa_', MD5(DATABASE()))"


class Pair(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_ref: StrictStr = Field(min_length=1, max_length=255)
    question: StrictStr = Field(min_length=1, max_length=4000)
    answer: StrictStr = Field(min_length=1, max_length=4000)


class Extraction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    pairs: list[Pair] = Field(max_length=100)


class Equivalence(BaseModel):
    model_config = ConfigDict(extra='forbid')
    equivalent: StrictBool


@dataclass(frozen=True)
class MiningSummary:
    staged: int = 0
    kept: int = 0
    discarded: int = 0


def _json(llm, messages, schema):
    raw = llm.invoke(messages).content
    if not isinstance(raw, str) or len(raw) > 50000:
        raise ValueError('QA model must return bounded JSON text')
    def invalid_constant(value):
        raise ValueError('Non-finite JSON value')
    try:
        return schema.model_validate(json.loads(raw, parse_constant=invalid_constant))
    except (ValueError, TypeError) as exc:
        raise ValueError('QA model returned invalid structured JSON') from exc


def _normalize(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def _pair_key(pair):
    return _normalize(pair['question']), _normalize(pair['answer'])


@contextmanager
def _mining_lock(session_factory, lock_name=_LOCK):
    with session_factory() as probe:
        engine = probe.get_bind()
    with engine.connect() as connection:
        if connection.execute(text(f'SELECT GET_LOCK({lock_name}, 0)')).scalar() != 1:
            connection.rollback()
            raise RuntimeError('Conversation mining is already running')
        try:
            connection.commit()
            yield
        finally:
            try:
                connection.rollback()
                connection.execute(text(f'SELECT RELEASE_LOCK({lock_name})'))
                connection.commit()
            except Exception:
                connection.invalidate()
                raise


def _eligible_conversations(session_factory):
    # A completed TURN is eligible even while its conversation remains ongoing.
    # The snapshot's source IDs stay stable when later turns are appended.
    with session_factory() as session:
        messages = session.scalars(select(Message).order_by(Message.conversation_id, Message.id)).all()
        seen = set(session.scalars(select(QaExtractionStaging.source_ref)))
    groups = OrderedDict()
    for row in messages:
        groups.setdefault(row.conversation_id, []).append(row)
    for conversation_id, messages in groups.items():
        turns = []
        group = []
        def append_turn(group):
            completed = completed_turns(group)
            if not completed:
                return
            user, final = completed[0][0].content, completed[0][-1].content
            ref = f'conversation:{conversation_id}:turn:{group[0].id}-{group[-1].id}'
            if ref in seen or not user.strip() or not final.strip():
                return
            if len(user) + len(final) > MAX_TURN_CHARS or _CREDENTIAL.search(user + '\n' + final):
                return
            turns.append({'source_ref':ref, 'user':user, 'assistant':final})
        for row in messages:
            if row.role == 'user':
                append_turn(group)
                group = [row]
            elif group:
                group.append(row)
        append_turn(group)
        if turns:
            yield {'id':conversation_id, 'turns':turns}


def _batches(conversations, batch_size):
    batch = []
    for conversation in conversations:
        # Split only between complete turns; never truncate evidence.
        for turn in conversation['turns']:
            item = {'id':conversation['id'], 'turns':[turn]}
            # Escaped control characters can expand a raw eligible turn beyond
            # the serialized request budget even when this batch is empty.
            if len(json.dumps({'conversations':[item]}, ensure_ascii=False)) > MAX_REQUEST_CHARS:
                continue
            candidate = [*batch, item]
            size = len(json.dumps({'conversations':candidate}, ensure_ascii=False))
            if batch and (len({c['id'] for c in candidate}) > batch_size or size > MAX_REQUEST_CHARS):
                yield batch
                batch = []
            if batch and batch[-1]['id'] == item['id']:
                batch[-1]['turns'].append(turn)
            else:
                batch.append(item)
    if batch:
        yield batch


def _cosine(left, right):
    denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    if not denominator:
        raise ValueError('Zero embedding cannot establish equivalence')
    return sum(x*y for x, y in zip(left, right)) / denominator


def _dedup(staged, existing, llm, embedder):
    pairs = [*existing, *[{'question':r.question,'answer':r.answer} for r in staged]]
    if not staged:
        return {}
    vectors = None
    accepted = list(range(len(existing)))
    accepted_keys = {_pair_key(pair) for pair in existing}
    decisions = {}
    for offset, row in enumerate(staged, start=len(existing)):
        current = pairs[offset]
        if _pair_key(current) in accepted_keys:
            decisions[row.id] = 'discarded'
            continue
        duplicate = False
        for index in accepted:
            previous = pairs[index]
            if vectors is None:
                vectors = embedder.encode([s for p in pairs for s in (p['question'], p['answer'])])
                if len(vectors) != len(pairs) * 2:
                    raise ValueError('QA embedding count mismatch')
                vectors = [validate_vector(v) for v in vectors]
            # Different numeric conditions cannot be discarded, even if the
            # embedding/model regards answers as near-identical.
            if re.findall(r'\d+(?:\.\d+)?', current['answer']) != re.findall(r'\d+(?:\.\d+)?', previous['answer']):
                continue
            if (_cosine(vectors[2*offset], vectors[2*index]) >= QUESTION_CANDIDATE_SCORE
                    and _cosine(vectors[2*offset+1], vectors[2*index+1]) >= ANSWER_CANDIDATE_SCORE
                    and _json(llm, equivalence_messages(previous, current), Equivalence).equivalent):
                duplicate = True
                break
        decisions[row.id] = 'discarded' if duplicate else 'kept'
        if not duplicate:
            accepted.append(offset)
            accepted_keys.add(_pair_key(current))
    return decisions


def mine_conversations(session_factory, llm, *, batch_size=20, embedder=None) -> MiningSummary:
    """Stage each bounded request, then finalize all nonterminal rows atomically.

    Empty extraction results have no durable marker in the supplied schema and
    are safely re-evaluated on rerun. Successful rows retain exact turn provenance.
    Model/BGE failure leaves committed extracted rows available for retry.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    staged_count = 0
    with _mining_lock(session_factory):
        for batch in _batches(_eligible_conversations(session_factory), batch_size):
            evidence = {t['source_ref']:t for c in batch for t in c['turns']}
            result = _json(llm, extraction_messages(batch), Extraction)
            for pair in result.pairs:
                source = evidence.get(pair.source_ref)
                if (source is None or not pair.question.strip() or not pair.answer.strip()
                        or pair.question not in source['user'] or pair.answer not in source['assistant']
                        or _CREDENTIAL.search(pair.question + '\n' + pair.answer)):
                    raise ValueError('Unsupported QA extraction evidence')
            batch_no = 'qa-' + sha256('\n'.join(evidence).encode()).hexdigest()[:48]
            with session_factory() as session, session.begin():
                for pair in result.pairs:
                    session.add(QaExtractionStaging(batch_no=batch_no, source_ref=pair.source_ref,
                        question=pair.question.strip(), answer=pair.answer.strip(), status='extracted'))
            staged_count += len(result.pairs)
        # Hold shared ingestion ownership across snapshot/judgment/final commit,
        # with no SQL transaction open during model/embedding I/O.
        with _mining_lock(session_factory, INGEST_LOCK):
            with session_factory() as session:
                staged = session.scalars(select(QaExtractionStaging).where(
                    QaExtractionStaging.status == 'extracted').order_by(QaExtractionStaging.id)).all()
                existing = [{'question':r.questions, 'answer':r.answer} for r in
                            session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id))]
            decisions = _dedup(staged, existing, llm, embedder or BgeM3Embedder())
            kept = discarded = 0
            if decisions:
                # Share Task 3's ingestion lock to serialize authoritative insertions.
                with session_factory() as session, session.begin():
                    exact = {_pair_key({'question':r.questions,'answer':r.answer}) for r in
                             session.scalars(select(KnowledgeChunk))}
                    for snapshot in staged:
                        row = session.get(QaExtractionStaging, snapshot.id)
                        if row.status != 'extracted':
                            raise RuntimeError('Staging status changed during mining')
                        pair = {'question':row.question,'answer':row.answer}
                        status = decisions[row.id]
                        if _pair_key(pair) in exact:
                            status = 'discarded'
                        row.status = status
                        if status == 'kept':
                            session.add(KnowledgeChunk(category='历史客服对话', questions=row.question,
                                answer=row.answer, content_type='faq', vectorize_status='pending'))
                            exact.add(_pair_key(pair))
                            kept += 1
                        else:
                            discarded += 1
        return MiningSummary(staged_count, kept, discarded)
