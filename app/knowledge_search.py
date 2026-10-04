"""Dense recall with ranked, authoritative MySQL knowledge hydration."""
from dataclasses import dataclass
import math

from sqlalchemy import select

from app.knowledge_db import KnowledgeChunk


@dataclass(frozen=True)
class KnowledgeMatch:
    question: str
    answer: str
    category: str


class KnowledgeSearchUnavailable(RuntimeError):
    """A dependency failed; callers may present a safe temporary miss."""


class KnowledgeSearch:
    def __init__(self, session_factory, embedder, store, *, min_score: float = 0.55):
        if not math.isfinite(min_score) or not -1 <= min_score <= 1:
            raise ValueError('min_score must be finite and within [-1, 1]')
        self.session_factory = session_factory
        self.embedder = embedder
        self.store = store
        self.min_score = min_score

    def search(self, keyword: str, *, limit: int = 5) -> list[KnowledgeMatch]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('limit must be an integer within [1, 100]')
        try:
            # No MySQL transaction is open during local inference or Milvus I/O.
            vectors = self.embedder.encode([keyword])
            if len(vectors) != 1:
                raise ValueError('Expected one query embedding')
            hits = self.store.search(vectors[0], limit=limit)
            ids = [hit.id for hit in hits[:limit]
                   if math.isfinite(hit.score) and hit.score >= self.min_score]
            if not ids:
                return []
            with self.session_factory() as session:
                rows = session.scalars(select(KnowledgeChunk).where(
                    KnowledgeChunk.id.in_(ids), KnowledgeChunk.vectorize_status == 'done')).all()
                by_id = {row.id: row for row in rows}
                return [KnowledgeMatch(question=by_id[id_].questions,
                                       answer=by_id[id_].answer,
                                       category=by_id[id_].category)
                        for id_ in ids if id_ in by_id]
        except Exception:
            # Neither connection details nor private adapter exceptions leave this boundary.
            raise KnowledgeSearchUnavailable('Knowledge search is temporarily unavailable') from None
