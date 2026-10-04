"""Short MySQL transactions surrounding recoverable vector I/O."""
from sqlalchemy import select

from app.embedding import validate_vector
from app.knowledge_db import KnowledgeChunk


def knowledge_text(row) -> str:
    return f'category: {row.category}\nquestions: {row.questions}\nanswer: {row.answer}'


def index_pending(session_factory, embedder, store, limit: int = 100) -> int:
    if limit <= 0:
        raise ValueError('limit must be positive')
    # Snapshot scalar values before closing the read transaction and connection.
    with session_factory() as session:
        pending = [(row.id, knowledge_text(row)) for row in session.scalars(
            select(KnowledgeChunk).where(KnowledgeChunk.vectorize_status == 'pending')
            .order_by(KnowledgeChunk.id).limit(limit))]
    completed = 0
    for id, content in pending:
        vectors = embedder.encode([content])
        if len(vectors) != 1:
            raise ValueError('Expected one embedding per chunk')
        vector = validate_vector(vectors[0])
        returned_id = store.upsert(id, vector)
        if type(returned_id) is not int or returned_id != id:
            raise ValueError('Milvus upsert returned an unexpected primary key')
        with session_factory() as session, session.begin():
            row = session.get(KnowledgeChunk, id, with_for_update=True)
            # An explicit concurrent edit/reset must never get the stale vector.
            if row is not None and row.vectorize_status == 'pending' and knowledge_text(row) == content:
                row.vector_id = str(id)
                row.vectorize_status = 'done'
                completed += 1
    return completed
