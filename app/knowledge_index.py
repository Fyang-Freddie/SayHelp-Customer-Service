"""Short MySQL transactions surrounding serialized, recoverable vector I/O."""
from contextlib import contextmanager
from sqlalchemy import func, select, text

from app.embedding import EmbeddingInputTooLongError, validate_vector
from app.knowledge_db import KnowledgeChunk

_LOCK_NAME = "CONCAT('sayhelp_index_', MD5(DATABASE()))"


def knowledge_text(row) -> str:
    return f'category: {row.category}\nquestions: {row.questions}\nanswer: {row.answer}'


def count_pending(session_factory) -> int:
    with session_factory() as session:
        return session.scalar(select(func.count()).select_from(KnowledgeChunk)
                              .where(KnowledgeChunk.vectorize_status == 'pending'))


@contextmanager
def _index_writer(session_factory):
    with session_factory() as probe:
        engine = probe.get_bind()
    # Connection-scoped lock, not a transaction lock. Keep this connection pinned
    # through inference/upsert/commit so every process has the same ownership gate.
    with engine.connect() as connection:
        locked = connection.execute(text(f'SELECT GET_LOCK({_LOCK_NAME}, 0)')).scalar()
        if locked != 1:
            connection.rollback()
            raise RuntimeError('Knowledge indexing is already running or its lock is unavailable')
        try:
            connection.commit()
            yield
        finally:
            try:
                connection.rollback()
                connection.execute(text(f'SELECT RELEASE_LOCK({_LOCK_NAME})'))
                connection.commit()
            except Exception:
                connection.invalidate()
                raise


def index_pending(session_factory, embedder, store, limit: int = 100) -> int:
    if limit <= 0:
        raise ValueError('limit must be positive')
    with _index_writer(session_factory):
        with session_factory() as session:
            pending = [(row.id, knowledge_text(row), row.content_type, row.section_path) for row in session.scalars(
                select(KnowledgeChunk).where(KnowledgeChunk.vectorize_status == 'pending')
                .order_by(KnowledgeChunk.id).limit(limit))]
        completed = 0
        for id, content, content_type, section_path in pending:
            try:
                vectors = embedder.encode([content])
            except EmbeddingInputTooLongError as error:
                raise EmbeddingInputTooLongError(
                    error.input_index, error.token_count, error.limit, chunk_id=id,
                    content_type=content_type, section_path=section_path) from error
            if len(vectors) != 1:
                raise ValueError('Expected one embedding per chunk')
            returned_id = store.upsert(id, validate_vector(vectors[0]))
            if type(returned_id) is not int or returned_id != id:
                raise ValueError('Milvus upsert returned an unexpected primary key')
            with session_factory() as session, session.begin():
                row = session.get(KnowledgeChunk, id, with_for_update=True)
                # Concurrent explicit edits stay pending; the next writer retries.
                if row is not None and row.vectorize_status == 'pending' and knowledge_text(row) == content:
                    row.vector_id = str(id)
                    row.vectorize_status = 'done'
                    completed += 1
        return completed
