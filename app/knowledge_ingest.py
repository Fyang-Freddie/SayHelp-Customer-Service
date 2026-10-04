"""Transactional, repeatable FAQ and Markdown ingestion into pending rows.

Markdown identity is its complete ordered chain of authoritative fields. Without
source IDs in the supplied schema, changed documents append a new intact chain;
existing chains and vector states are never rewritten or removed.
"""

from contextlib import contextmanager
from pathlib import Path
import unicodedata

from sqlalchemy import select, text

from app.db import Faq
from app.knowledge_chunking import chunk_markdown, _fence_open, _fence_close
from app.knowledge_db import KnowledgeChunk

_FIELDS = ('category', 'questions', 'answer', 'section_path', 'content_type', 'is_key_clause')
_LOCK_NAME = "CONCAT('sayhelp_ingest_', MD5(DATABASE()))"


def _normalize(value):
    if not isinstance(value, str):
        return value
    value = unicodedata.normalize('NFC', value).replace('\r\n', '\n').replace('\r', '\n')
    lines = []
    fence = None
    has_fence = False
    for line in value.split('\n'):
        if fence:
            lines.append(line)
            if _fence_close(line, fence):
                fence = None
        else:
            fence = _fence_open(line)
            has_fence = has_fence or fence is not None
            lines.append(line if fence else line.rstrip())
    normalized = '\n'.join(lines)
    # Whitespace within a code fence is authoritative, including at EOF when
    # the fence is unclosed. Ordinary FAQ/prose keeps its previous normalization.
    return normalized.strip('\n') if has_fence else normalized.strip()


def _key(row):
    return tuple(_normalize(getattr(row, field)) for field in _FIELDS)


@contextmanager
def _ingestion_session(session_factory):
    # Pin the physical connection until after COMMIT and RELEASE_LOCK. A Session
    # bound directly to the engine can return its connection at commit too early.
    with session_factory() as probe:
        engine = probe.get_bind()
    with engine.connect() as connection:
        locked = connection.execute(text(f'SELECT GET_LOCK({_LOCK_NAME}, 30)')).scalar()
        if locked != 1:
            connection.rollback()
            raise RuntimeError('Knowledge ingestion is already running or its lock is unavailable')
        try:
            connection.commit()
            with session_factory(bind=connection) as session, session.begin():
                yield session
        finally:
            try:
                connection.rollback()
                connection.execute(text(f'SELECT RELEASE_LOCK({_LOCK_NAME})'))
                connection.commit()
            except Exception:
                # Never return a connection still owning an advisory lock to pool.
                connection.invalidate()
                raise


def _new_row(values):
    return KnowledgeChunk(**dict(zip(_FIELDS, values)), vectorize_status='pending')


def ingest_faq(session_factory) -> list[int]:
    """Copy real FAQ questions without deleting seeds or resetting done rows."""
    with _ingestion_session(session_factory) as session:
        existing = {
            _key(row): row for row in session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id))
            if row.prev_chunk_id is None and row.next_chunk_id is None
        }
        ids = []
        for faq in session.scalars(select(Faq).order_by(Faq.id)):
            values = tuple(_normalize(value) for value in (
                faq.category, faq.question, faq.answer, None, 'faq', 0))
            row = existing.get(values)
            if row is None:
                row = _new_row(values)
                session.add(row)
                session.flush()
                existing[values] = row
            ids.append(row.id)
        return ids


def _find_chain(rows, keys):
    by_id = {row.id: row for row in rows}
    for head in rows:
        if head.prev_chunk_id is not None:
            continue
        chain = []
        row = head
        for key in keys:
            if row is None or _key(row) != key:
                break
            if row.prev_chunk_id != (chain[-1] if chain else None):
                break
            chain.append(row.id)
            next_id = row.next_chunk_id
            row = by_id.get(next_id)
        else:
            if next_id is None:
                return chain
    return None


def ingest_markdown(session_factory, path, content_type) -> list[int]:
    """Append or reuse a whole ordered document chain in one locked transaction."""
    drafts = chunk_markdown(Path(path).read_text(encoding='utf-8'),
                            content_type=_normalize(content_type))
    keys = [_key(draft) for draft in drafts]
    if not keys:
        return []
    with _ingestion_session(session_factory) as session:
        existing = session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)).all()
        ids = _find_chain(existing, keys)
        if ids is not None:
            return ids
        rows = [_new_row(key) for key in keys]
        session.add_all(rows)
        session.flush()
        for index, row in enumerate(rows):
            row.prev_chunk_id = rows[index - 1].id if index else None
            row.next_chunk_id = rows[index + 1].id if index + 1 < len(rows) else None
        return [row.id for row in rows]
