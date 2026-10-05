"""Public history and exact document previews; never infer a source by similarity."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from app.knowledge_chunking import chunk_markdown

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / 'knowledge_db'


def read_document(filename: str, directory: Path = KNOWLEDGE_DIR) -> dict:
    root = directory.resolve()
    path = root / filename
    if (Path(filename).name != filename or '/' in filename or '\\' in filename
            or not filename.endswith('.md') or path.is_symlink()
            or path.resolve().parent != root or not path.is_file()):
        raise FileNotFoundError('Knowledge document not found')
    text = path.read_text(encoding='utf-8')
    return {'filename': filename, 'text': text,
            'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()}


@lru_cache(maxsize=32)
def _document_chunks(path: str, modified: int, size: int):
    # File identity invalidates this cache when a document changes.
    document = read_document(Path(path).name, Path(path).parent)
    return document, chunk_markdown(document['text'], content_type='manual')


def sources_from_tool(content: str | None, directory: Path = KNOWLEDGE_DIR) -> list[dict]:
    try:
        payload = json.loads(content or '')
    except (ValueError, TypeError):
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get('matches'), list):
        return []
    documents = []
    for path in directory.glob('*.md'):
        try:
            stat = path.stat()
            documents.append((path.name, *_document_chunks(str(path), stat.st_mtime_ns, stat.st_size)))
        except (OSError, UnicodeError, ValueError):
            continue
    sources = []
    for number, match in enumerate(payload['matches'], start=1):
        if not isinstance(match, dict) or not all(isinstance(match.get(key), str) for key in ('question', 'answer', 'category')):
            continue
        answer = match['answer'].strip()
        if not answer:
            continue
        candidates = [(name, document, draft) for name, document, drafts in documents
                      for draft in drafts if draft.answer.strip() == answer]
        source = {'n': number, 'question': match['question'], 'answer': match['answer'],
                  'category': match['category'], 'section_path': None, 'source_file': None,
                  'source_line': None, 'document_sha256': None}
        if len(candidates) == 1:
            name, document, draft = candidates[0]
            source.update(source_file=name, section_path=draft.section_path,
                          source_line=draft.source_line, document_sha256=document['sha256'])
        sources.append(source)
    return sources


def public_messages(rows) -> list[dict]:
    result, sources, faq_calls = [], [], set()
    for row in rows:
        if row.role == 'user':
            sources, faq_calls = [], set()
        if row.role == 'assistant' and row.tool_calls:
            faq_calls.update(call.get('id') for call in row.tool_calls
                             if isinstance(call, dict) and call.get('name') == 'query_faq')
            continue
        if row.role == 'tool':
            if row.tool_call_id in faq_calls:
                sources = sources_from_tool(row.content)
            continue
        if row.role in ('user', 'assistant') and row.content:
            result.append({'id': str(row.id), 'role': row.role, 'content': row.content,
                           'created_at': row.created_at.isoformat(),
                           'sources': sources if row.role == 'assistant' else []})
    return result
