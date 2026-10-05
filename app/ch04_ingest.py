"""Manifest-only real corpus ingestion; old chains remain available for audit."""
import hashlib
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import undefer

from app.knowledge_chunking import chunk_markdown
from app.knowledge_db import KnowledgeChunk
from app.knowledge_ingest import _ingestion_session
from app.knowledge_types import CorpusSnapshot, EvidenceChunk

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = {
    'knowledge_db/after-sales-manual.md': 'manual',
    'knowledge_db/product-faq.md': 'faq',
    'knowledge_db/product-specs.md': 'manual',
    'knowledge_db/returns-policy.md': 'policy',
}
_FIELDS = ('category','questions','answer','section_path','content_type','is_key_clause')


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def product_category(section: str, source_file: str) -> str:
    for needle, category in (('猫砂盆','猫砂盆'),('饮水机','饮水机'),('猫爬架','猫爬架'),('加热垫','加热垫'),('喂食器','喂食器'),('摄像头','摄像头')):
        if needle in section:
            return category
    return {'knowledge_db/returns-policy.md':'政策','knowledge_db/after-sales-manual.md':'售后',
            'knowledge_db/product-faq.md':'购物','knowledge_db/product-specs.md':'商品'}[source_file]


def load_documents(manifest: Path) -> list[tuple[str,str,str,str]]:
    try:
        data = json.loads(Path(manifest).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise ValueError('Corpus manifest cannot be read') from None
    if not isinstance(data,dict) or set(data) != {'documents'} or not isinstance(data['documents'],list):
        raise ValueError('Corpus manifest must contain documents')
    seen, documents = set(), []
    root = ROOT.resolve()
    for entry in data['documents']:
        if not isinstance(entry,dict) or set(entry) != {'source_file','content_type'}:
            raise ValueError('Invalid corpus manifest entry')
        name, kind = entry['source_file'], entry['content_type']
        if not isinstance(name,str) or name not in DOCUMENTS or name in seen or kind != DOCUMENTS[name]:
            raise ValueError('Manifest source is not in the explicit real corpus')
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != (root/'knowledge_db').resolve():
            raise ValueError('Corpus source must be a local whitelisted Markdown file')
        body = path.read_text(encoding='utf-8')
        documents.append((name,kind,body,digest_text(body)))
        seen.add(name)
    if seen != set(DOCUMENTS):
        raise ValueError('Manifest must contain exactly the four real corpus documents')
    return documents


def row_evidence(row: KnowledgeChunk) -> EvidenceChunk:
    return EvidenceChunk(id=row.id, question=row.questions, answer=row.answer,
                         category=row.category, product_category=row.product_category,
                         content_type=row.content_type, section_path=row.section_path,
                         source_file=row.source_file, source_start_line=row.source_start_line,
                         source_end_line=row.source_end_line, source_digest=row.source_digest)


def index_text(chunk: EvidenceChunk) -> str:
    return f'category: {chunk.category}\nquestions: {chunk.question}\nanswer: {chunk.answer}'


def index_metadata(chunk: EvidenceChunk) -> dict[str,str]:
    return {'category': chunk.category, 'product_category': chunk.product_category or '',
            'content_type': chunk.content_type or '', 'source_digest': chunk.source_digest or ''}


def payload_digest(chunk: EvidenceChunk) -> str:
    payload = {'text': index_text(chunk), **index_metadata(chunk)}
    return digest_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',',':')))


def _keys(row):
    return tuple(getattr(row,name) for name in _FIELDS)


def _unique_chain(rows, drafts, source_file, source_digest):
    by_id = {row.id:row for row in rows}
    keys = [_keys(draft) for draft in drafts]
    matches = []
    for head in rows:
        if head.prev_chunk_id is not None:
            continue
        chain, row = [], head
        for expected in keys:
            if row is None or _keys(row) != expected or row.prev_chunk_id != (chain[-1].id if chain else None):
                break
            legacy = row.source_file is None and row.source_digest is None
            exact = row.source_file == source_file and row.source_digest == source_digest
            if not (legacy or exact):
                break
            chain.append(row)
            next_id = row.next_chunk_id
            row = by_id.get(next_id)
        else:
            if next_id is None:
                matches.append(chain)
    return matches[0] if len(matches)==1 else None


def ingest_corpus(session_factory, manifest: Path) -> CorpusSnapshot:
    # All sources/positions are validated before any database write.
    parsed = [(name,kind,digest,chunk_markdown(body, content_type=kind))
              for name,kind,body,digest in load_documents(manifest)]
    if any(not drafts for _,_,_,drafts in parsed):
        raise ValueError('Corpus document has no knowledge chunks')
    files, chunks = {}, {}
    with _ingestion_session(session_factory) as session:
        rows = list(session.scalars(select(KnowledgeChunk).options(undefer('*')).order_by(KnowledgeChunk.id)))
        for name,kind,digest,drafts in parsed:
            chain = _unique_chain(rows,drafts,name,digest)
            if chain is None:
                chain = [KnowledgeChunk(**dict(zip(_FIELDS,_keys(draft))), vectorize_status='pending') for draft in drafts]
                session.add_all(chain)
                session.flush()
                for i,row in enumerate(chain):
                    row.prev_chunk_id = chain[i-1].id if i else None
                    row.next_chunk_id = chain[i+1].id if i+1<len(chain) else None
                rows.extend(chain)
            files[name] = digest
            for row,draft in zip(chain,drafts,strict=True):
                row.source_file = name
                row.source_start_line = draft.source_start_line
                row.source_end_line = draft.source_end_line
                row.source_digest = digest
                row.product_category = product_category(draft.section_path,name)
                chunks[row.id] = row_evidence(row)
    payloads = {id_:payload_digest(chunk) for id_,chunk in chunks.items()}
    identity = {'files':files, 'payloads': sorted(payloads.values())}
    corpus_digest = digest_text(json.dumps(identity,sort_keys=True,ensure_ascii=False,separators=(',',':')))
    return CorpusSnapshot(files=files,chunks=chunks,payload_digests=payloads,corpus_digest=corpus_digest)


def verify_source_versions(snapshot: CorpusSnapshot) -> None:
    for name,digest in snapshot.files.items():
        path = ROOT / name
        if name not in DOCUMENTS or path.is_symlink() or digest_text(path.read_text(encoding='utf-8')) != digest:
            raise RuntimeError('Corpus source changed during indexing; re-ingest before retry')


def read_current_corpus(session_factory,manifest:Path,collection_name=None)->CorpusSnapshot:
    """Read and verify current exact source chains; never ingest or mutate on chat."""
    files,chunks={},{}
    with session_factory() as session:
        rows=list(session.scalars(select(KnowledgeChunk).options(undefer('*')).order_by(KnowledgeChunk.id)))
        for name,kind,body,digest in load_documents(manifest):
            drafts=chunk_markdown(body,content_type=kind)
            chain=_unique_chain([r for r in rows if r.source_file==name and r.source_digest==digest],drafts,name,digest)
            if not chain: raise RuntimeError('Current source chain missing or ambiguous; rebuild knowledge')
            files[name]=digest
            for row,draft in zip(chain,drafts,strict=True):
                c=row_evidence(row)
                if c.source_start_line!=draft.source_start_line or c.source_end_line!=draft.source_end_line or c.product_category!=product_category(draft.section_path,name):
                    raise RuntimeError('Current source provenance differs; rebuild knowledge')
                chunks[row.id]=c
        payloads={id_:payload_digest(c) for id_,c in chunks.items()}
        if collection_name is not None:
            from app.ch04_db import KnowledgeIndexState
            states={r.chunk_id:r for r in session.scalars(select(KnowledgeIndexState).where(KnowledgeIndexState.collection_name==collection_name))}
            if not set(chunks)<=set(states) or any(states[i].status!='done' or states[i].payload_digest!=digest for i,digest in payloads.items()):
                raise RuntimeError('Independent index build is incomplete or stale')
    identity={'files':files,'payloads':sorted(payloads.values())}
    return CorpusSnapshot(files,chunks,payloads,digest_text(json.dumps(identity,sort_keys=True,ensure_ascii=False,separators=(',',':'))))
