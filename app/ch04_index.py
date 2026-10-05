"""Recoverable index writes with short MySQL transactions and verified source state."""
from sqlalchemy import select
from sqlalchemy.orm import undefer

from app.ch04_db import KnowledgeIndexState
from app.ch04_ingest import index_metadata, index_text, payload_digest, row_evidence, verify_source_versions
from app.embedding import validate_vector
from app.knowledge_db import KnowledgeChunk
from app.knowledge_index import _index_writer
from app.knowledge_types import BuildSummary, CorpusSnapshot


def _current_chunk(session_factory, id_):
    with session_factory() as session:
        row = session.get(KnowledgeChunk,id_,options=[undefer('*')])
        return row_evidence(row) if row else None


def _assert_current(session_factory, chunk, digest):
    current = _current_chunk(session_factory,chunk.id)
    if current != chunk or payload_digest(current) != digest:
        raise RuntimeError(f'Index corpus changed at chunk {chunk.id}; re-ingest before retry')


def _entity_matches(entity, chunk):
    expected = {'id':chunk.id,'text':index_text(chunk),**index_metadata(chunk)}
    return isinstance(entity,dict) and all(entity.get(key)==value for key,value in expected.items())


def _set_state(session_factory, collection, chunk, digest, status, *, error=None):
    with session_factory.begin() as session:
        if status=='done':
            row = session.get(KnowledgeChunk,chunk.id,options=[undefer('*')],with_for_update=True)
            if row is None or row_evidence(row) != chunk or payload_digest(row_evidence(row)) != digest:
                raise RuntimeError(f'Index corpus changed at chunk {chunk.id}; refusing acknowledgement')
        state = session.get(KnowledgeIndexState,(collection,chunk.id),with_for_update=True)
        if state is None:
            state = KnowledgeIndexState(collection_name=collection,chunk_id=chunk.id,payload_digest=digest,status=status,error=error)
            session.add(state)
        else:
            state.payload_digest=digest
            state.status=status
            state.error=error


def build_index(session_factory, embedder, store, snapshot: CorpusSnapshot, batch_size=32) -> BuildSummary:
    if type(batch_size) is not int or not 1<=batch_size<=128:
        raise ValueError('batch_size must be within [1,128]')
    if store.collection_name=='knowledge':
        raise ValueError('Old knowledge collection must never be rebuilt by chapter 4')
    if not snapshot.chunks or set(snapshot.payload_digests)!=set(snapshot.chunks):
        raise ValueError('Corpus snapshot is incomplete')
    collection=store.collection_name
    with _index_writer(session_factory):
        verify_source_versions(snapshot)
        store.ensure_collection()
        existing=store.read_entities()
        with session_factory() as session:
            states={row.chunk_id:(row.status,row.payload_digest) for row in session.scalars(
                select(KnowledgeIndexState).where(KnowledgeIndexState.collection_name==collection))}
        pending, reused = [], 0
        for id_,chunk in snapshot.chunks.items():
            digest=snapshot.payload_digests[id_]
            _assert_current(session_factory,chunk,digest)
            if states.get(id_)==('done',digest) and _entity_matches(existing.get(id_),chunk):
                reused+=1
            else:
                _set_state(session_factory,collection,chunk,digest,'pending')
                pending.append(chunk)
        indexed=0
        for start in range(0,len(pending),batch_size):
            batch=pending[start:start+batch_size]
            # No MySQL transaction surrounds inference, vector upsert or verification.
            try:
                vectors=embedder.encode([index_text(chunk) for chunk in batch])
                if len(vectors)!=len(batch):
                    raise ValueError('Wrong embedding count')
                vectors=[validate_vector(vector) for vector in vectors]
            except Exception:
                raise RuntimeError(f'Index embedding failed at chunk {batch[0].id}; pending rows can be retried') from None
            for chunk,vector in zip(batch,vectors,strict=True):
                digest=snapshot.payload_digests[chunk.id]
                try:
                    id_=store.upsert(chunk.id,vector,index_text(chunk),index_metadata(chunk))
                    if type(id_) is not int or id_!=chunk.id:
                        raise ValueError('Wrong vector acknowledgement ID')
                    entity=store.read_entities([chunk.id]).get(chunk.id)
                    if not _entity_matches(entity,chunk):
                        raise ValueError('Vector entity did not preserve expected text and metadata')
                    _set_state(session_factory,collection,chunk,digest,'done')
                except Exception:
                    try:
                        _set_state(session_factory,collection,chunk,digest,'pending',error='Index write or acknowledgement failed; retry after checking dependencies')
                    except Exception:
                        pass  # Preserve the original safe failure; no claim of recorded status.
                    raise RuntimeError(f'Index write failed at chunk {chunk.id}; pending rows can be retried') from None
                indexed+=1
        verify_source_versions(snapshot)
        for id_,chunk in snapshot.chunks.items():
            _assert_current(session_factory,chunk,snapshot.payload_digests[id_])
        stale=sorted(set(store.read_entities())-set(snapshot.chunks))
        if stale:
            store.delete_ids(stale)
        verified=store.read_entities()
        if set(verified)!=set(snapshot.chunks) or not all(_entity_matches(verified[id_],chunk) for id_,chunk in snapshot.chunks.items()):
            raise RuntimeError('Index final verification failed; do not switch online retrieval')
        return BuildSummary(indexed=indexed,reused=reused,removed=len(stale),verified=len(verified),corpus_digest=snapshot.corpus_digest)
