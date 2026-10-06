"""Four native retrieval strategies with authoritative ordered MySQL hydration."""
from time import perf_counter
from sqlalchemy import select
from sqlalchemy.orm import undefer
from app.ch04_ingest import row_evidence,index_text,index_metadata,verify_source_versions
from app.knowledge_db import KnowledgeChunk
from app.knowledge_types import RetrievalResult,RankedChunk,StageHit,KnowledgeFilters

class IndexStaleError(RuntimeError): pass

class RagRetrieval:
    def __init__(self,session_factory,embedder,store,reranker,corpus):
        self.session_factory=session_factory;self.embedder=embedder;self.store=store;self.reranker=reranker;self.corpus=corpus
    def retrieve(self,query,strategy,filters=None):
        if strategy not in {'dense','bm25','hybrid','hybrid_rerank'}: raise ValueError('Invalid retrieval strategy')
        filters=filters if filters is not None else KnowledgeFilters()
        try: verify_source_versions(self.corpus)
        except (OSError,RuntimeError): raise IndexStaleError('Corpus source changed; rebuild index') from None
        timings={};started=perf_counter();vector=None
        before=perf_counter()
        entities=self.store.read_entities()
        if set(entities)!=set(self.corpus.chunks):
            raise IndexStaleError('Indexed corpus missing or contains stale IDs; rebuild index')
        for id_,chunk in self.corpus.chunks.items():
            entity=entities[id_]
            if entity.get('id')!=id_ or entity.get('text')!=index_text(chunk) or any(entity.get(k)!=v for k,v in index_metadata(chunk).items()):
                raise IndexStaleError('Indexed corpus text or metadata changed; rebuild index')
        timings['index_verification']=(perf_counter()-before)*1000
        if strategy!='bm25':
            before=perf_counter();vector=self.embedder.encode([query.search_text])[0];timings['embedding']=(perf_counter()-before)*1000
        stored=self.store.retrieve(query,vector,strategy,filters)
        timings.update(stored.timings_ms)
        hits=stored.hits
        ids=[h.chunk_id for h in hits]
        if len(ids)>50 or len(set(ids))!=len(ids): raise IndexStaleError('Invalid indexed candidate IDs')
        with self.session_factory() as session:
            rows=list(session.scalars(select(KnowledgeChunk).options(undefer('*')).where(KnowledgeChunk.id.in_(ids))))
            chunks={row.id:row_evidence(row) for row in rows}
        candidates=[]
        for hit in hits:
            c=chunks.get(hit.chunk_id);entity=stored.entities.get(hit.chunk_id)
            if c is None or c!=self.corpus.chunks.get(hit.chunk_id) or entity is None:
                raise IndexStaleError('Source chunk missing or changed; rebuild index')
            if entity.get('id')!=c.id or entity.get('text')!=index_text(c) or any(entity.get(k)!=v for k,v in index_metadata(c).items()):
                raise IndexStaleError('Indexed text or metadata differs from authoritative source')
            if any(getattr(c,k)!=v for k,v in filters.model_dump(exclude_none=True).items()):
                raise IndexStaleError('Indexed filter returned an inconsistent source')
            candidates.append(RankedChunk(c,hit.rank,hit.score))
        ranked=candidates[:10];stages=dict(stored.stage_hits)
        if strategy=='hybrid_rerank':
            before=perf_counter();ranked=self.reranker.rerank(query.standard_question,[i.chunk for i in candidates],limit=10)
            timings['reranker']=(perf_counter()-before)*1000
            stages['reranker']=[StageHit(i.chunk.id,i.rank,i.score,'reranker') for i in ranked]
        timings['retrieval_total']=(perf_counter()-started)*1000
        return RetrievalResult(strategy,query,candidates,ranked,stages,timings)
