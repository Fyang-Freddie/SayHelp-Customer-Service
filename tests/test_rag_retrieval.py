"""Unified retrieval hydrates authoritative rows and never embeds pure BM25."""
import importlib
import math

import pytest
from sqlalchemy import select
from sqlalchemy.orm import undefer
from app.ch04_ingest import index_text,index_metadata
from app.knowledge_db import KnowledgeChunk
from app.knowledge_types import PreparedQuery,KnowledgeFilters,RankedChunk,StageHit,StoreResult
from test_db import database_url,sessions
from test_ch04_ingest import corpus


def api(): return importlib.import_module('app.rag_retrieval')


class Embedder:
    def __init__(self): self.calls=[]
    def encode(self,texts): self.calls.append(texts); return [[1/math.sqrt(1024)]*1024]


class Store:
    collection_name='knowledge_ch04'
    def __init__(self,snapshot):
        self.chunks=sorted(snapshot.chunks.values(),key=lambda c:(c.content_type!='manual',c.id))[:50]
        self.entities={c.id:{'id':c.id,'text':index_text(c),**index_metadata(c)} for c in self.chunks}
        self.calls=[]
    def retrieve(self,query,vector,strategy,filters):
        self.calls.append((query,vector,strategy,filters))
        stage='rrf' if strategy in {'hybrid','hybrid_rerank'} else strategy
        hits=[StageHit(c.id,i+1,.032-i*.0001 if stage=='rrf' else .95-i*.001,stage) for i,c in enumerate(self.chunks)]
        return StoreResult(hits,{stage:hits},self.entities)


class Reranker:
    def __init__(self): self.calls=[]
    def rerank(self,query,chunks,limit=10):
        self.calls.append((query,chunks,limit))
        return [RankedChunk(c,i+1,10-i) for i,c in enumerate(chunks[::-1][:limit])]


@pytest.fixture
def fifty(sessions,corpus):
    ingest,manifest,directory=corpus
    body='# 商品规格手册\n'+ '\n'.join(f'## 测试候选{i}\n占位边界段落{i}。' for i in range(50))
    (directory/'product-specs.md').write_text(body,encoding='utf-8')
    snapshot=ingest.ingest_corpus(sessions,manifest)
    assert len(snapshot.chunks)>=50
    return snapshot


def test_four_strategies_share_query_and_filters_bm25_never_embeds(sessions,fifty):
    snapshot=fifty;embedder=Embedder();store=Store(snapshot);reranker=Reranker()
    retrieval=api().RagRetrieval(sessions,embedder,store,reranker,snapshot)
    query=PreparedQuery('口语问题','标准问题','标准问题\n检索同义词')
    filters=KnowledgeFilters(content_type='manual')
    bm=retrieval.retrieve(query,'bm25',filters)
    assert embedder.calls==[] and store.calls[-1][1] is None
    assert len(bm.candidates)==50 and len(bm.ranked)==10
    for strategy in ('dense','hybrid','hybrid_rerank'):
        result=retrieval.retrieve(query,strategy,filters)
        assert result.query is query and len(result.candidates)==50 and len(result.ranked)==10
    assert len(embedder.calls)==3 and all(c==[query.search_text] for c in embedder.calls)
    assert all(call[0] is query and call[3] is filters for call in store.calls)
    question,chunks,limit=reranker.calls[0]
    assert question==query.standard_question and len(chunks)==50 and limit==10
    assert result.ranked[0].chunk.id==store.chunks[-1].id
    assert 'reranker' in result.stage_hits


@pytest.mark.parametrize('field',['text','source_digest','product_category','missing_mysql','changed_mysql','unknown_id'])
def test_stale_entity_is_service_error_not_normal_low_confidence(sessions,corpus,field):
    ingest,manifest,_=corpus;snapshot=ingest.ingest_corpus(sessions,manifest)
    store=Store(snapshot);first=store.chunks[0]
    if field in {'text','source_digest','product_category'}: store.entities[first.id][field]='changed'
    if field=='missing_mysql':
        with sessions.begin() as session: session.delete(session.get(KnowledgeChunk,first.id))
    if field=='changed_mysql':
        with sessions.begin() as session: session.get(KnowledgeChunk,first.id).answer='changed'
    if field=='unknown_id':
        from dataclasses import replace
        store.chunks[0]=replace(first,id=999)
        store.entities[999]={**store.entities[first.id],'id':999}
    with pytest.raises(api().IndexStaleError):
        api().RagRetrieval(sessions,Embedder(),store,Reranker(),snapshot).retrieve(PreparedQuery('原话','标准','标准'),'hybrid',KnowledgeFilters())


def test_missing_entity_and_reordered_hits_do_not_guess_sources(sessions,corpus):
    ingest,manifest,_=corpus;snapshot=ingest.ingest_corpus(sessions,manifest)
    store=Store(snapshot);store.chunks.reverse()
    retrieval=api().RagRetrieval(sessions,Embedder(),store,Reranker(),snapshot)
    query=PreparedQuery('原话','标准','标准')
    result=retrieval.retrieve(query,'dense',KnowledgeFilters())
    assert [c.chunk.id for c in result.candidates]==[c.id for c in store.chunks]
    del store.entities[store.chunks[0].id]
    with pytest.raises(api().IndexStaleError): retrieval.retrieve(query,'bm25',KnowledgeFilters())


def test_read_current_corpus_is_readonly_exact_chain_and_requires_independent_done(sessions,corpus):
    from app import ch04_ingest
    from app.ch04_db import KnowledgeIndexState
    ingest,manifest,directory=corpus;snapshot=ingest.ingest_corpus(sessions,manifest)
    assert hasattr(ch04_ingest,'read_current_corpus'),'Read-only current corpus port missing'
    assert ch04_ingest.read_current_corpus(sessions,manifest)==snapshot
    with pytest.raises(RuntimeError,match='incomplete|stale'): ch04_ingest.read_current_corpus(sessions,manifest,'knowledge_ch04')
    with sessions.begin() as session:
        for id_,digest in snapshot.payload_digests.items(): session.add(KnowledgeIndexState(collection_name='knowledge_ch04',chunk_id=id_,payload_digest=digest,status='done'))
    assert ch04_ingest.read_current_corpus(sessions,manifest,'knowledge_ch04')==snapshot
    with sessions.begin() as session:
        row=session.get(KnowledgeChunk,next(iter(snapshot.chunks)),options=[undefer('*')]);row.source_end_line=999
    with pytest.raises(RuntimeError,match='provenance'): ch04_ingest.read_current_corpus(sessions,manifest,'knowledge_ch04')
    with sessions() as session: assert len(list(session.scalars(select(KnowledgeChunk))))==len(snapshot.chunks)


def test_read_current_corpus_accepts_completed_new_version_with_retained_old_build_audit(sessions,corpus):
    from app import ch04_ingest,ch04_index
    from test_ch04_index import Embedder as BuildEmbedder,Store as BuildStore
    ingest,manifest,directory=corpus
    old=ingest.ingest_corpus(sessions,manifest);store=BuildStore()
    ch04_index.build_index(sessions,BuildEmbedder(),store,old)
    path=directory/'product-specs.md'
    path.write_text(path.read_text(encoding='utf-8')+'\n新版规范5kg。',encoding='utf-8')
    current=ingest.ingest_corpus(sessions,manifest)
    ch04_index.build_index(sessions,BuildEmbedder(),store,current)
    assert set(old.chunks)-set(current.chunks)
    assert ch04_ingest.read_current_corpus(sessions,manifest,store.collection_name)==current
