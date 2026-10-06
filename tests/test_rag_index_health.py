"""A lost index is an infrastructure failure, not normal missing knowledge."""
import sys
from pathlib import Path
import pytest
from app.ch04_ingest import index_text,index_metadata
from app.knowledge_types import PreparedQuery,KnowledgeFilters,StoreResult
from app.rag_retrieval import RagRetrieval,IndexStaleError
from test_db import database_url,sessions
from test_ch04_ingest import corpus
from test_rag_retrieval import Store,Embedder,Reranker


@pytest.mark.parametrize('mode',['empty','missing','unexpected','off_candidate_metadata'])
def test_formal_server_index_must_match_current_corpus_even_when_hits_look_valid(sessions,corpus,mode):
    ingest,manifest,_=corpus;snapshot=ingest.ingest_corpus(sessions,manifest)
    class Server(Store):
        def __init__(self):
            super().__init__(snapshot)
            self.server={c.id:{'id':c.id,'text':index_text(c),**index_metadata(c)} for c in snapshot.chunks.values()}
            self.checked=False
        def read_entities(self,ids=None): self.checked=True;return self.server
        def retrieve(self,*args):
            if mode=='empty': return StoreResult([],{}, {})
            return super().retrieve(*args)
    store=Server()
    if mode=='empty': store.server={}
    if mode=='missing':
        first=store.chunks.pop(0);del store.server[first.id]
    if mode=='unexpected': store.server[999]={'id':999,'text':'stale-index-only-row'}
    if mode=='off_candidate_metadata':
        store.chunks=store.chunks[:2];last=list(snapshot.chunks)[-1]
        assert last not in {c.id for c in store.chunks}
        store.server[last]['product_category']='invalid-stale-category'
    with pytest.raises(IndexStaleError):
        RagRetrieval(sessions,Embedder(),store,Reranker(),snapshot).retrieve(PreparedQuery('原话','标准','标准'),'bm25',KnowledgeFilters())
    assert store.checked
