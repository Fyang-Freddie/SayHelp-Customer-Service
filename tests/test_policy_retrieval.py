import asyncio
from importlib import import_module
from types import SimpleNamespace
from app.knowledge_types import EvidenceChunk,RankedChunk,RetrievalResult
import pytest


def chunk(id,source,kind):
    return EvidenceChunk(id,'退货政策','未拆封商品的退货政策完整条款','政策',None,kind,'政策',source,1,2,'a'*64)

class Retrieval:
    def __init__(self):
        self.corpus=SimpleNamespace(chunks={1:chunk(1,'knowledge_db/returns-policy.md','policy'),2:chunk(2,'knowledge_db/after-sales-manual.md','manual'),3:chunk(3,'knowledge_db/product-specifications.md','manual')})
        self.requests=[]; self.reranks=[]; self.reranker=self
    def retrieve(self,query,strategy,filters=None,*,allowed_ids=None):
        self.requests.append((query.search_text,strategy,allowed_ids))
        hits=[RankedChunk(self.corpus.chunks[i],n,.8) for n,i in enumerate([1,2],1)]
        return RetrievalResult(strategy,query,hits,hits,{})
    def rerank(self,query,chunks,limit):
        self.reranks.append((query,[c.id for c in chunks],limit))
        return [RankedChunk(c,n,.9) for n,c in enumerate(chunks,1)]

def prepare(retrieval):
    try: cls=import_module('app.policy_retrieval').PolicyRetrieval
    except ModuleNotFoundError: pytest.fail('Policy retrieval missing')
    policy=SimpleNamespace(thresholds={'hybrid_rerank':.5},assess=lambda result:SimpleNamespace(sufficient=True,reason='passed'))
    return asyncio.run(cls(retrieval,policy,SimpleNamespace(context_token_budget=4096,response_token_reserve=512)).prepare('耳机能退吗',['耳机退货条件','耳机退款政策','耳机售后保障'],SimpleNamespace(issues=())))

def test_policy_only_recall_dedup_and_single_original_query_rerank():
    retrieval=Retrieval(); result=prepare(retrieval)
    assert len(retrieval.requests)==4
    assert all(strategy=='hybrid' and allowed==frozenset({1,2}) for _,strategy,allowed in retrieval.requests)
    assert retrieval.reranks==[('耳机能退吗',[1,2],10)]
    assert {c['chunk_id'] for c in result['citations']}=={'1','2'}
    assert result['knowledge_confidence']['sufficient'] is True

def test_retrieval_failure_exposes_no_evidence_or_eligibility():
    retrieval=Retrieval()
    def failed(*a,**kw): raise RuntimeError('backend issue')
    retrieval.retrieve=failed
    result=prepare(retrieval)
    assert result['evidence']==[] and not result['knowledge_confidence']['sufficient']
    assert result['stop_reason']=='retrieval_error'


def test_allowed_policy_ids_filter_both_native_search_legs():
    from tests.test_ch04_milvus import Spy,vector
    from app.hybrid_store import HybridMilvusStore
    from app.knowledge_types import PreparedQuery,KnowledgeFilters
    spy=Spy(); store=HybridMilvusStore('unused',client=spy)
    store.retrieve(PreparedQuery('退货','退货','退货'),vector(),'hybrid',KnowledgeFilters(),allowed_ids=frozenset({1,2}))
    call=next(kw for name,kw in spy.calls if name=='hybrid')
    assert all('id in {allowed_ids}' in req.expr and req.expr_params['allowed_ids']==[1,2] for req in call['reqs'])
    assert all(kw['filter_params']['allowed_ids']==[1,2] for name,kw in spy.calls if name=='search')


def test_empty_policy_id_set_never_means_unfiltered_search():
    from tests.test_ch04_milvus import Spy,vector
    from app.hybrid_store import HybridMilvusStore
    from app.knowledge_types import PreparedQuery,KnowledgeFilters
    spy=Spy(); store=HybridMilvusStore('unused',client=spy)
    result=store.retrieve(PreparedQuery('退货','退货','退货'),vector(),'hybrid',KnowledgeFilters(),allowed_ids=frozenset())
    assert not result.hits and not spy.calls
