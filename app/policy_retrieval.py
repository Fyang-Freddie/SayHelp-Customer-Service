"""Policy-source-only retrieval; fuse candidate ranks then rerank once for the original query."""
import asyncio,math
from dataclasses import asdict
from app.knowledge_types import PreparedQuery,RankedChunk,RetrievalResult
from app.evidence import select_prompt_evidence
from app.rag_retrieval import IndexStaleError
from app.config import POLICY_SEARCH_CONCURRENCY,POLICY_RRF_K,POLICY_CANDIDATES,POLICY_TOP_K

POLICY_SOURCES=frozenset({'knowledge_db/returns-policy.md','knowledge_db/after-sales-manual.md'})

class PolicyRetrieval:
    def __init__(self,retrieval,confidence_policy,settings):
        self.retrieval,self.policy,self.settings=retrieval,confidence_policy,settings

    async def prepare(self,query,queries,item):
        threshold=self.policy.thresholds.get('hybrid_rerank')
        def blocked(reason): return {'evidence':[],'citations':[],'knowledge_confidence':{'sufficient':False,'reason':reason,'score':None,'threshold':threshold},'stop_reason':reason}
        if item.issues: return blocked('order_data_inconsistent')
        if len(queries)!=3 or any(not isinstance(q,str) or not q.strip() for q in queries) or len(set(queries))!=3:
            return blocked('expansion_invalid_response')
        allowed=frozenset(id for id,c in self.retrieval.corpus.chunks.items() if c.source_file in POLICY_SOURCES)
        if not allowed: return blocked('missing_policy_sources')
        merged_queries=list(dict.fromkeys([query,*queries]))
        gate=asyncio.Semaphore(POLICY_SEARCH_CONCURRENCY)
        async def search(text):
            async with gate:
                return await asyncio.to_thread(self.retrieval.retrieve,PreparedQuery(text,text,text),'hybrid',allowed_ids=allowed)
        try:
            results=await asyncio.gather(*(search(text) for text in merged_queries))
            scores={}; chunks={}
            for result in results:
                seen=set()
                for hit in result.candidates:
                    if hit.chunk.id not in allowed or hit.chunk!=self.retrieval.corpus.chunks[hit.chunk.id]: raise ValueError('Unexpected policy evidence')
                    if hit.chunk.id in seen: continue
                    seen.add(hit.chunk.id)
                    chunks[hit.chunk.id]=hit.chunk
                    scores[hit.chunk.id]=scores.get(hit.chunk.id,0)+1/(POLICY_RRF_K+hit.rank)
            ids=sorted(scores,key=lambda id:(-scores[id],id))[:POLICY_CANDIDATES]
            candidates=[RankedChunk(chunks[id],rank,scores[id]) for rank,id in enumerate(ids,1)]
            ranked=await asyncio.to_thread(self.retrieval.reranker.rerank,query,[chunks[id] for id in ids],limit=POLICY_TOP_K) if ids else []
            if len({hit.chunk.id for hit in ranked})!=len(ranked) or any(hit.chunk.id not in ids or not math.isfinite(hit.score) for hit in ranked): raise ValueError('Invalid reranker output')
            prepared=PreparedQuery(query,query,query)
            result=RetrievalResult('hybrid_rerank',prepared,candidates,ranked,{})
            decision=self.policy.assess(result)
            if not decision.sufficient: return blocked('weak_policy_evidence')
            selected=select_prompt_evidence(prepared,result,self.settings)
            if not selected.citations: return blocked('weak_policy_evidence')
            by_id={str(hit.chunk.id):hit for hit in ranked}
            evidence=[{**asdict(by_id[c['chunk_id']].chunk),'n':c['n'],'rank':by_id[c['chunk_id']].rank,'score':by_id[c['chunk_id']].score} for c in selected.citations]
            return {'evidence':evidence,'citations':selected.citations,'knowledge_confidence':{'sufficient':True,'reason':decision.reason,'score':ranked[0].score,'threshold':threshold},'stop_reason':None}
        except Exception as error:
            return blocked('index_stale' if isinstance(error,IndexStaleError) else 'retrieval_error')
