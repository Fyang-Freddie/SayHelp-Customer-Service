"""Warmed uncached comparison on 12 authored calibration cases; no remote LLM calls."""
import os,sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('eval/ch04/latency_local_comparison.json'))
args=parser.parse_args()
from app.config import Settings
from app.db import make_session_factory
from app.ch04_ingest import ROOT,read_current_corpus
from app.embedding import BgeM3Embedder
from app.reranking import BgeReranker
from app.hybrid_store import HybridMilvusStore
from app.rag_retrieval import RagRetrieval
from app.knowledge_types import PreparedQuery,KnowledgeFilters
settings=Settings.from_env();sf=make_session_factory(settings.database_url)
store=HybridMilvusStore(settings.milvus_uri,collection_name=settings.knowledge_collection)
corpus=read_current_corpus(sf,ROOT/'eval/ch04/corpus.json',settings.knowledge_collection)
embed=BgeM3Embedder(settings.bge_cache_dir);rerank=BgeReranker(settings.bge_cache_dir)
retrieval=RagRetrieval(sf,embed,store,rerank,corpus)
artifact=json.loads(Path(settings.knowledge_confidence_path).read_text(encoding='utf-8'))
from app.evaluation.calibration import ConfidencePolicy,fingerprints
ConfidencePolicy.load(settings.knowledge_confidence_path,fingerprints(corpus,settings,ROOT/'eval/ch04/calibration.json'),settings.knowledge_confidence_overrides)
from app.evaluation.dataset import load_cases,bind_gold
bound={b.case.eval_id:b for b in bind_gold(load_cases(ROOT/'eval/ch04/calibration.json'),corpus,sf)}
rows=artifact['strategies']['hybrid_rerank']['samples']
filters={k:b.case.filters for k,b in bound.items()}
# Warmup excluded; all measured requests run both real models and real Milvus.
measured=[]
try:
 retrieval.retrieve(PreparedQuery(**rows[0]['prepared_query']),'hybrid_rerank',filters[rows[0]['eval_id']])
 for i,row in enumerate(rows):
  query=PreparedQuery(**row['prepared_query'])
  for mode in (['before','after'] if i%2==0 else ['after','before']):
   rerank.batch_size=8 if mode=='before' else 4;retrieval.diagnostics=mode=='before'
   result=retrieval.retrieve(query,'hybrid_rerank',filters[row['eval_id']])
   ranked=[r.chunk.id for r in result.ranked];gold=set(bound[row['eval_id']].gold_ids)
   entry={'eval_id':row['eval_id'],'answerable':row['answerable'],'mode':mode,'candidate_ids':[r.chunk.id for r in result.candidates],'ranked_ids':ranked,'scores':[r.score for r in result.ranked],'timings_ms':result.timings_ms,'recall_at_10':len(gold&set(ranked))/len(gold) if gold else None,'mrr':next((1/(j+1) for j,k in enumerate(ranked) if k in gold),0) if gold else None}
   measured.append(entry)
   args.output.parent.mkdir(parents=True,exist_ok=True)
   args.output.write_text(json.dumps({'method':'12 authored prepared queries, warmed models, alternating before/after order; no answer or inference cache, no remote requests','rows':measured},ensure_ascii=False,indent=2),encoding='utf-8')
   print(row['eval_id'],mode,round(result.timings_ms['retrieval_total']),flush=True)
finally:store.close();sf.kw['bind'].dispose()
