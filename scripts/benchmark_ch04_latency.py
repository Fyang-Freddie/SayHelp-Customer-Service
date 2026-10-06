"""Read-only, real-service latency measurement; never writes chat/pool records.

Default replays authored prepared queries locally. --include-generation sends
four authored cases and their evidence to the configured model provider.
Run from repository root: python scripts/benchmark_ch04_latency.py --help
"""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.config import Settings
from app.db import make_session_factory
from app.ch04_ingest import ROOT,read_current_corpus
from app.embedding import BgeM3Embedder
from app.reranking import BgeReranker
from app.hybrid_store import HybridMilvusStore
from app.rag_retrieval import RagRetrieval
from app.knowledge_types import PreparedQuery
from app.evaluation.dataset import load_cases,bind_gold
from app.evaluation.calibration import ConfidencePolicy,fingerprints
from app.evidence import select_prompt_evidence


async def measure(args):
    settings=Settings.from_env()
    sf=make_session_factory(settings.database_url)
    store=HybridMilvusStore(settings.milvus_uri,collection_name=settings.knowledge_collection)
    try:
        corpus=read_current_corpus(sf,ROOT/'eval/ch04/corpus.json',settings.knowledge_collection)
        bound=bind_gold(load_cases(ROOT/'eval/ch04/calibration.json'),corpus,sf)
        selected=[b for b in bound if b.case.eval_id in {'K01','K03','K07','K08'}]
        artifact=json.loads(Path(settings.knowledge_confidence_path).read_text(encoding='utf-8'))
        prepared={r['eval_id']:r['prepared_query'] for r in artifact['strategies']['hybrid_rerank']['samples']}
        policy=ConfidencePolicy.load(settings.knowledge_confidence_path,fingerprints(corpus,settings,ROOT/'eval/ch04/calibration.json'),settings.knowledge_confidence_overrides)
        retrieval=RagRetrieval(sf,BgeM3Embedder(settings.bge_cache_dir),store,BgeReranker(settings.bge_cache_dir,batch_size=args.batch_size),corpus,diagnostics=args.diagnostics)
        if args.include_generation:
            from app.model_service import ModelService
            from app.query_understanding import QueryUnderstanding
            from app.rag_generation import RagGenerator
            model=ModelService(settings);understand=QueryUnderstanding(model);generator=RagGenerator(model)
        rows=[]
        for repeat in range(args.repeats):
            for item in selected:
                start=perf_counter();row={'eval_id':item.case.eval_id,'answerable':item.case.answerable,'repeat':repeat,'cold':not rows}
                try:
                    query=await understand.prepare(item.case.query) if args.include_generation else PreparedQuery(**prepared[item.case.eval_id])
                    row['understanding_ms']=(perf_counter()-start)*1000
                    row['prepared_query']=asdict(query)
                    if query.intent!='knowledge':
                        row.update(outcome=query.intent,total_ms=(perf_counter()-start)*1000);rows.append(row);continue
                    result=await asyncio.to_thread(retrieval.retrieve,query,'hybrid_rerank',item.case.filters)
                    prompt=select_prompt_evidence(query,result,settings)
                    row.update(timings_ms=result.timings_ms,candidates=len(result.candidates),ranked_ids=[r.chunk.id for r in result.ranked],scores=[r.score for r in result.ranked],confidence=asdict(policy.assess(result)))
                    row['recall_at_10']=len(set(item.gold_ids)&{r.chunk.id for r in result.ranked})/len(item.gold_ids) if item.gold_ids else None
                    row['mrr']=next((1/r.rank for r in result.ranked if r.chunk.id in item.gold_ids),0) if item.gold_ids else None
                    if args.include_generation:
                        before=perf_counter();answer=await generator.generate(query,prompt,policy.assess(result))
                        row.update(generation_ms=(perf_counter()-before)*1000,useful=answer.useful,answer=answer.answer)
                    row.update(outcome='complete',total_ms=(perf_counter()-start)*1000)
                except Exception as error:
                    row.update(outcome='error',error_type=type(error).__name__,total_ms=(perf_counter()-start)*1000)
                rows.append(row)
                print(json.dumps({k:row[k] for k in ('eval_id','cold','outcome','total_ms')},ensure_ascii=False),flush=True)
                args.output.parent.mkdir(parents=True,exist_ok=True)
                report={'mode':'live_model' if args.include_generation else 'local_prepared_replay','rows':rows,'batch_size':args.batch_size,'diagnostics':args.diagnostics,'limitations':['Read-only pipeline timing excludes HTTP, browser and persistence.','First case includes local model loading; other cases reuse model objects.','Four authored calibration cases are a diagnostic sample, not a full quality evaluation.'],'fingerprints':fingerprints(corpus,settings,ROOT/'eval/ch04/calibration.json')}
                args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        return rows
    finally:
        store.close();sf.kw['bind'].dispose()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-generation',action='store_true')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.repeats<=10:parser.error('repeats must be within 1..10')
    rows=asyncio.run(measure(args))
    if any(r['outcome']=='error' for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
