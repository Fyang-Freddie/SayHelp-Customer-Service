"""Calibrate only the frozen 12 authored questions against real services."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import platform
from importlib.metadata import version
from app.config import Settings
from app.db import make_session_factory
from app.ch04_ingest import ROOT,read_current_corpus
from app.embedding import BgeM3Embedder
from app.reranking import BgeReranker
from app.hybrid_store import HybridMilvusStore
from app.model_service import ModelService
from app.query_understanding import QueryUnderstanding
from app.rag_retrieval import RagRetrieval
from app.evidence import select_prompt_evidence
from app.evaluation.dataset import load_cases,bind_gold
from app.evaluation.calibration import STRATEGIES,choose_threshold,fingerprints


async def calibrate(settings,dataset):
    cases=load_cases(dataset)
    if len(cases)!=12 or any(c._split!='calibration' for c in cases): raise ValueError('Only independent calibration12 may choose thresholds')
    sf=make_session_factory(settings.database_url)
    store=HybridMilvusStore(settings.milvus_uri,collection_name=settings.knowledge_collection)
    try:
        corpus=read_current_corpus(sf,ROOT/'eval/ch04/corpus.json',settings.knowledge_collection)
        bound=bind_gold(cases,corpus,sf)
        retrieval=RagRetrieval(sf,BgeM3Embedder(settings.bge_cache_dir),store,BgeReranker(settings.bge_cache_dir),corpus)
        understand=QueryUnderstanding(ModelService(settings))
        samples={s:[] for s in STRATEGIES}
        for item in bound:
            query=await understand.prepare(item.case.query)
            for strategy in STRATEGIES:
                result=await asyncio.to_thread(retrieval.retrieve,query,strategy,item.case.filters)
                prompt=select_prompt_evidence(query,result,settings)
                present={int(c['chunk_id']) for c in prompt.citations}
                score=result.ranked[0].score if result.ranked else None
                positive=item.case.answerable and query.intent=='knowledge' and set(item.gold_ids)<=present
                samples[strategy].append({'eval_id':item.case.eval_id,'query':item.case.query,'prepared_query':asdict(query),
                    'score':score,'answerable':item.case.answerable,'gold_ids':item.gold_ids,'prompt_ids':sorted(present),
                    'positive':positive,'prompt_tokens':prompt.prompt_tokens,'relevance_order':prompt.relevance_order,
                    'stage_hits':{k:[asdict(h) for h in v] for k,v in result.stage_hits.items()},'timings_ms':result.timings_ms})
                print(f'calibration {item.case.eval_id}/{strategy} complete',flush=True)
        strategies={}
        for strategy,rows in samples.items():
            observed=[(r['score'],r['positive']) for r in rows if r['score'] is not None]
            chosen=choose_threshold(observed)
            chosen.update(samples=rows,positive_count=sum(r['positive'] for r in rows),negative_count=sum(not r['positive'] for r in rows),
                empty_count=sum(r['score'] is None for r in rows))
            chosen['false_reject']+=sum(r['positive'] and r['score'] is None for r in rows)
            strategies[strategy]=chosen
        return {'version':1,'fingerprints':fingerprints(corpus,settings,dataset),'strategies':strategies,
            'explicit_overrides':settings.knowledge_confidence_overrides,
            'software':{k:version(k) for k in ('pymilvus','sentence-transformers','langchain-core')},'python':platform.python_version(),
            'limits':['12 authored calibration questions; error rates are in-sample and not probability estimates.',
            'Positive means answerable AND all gold evidence survives whole-chunk budget; missing evidence is negative.',
            'Conservative approximate Chinese token budget: 1 character per token plus message overhead; not vendor tokenizer.',
            'RRF rank fusion scores may tie across known/unknown questions; zero false-accept priority can reject many known questions.',
            'Native hybrid timing includes separately labelled dense/BM25 diagnostic calls.']}
    finally:
        store.close();sf.kw['bind'].dispose()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'eval/ch04/calibration.json')
    parser.add_argument('--output',type=Path,default=ROOT/'eval/ch04/confidence.json')
    args=parser.parse_args(argv)
    import torch
    torch.set_num_threads(8)
    artifact=asyncio.run(calibrate(Settings.from_env(),args.dataset))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(artifact,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report=['# ch04 confidence calibration','',*artifact['limits'],'','| strategy | threshold | positives | negatives | false accept | false reject |','|---|---:|---:|---:|---:|---:|']
    for name,data in artifact['strategies'].items():
        report.append(f"| {name} | {data['threshold']:.8g} | {data['positive_count']} | {data['negative_count']} | {data['false_accept']} | {data['false_reject']} |")
    args.output.with_suffix('.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('Confidence artifact saved: '+str(args.output))

if __name__=='__main__': main()
