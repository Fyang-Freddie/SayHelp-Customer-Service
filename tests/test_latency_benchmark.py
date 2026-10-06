"""Every benchmark outcome must be persisted, including clarification."""
import asyncio
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from app.knowledge_types import PreparedQuery


def test_all_nonknowledge_cases_are_saved_without_running_inference(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('latency_benchmark',Path(__file__).resolve().parents[1]/'scripts/benchmark_ch04_latency.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    artifact=tmp_path/'confidence.json'
    artifact.write_text(json.dumps({'strategies':{'hybrid_rerank':{'samples':[
        {'eval_id':id_,'prepared_query':asdict(PreparedQuery('raw','standard','search',intent='clarify',clarification='provide model'))}
        for id_ in ('K01','K03')]}}}),encoding='utf-8')
    settings=SimpleNamespace(database_url='unused',milvus_uri='unused',knowledge_collection='test',knowledge_confidence_path=str(artifact),knowledge_confidence_overrides={},bge_cache_dir=None)
    monkeypatch.setattr(module.Settings,'from_env',lambda:settings)
    closed=[]
    factory=SimpleNamespace(kw={'bind':SimpleNamespace(dispose=lambda:closed.append('db'))})
    monkeypatch.setattr(module,'make_session_factory',lambda _:factory)
    monkeypatch.setattr(module,'HybridMilvusStore',lambda *a,**k:SimpleNamespace(close=lambda:closed.append('store')))
    monkeypatch.setattr(module,'read_current_corpus',lambda *a:object())
    monkeypatch.setattr(module,'load_cases',lambda *a:[])
    bound=[SimpleNamespace(case=SimpleNamespace(eval_id=id_,answerable=False,query='raw')) for id_ in ('K01','K03')]
    monkeypatch.setattr(module,'bind_gold',lambda *a:bound)
    monkeypatch.setattr(module,'fingerprints',lambda *a:{})
    monkeypatch.setattr(module.ConfidencePolicy,'load',lambda *a:object())
    def must_not_run(*a):raise AssertionError('clarification cannot invoke inference')
    monkeypatch.setattr(module,'RagRetrieval',lambda *a,**k:SimpleNamespace(retrieve=must_not_run))
    args=SimpleNamespace(include_generation=False,batch_size=4,diagnostics=False,repeats=1,output=tmp_path/'report.json')
    rows=asyncio.run(module.measure(args))
    assert args.output.exists(),'Clarification outcomes must produce a report'
    saved=json.loads(args.output.read_text(encoding='utf-8'))
    assert [r['eval_id'] for r in saved['rows']]==['K01','K03']
    assert [r['outcome'] for r in saved['rows']]==['clarify','clarify']
    assert saved['rows']==rows and closed==['store','db']
