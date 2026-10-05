"""Prompt selection keeps complete evidence, then arranges/cites final positions."""
import importlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages.utils import count_tokens_approximately
from app.knowledge_types import EvidenceChunk,RankedChunk,RetrievalResult,PreparedQuery


def api(): return importlib.import_module('app.evidence')


def result(count=10,answer='原文依据'):
    query=PreparedQuery('用户原话','标准问法','检索问法',synonyms=['仅检索扩展'])
    ranked=[RankedChunk(EvidenceChunk(i,'章节问题'+str(i),answer,'商品','猫砂盆','manual','商品规格 / 章节'+str(i),'knowledge_db/product-specs.md',i,i+1,'a'*64),i,10-i) for i in range(1,count+1)]
    return query,RetrievalResult('hybrid_rerank',query,ranked,ranked,{})


def settings(budget=20000): return SimpleNamespace(context_token_budget=budget,response_token_reserve=512)


def test_ten_relevance_ranks_put_best_at_front_and_second_at_end_with_final_numbers():
    query,retrieved=result()
    evidence=api().select_prompt_evidence(query,retrieved,settings())
    assert evidence.relevance_order==[1,3,5,7,9,10,8,6,4,2]
    assert [c['n'] for c in evidence.citations]==list(range(1,11))
    assert [int(c['chunk_id']) for c in evidence.citations]==evidence.relevance_order
    assert evidence.citations[0]['section_path']=='商品规格 / 章节1'
    assert evidence.citations[0]['source_start_line']==1 and evidence.citations[0]['source_url']=='/v1/knowledge/documents/product-specs.md'
    body='\n'.join(m.content for m in evidence.messages)
    assert '用户原话' in body and '标准问法' in body and '仅检索扩展' not in body
    assert evidence.prompt_tokens==count_tokens_approximately(evidence.messages,chars_per_token=1.0)


def test_budget_removes_low_relevance_whole_chunks_then_renumbers():
    query,retrieved=result(answer='原文证据。'*100)
    full=api().select_prompt_evidence(query,retrieved,settings())
    evidence=api().select_prompt_evidence(query,retrieved,settings(full.prompt_tokens//2+512))
    assert 0<len(evidence.citations)<10
    assert {int(c['chunk_id']) for c in evidence.citations}==set(range(1,len(evidence.citations)+1))
    assert all(c['answer']=='原文证据。'*100 for c in evidence.citations)
    assert [c['n'] for c in evidence.citations]==list(range(1,len(evidence.citations)+1))
    assert evidence.prompt_tokens+512<=full.prompt_tokens//2+512


def test_no_complete_chunk_can_fit_yields_empty_evidence():
    query,retrieved=result(count=1,answer='长'*5000)
    evidence=api().select_prompt_evidence(query,retrieved,settings(2000))
    assert evidence.citations==[] and evidence.relevance_order==[]


def test_confidence_calibration_uses_actual_score_scale_and_conservative_ties():
    cal=importlib.import_module('app.evaluation.calibration')
    chosen=cal.choose_threshold([(0.031,False),(0.032,True),(0.033,True)])
    assert .031<chosen['threshold']<=.032 and chosen['false_accept']==0 and chosen['false_reject']==0
    tied=cal.choose_threshold([(0.032,False),(0.032,True)])
    assert tied['threshold']>.032 and tied['false_accept']==0 and tied['false_reject']==1


def test_confidence_missing_or_mismatched_artifact_cannot_default_to_dense_point55(tmp_path):
    cal=importlib.import_module('app.evaluation.calibration')
    with pytest.raises(cal.ConfidenceConfigurationError): cal.ConfidencePolicy.load(tmp_path/'missing.json',{'corpus_digest':'expected'})
    artifact={'version':1,'fingerprints':{'corpus_digest':'wrong'},'strategies':{name:{'threshold':.031} for name in ('dense','bm25','hybrid','hybrid_rerank')}}
    path=tmp_path/'confidence.json';path.write_text(json.dumps(artifact),encoding='utf-8')
    with pytest.raises(cal.ConfidenceConfigurationError): cal.ConfidencePolicy.load(path,{'corpus_digest':'expected'})


@pytest.mark.parametrize('override',['{"bm25": 8.1}', '{"bm25": null}', '{"dense": true}', '{"unknown": 0.1}'])
def test_explicit_confidence_overrides_are_validated_and_never_legacy_default(monkeypatch,override):
    from app.config import Settings
    for name,value in {'CHAT_BASE_URL':'https://example.invalid','CHAT_MODEL':'test','CHAT_API_KEY':'placeholder','DATABASE_URL':'mysql+pymysql://test','KNOWLEDGE_CONFIDENCE_OVERRIDES':override}.items(): monkeypatch.setenv(name,value)
    if override=='{"bm25": 8.1}':
        settings=Settings.from_env()
        assert getattr(settings,'knowledge_confidence_overrides',None)=={'bm25':8.1}
        assert settings.knowledge_confidence_path=='eval/ch04/confidence.json'
    else:
        with pytest.raises(ValueError,match='CONFIDENCE_OVERRIDES'): Settings.from_env()


def test_empty_and_score_threshold_use_strategy_scale_with_explicit_override(tmp_path):
    from app.evaluation.calibration import ConfidencePolicy
    query,retrieved=result(1)
    artifact={'version':1,'fingerprints':{'corpus_digest':'expected'},'strategies':{name:{'threshold':10.0} for name in ('dense','bm25','hybrid','hybrid_rerank')}}
    path=tmp_path/'confidence.json';path.write_text(json.dumps(artifact),encoding='utf-8')
    policy=ConfidencePolicy.load(path,{'corpus_digest':'expected'})
    assert not policy.assess(retrieved).sufficient
    policy=ConfidencePolicy.load(path,{'corpus_digest':'expected'},{'hybrid_rerank':8.0})
    assert policy.assess(retrieved).sufficient and policy.overrides=={'hybrid_rerank':8.0}
    assert not policy.assess(result(0)[1]).sufficient
