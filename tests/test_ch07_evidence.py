import json
from dataclasses import asdict
from types import SimpleNamespace
from langchain_core.messages import HumanMessage
from app.context_budget import estimate_tokens
from app.evidence import select_prompt_evidence
from test_rag_evidence import result


def test_evidence_caps_whole_serialized_chunks_and_top_k():
    query,retrieved=result(count=10,answer='完整原文'*100)
    settings=SimpleNamespace(layered_context=True,rerank_top_k=3,context_token_budget=18000,response_token_reserve=2000)
    selected=select_prompt_evidence(query,retrieved,settings)
    assert 0<len(selected.citations)<=3
    by_id={str(item.chunk.id):item for item in retrieved.ranked}
    payload=[{**asdict(by_id[c['chunk_id']].chunk),'n':c['n'],
              'rank':by_id[c['chunk_id']].rank,'score':by_id[c['chunk_id']].score} for c in selected.citations]
    assert estimate_tokens([HumanMessage(json.dumps({'本轮检索证据':payload},ensure_ascii=False))])<=3*512
    assert all(c['answer']=='完整原文'*100 for c in selected.citations)
