"""Fixed-model reranking checks plus an actual cached-model smoke."""
import importlib
import math
from concurrent.futures import ThreadPoolExecutor

import pytest
from app.ch04_ingest import index_text,load_documents,digest_text,ROOT
from app.knowledge_chunking import chunk_markdown
from app.knowledge_types import EvidenceChunk


def api(): return importlib.import_module('app.reranking')


def chunk(id_):
    return EvidenceChunk(id_,'测试候选'+str(id_),'测试边界文本','测试',None,'manual','测试 / 边界','knowledge_db/product-specs.md',1,2,'a'*64)


class Tokenizer:
    model_max_length=512
    def __call__(self,queries,texts,**kwargs):
        assert kwargs['truncation'] is False and kwargs['add_special_tokens'] is True
        return {'input_ids':[list(range(len(q)+len(t)+3)) for q,t in zip(queries,texts)]}


class Model:
    max_seq_length=512
    tokenizer=Tokenizer()
    def __init__(self,scores=None,error=None): self.scores=scores;self.error=error;self.calls=[]
    def predict(self,pairs,**kwargs):
        self.calls.append((pairs,kwargs))
        if self.error: raise self.error
        return self.scores if self.scores is not None else [float(i//2) for i in range(len(pairs))]


def test_all_fifty_candidates_are_scored_then_top_ten_stably_ranked():
    import torch
    candidates=[chunk(i+1) for i in range(50)]
    model=Model();reranker=api().BgeReranker(model=model)
    ranked=reranker.rerank('原问题',candidates)
    assert len(ranked)==10 and [r.chunk.id for r in ranked[:4]]==[49,50,47,48]
    assert [r.rank for r in ranked]==list(range(1,11))
    pairs,kwargs=model.calls[0]
    assert pairs==[('原问题',index_text(c)) for c in candidates]
    assert kwargs['batch_size']==8 and isinstance(kwargs['activation_fn'],torch.nn.Identity)
    assert kwargs['convert_to_numpy'] is True


@pytest.mark.parametrize('scores',[[float('nan')],[float('inf')],[],[1,2]])
def test_invalid_or_wrong_count_model_scores_fail_without_fallback(scores):
    with pytest.raises(ValueError): api().BgeReranker(model=Model(scores=scores)).rerank('原问题',[chunk(1)])


def test_full_pair_is_checked_before_model_truncation():
    model=Model()
    with pytest.raises(api().RerankInputTooLongError) as error:
        api().BgeReranker(model=model).rerank('长'*600,[chunk(7)])
    assert '7' in str(error.value) and 'product-specs.md' in str(error.value)
    assert '长'*50 not in str(error.value) and model.calls==[]


def test_model_failure_is_safe_and_does_not_fallback():
    with pytest.raises(RuntimeError) as error:
        api().BgeReranker(model=Model(error=RuntimeError('private-provider-error'))).rerank('问题',[chunk(1)])
    assert 'private-provider-error' not in str(error.value)


def test_lazy_single_initialization_and_serialized_inference(monkeypatch):
    import sentence_transformers
    from threading import Lock
    import time
    created=[]
    class OverlapProbe(Model):
        def __init__(self):
            super().__init__(); self.active=0; self.maximum_active=0; self.counter_lock=Lock()
        def predict(self,*args,**kwargs):
            with self.counter_lock:
                self.active+=1; self.maximum_active=max(self.maximum_active,self.active)
            try:
                time.sleep(.01)  # Deliberate overlap window exposes concurrent inference.
                return super().predict(*args,**kwargs)
            finally:
                with self.counter_lock: self.active-=1
    def create(name,**kwargs):
        assert name=='BAAI/bge-reranker-v2-m3'
        assert kwargs['revision']=='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
        model=OverlapProbe();created.append(model);return model
    monkeypatch.setattr(sentence_transformers,'CrossEncoder',create)
    reranker=api().BgeReranker()
    assert created==[] and reranker.rerank('问题',[])==[] and created==[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results=list(executor.map(lambda _:reranker.rerank('问题',[chunk(1),chunk(2)]),range(8)))
    assert len(created)==1 and len(created[0].calls)==8 and all(len(r)==2 for r in results)
    assert created[0].maximum_active==1


def test_real_fixed_model_returns_ten_and_exact_model_over_nearby_variants():
    candidates=[]
    for name,kind,body,digest in load_documents(ROOT/'eval/ch04/corpus.json'):
        for draft in chunk_markdown(body,content_type=kind):
            candidates.append(EvidenceChunk(len(candidates)+1,draft.questions,draft.answer,draft.category,None,kind,draft.section_path,name,draft.source_start_line,draft.source_end_line,digest))
    ranked=api().BgeReranker().rerank('MH-LP50是否有App远程监控和健康记录？',candidates)
    assert len(ranked)==10 and all(math.isfinite(r.score) for r in ranked)
    assert 'MH-LP50' in ranked[0].chunk.question
    assert '不含 App' in ranked[0].chunk.answer
    assert api().MODEL_ID=='BAAI/bge-reranker-v2-m3'
