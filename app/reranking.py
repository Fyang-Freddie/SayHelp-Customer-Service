"""Fixed BGE v2-m3 reranking, raw logits and no silent pair truncation."""
import math
from threading import Lock
from app.ch04_ingest import index_text
from app.knowledge_types import RankedChunk

MODEL_ID='BAAI/bge-reranker-v2-m3'
MODEL_REVISION='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'


class RerankInputTooLongError(ValueError):
    def __init__(self,chunk,count,limit):
        super().__init__(f'Rerank chunk {chunk.id} ({chunk.source_file}, {chunk.section_path}): full pair has {count} tokens, limit {limit}; text was not truncated')


class BgeReranker:
    def __init__(self,cache_dir=None,*,model=None):
        self.cache_dir=cache_dir
        self._model=model
        self._lock=Lock()

    def rerank(self,query,chunks,limit=10):
        if type(limit) is not int or not 1<=limit<=10: raise ValueError('Rerank limit must be within [1,10]')
        if not isinstance(query,str) or not query.strip(): raise ValueError('Rerank question must be nonempty')
        if len(chunks)>50 or len({c.id for c in chunks})!=len(chunks): raise ValueError('Rerank requires at most 50 unique candidates')
        if not chunks: return []
        pairs=[(query,index_text(chunk)) for chunk in chunks]
        with self._lock:
            import torch
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder
                    self._model=CrossEncoder(MODEL_ID,revision=MODEL_REVISION,cache_folder=self.cache_dir,activation_fn=torch.nn.Identity())
                except Exception:
                    raise RuntimeError('指定重排模型暂时不可用，请检查本地缓存。') from None
            model=self._model
            bounds=[getattr(model,'max_seq_length',None),getattr(model.tokenizer,'model_max_length',None)]
            bounds=[value for value in bounds if type(value) is int and 0<value<1000000]
            if not bounds: raise RuntimeError('Rerank tokenizer has no trustworthy sequence limit')
            maximum=min(bounds)
            try:
                tokens=model.tokenizer([q for q,_ in pairs],[text for _,text in pairs],truncation=False,add_special_tokens=True,padding=False)['input_ids']
                if len(tokens)!=len(chunks): raise ValueError('Wrong tokenization count')
            except Exception:
                raise RuntimeError('重排输入检查暂时不可用。') from None
            for chunk,ids in zip(chunks,tokens,strict=True):
                if len(ids)>maximum: raise RerankInputTooLongError(chunk,len(ids),maximum)
            try:
                scores=model.predict(pairs,batch_size=8,activation_fn=torch.nn.Identity(),convert_to_numpy=True,show_progress_bar=False)
            except Exception:
                raise RuntimeError('重排推理暂时不可用，请稍后重试。') from None
            if len(scores)!=len(chunks): raise ValueError('Rerank score count differs from candidate count')
            try: scores=[float(score) for score in scores]
            except (TypeError,ValueError): raise ValueError('Rerank scores must be scalars') from None
            if not all(math.isfinite(score) for score in scores): raise ValueError('Rerank scores must be finite')
        order=sorted(range(len(chunks)),key=lambda i:(-scores[i],i))[:limit]
        return [RankedChunk(chunks[i],rank,scores[i]) for rank,i in enumerate(order,1)]
