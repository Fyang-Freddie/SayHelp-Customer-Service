"""Observed per-strategy score thresholds, bound to immutable run inputs."""
import hashlib
import json
import math
from pathlib import Path
from app.knowledge_types import ConfidenceDecision

STRATEGIES=('dense','bm25','hybrid','hybrid_rerank')
class ConfidenceConfigurationError(RuntimeError): pass


def choose_threshold(samples):
    if not samples or any(not math.isfinite(s) or type(label) is not bool for s,label in samples):
        raise ValueError('Calibration needs finite observed scores and boolean labels')
    scores=sorted({s for s,_ in samples})
    candidates=set(scores)|{math.nextafter(scores[0],-math.inf),math.nextafter(scores[-1],math.inf)}
    candidates.update(a+(b-a)/2 for a,b in zip(scores,scores[1:]))
    choices=[]
    for threshold in candidates:
        fa=sum(score>=threshold and not label for score,label in samples)
        fn=sum(score<threshold and label for score,label in samples)
        choices.append(dict(threshold=threshold,false_accept=fa,false_reject=fn))
    return min(choices,key=lambda x:(x['false_accept'],x['false_reject'],-x['threshold']))


def fingerprints(corpus,settings,dataset):
    from app.embedding import MODEL_ID
    from app.reranking import MODEL_ID as RERANK,MODEL_REVISION
    from app.rag_prompts import SYSTEM_PROMPT
    return {'corpus_digest':corpus.corpus_digest,'calibration_digest':hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        'embedding_model':MODEL_ID,'reranker_model':RERANK,'reranker_revision':MODEL_REVISION,'query_model':settings.chat_model,
        'prompt_digest':hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        'context_token_budget':settings.context_token_budget,'response_token_reserve':settings.response_token_reserve,
        'token_counter':'langchain approximate chars_per_token=1.0','dense_k':50,'bm25_k':50,'rrf_k':60,'generation_k':10}

class ConfidencePolicy:
    def __init__(self,thresholds,fingerprints,overrides=None):
        self.thresholds=thresholds;self.fingerprints=fingerprints;self.overrides=overrides or {}
    @classmethod
    def load(cls,path,expected_fingerprints,overrides=None):
        try:
            data=json.loads(Path(path).read_text(encoding='utf-8'))
            if data['version']!=1 or data['fingerprints']!=expected_fingerprints or set(data['strategies'])!=set(STRATEGIES): raise ValueError()
            thresholds={name:data['strategies'][name]['threshold'] for name in STRATEGIES}
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in thresholds.values()): raise ValueError()
            if overrides:
                if not isinstance(overrides,dict) or not set(overrides)<=set(STRATEGIES) or any(type(v) not in (int,float) or not math.isfinite(v) for v in overrides.values()): raise ValueError()
                thresholds.update(overrides)
            return cls(thresholds,data['fingerprints'],overrides)
        except (OSError,ValueError,KeyError,TypeError):
            raise ConfidenceConfigurationError('Confidence artifact missing or incompatible; recalibrate explicitly') from None
    def assess(self,result):
        if not result.ranked: return ConfidenceDecision(False,'没有召回完整证据')
        score=result.ranked[0].score;threshold=self.thresholds[result.strategy]
        ok=score>=threshold
        return ConfidenceDecision(ok,f'{result.strategy} top1={score:.8g}, threshold={threshold:.8g}; '+('通过校准门槛' if ok else '检索证据置信度低'))
