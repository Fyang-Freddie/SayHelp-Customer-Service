"""Narrow, non-generative Jev choice adapter. No provider response escapes on failure."""
import json
import math
from urllib.parse import urlparse
import httpx
from app.ch06_types import INTENTS, IntentDecision, IntentOutcome, Usage
from app.intent_criteria import INSTRUCTIONS, CRITERIA


def strict_json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError('duplicate_key')
            result[key]=value
        return result
    def constant(value): raise ValueError('nonfinite')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)


def failure(reason,usage=None):
    return IntentOutcome(IntentDecision(intent='其他',confidence=0.0),reason,'',{},usage or Usage())


def _number(value):
    return type(value) in (int,float) and math.isfinite(value) and 0<=value<=1


def parse_response(raw: bytes) -> IntentOutcome:
    if len(raw)>128000: raise ValueError('oversized_response')
    payload=strict_json(raw)
    if not isinstance(payload,dict) or set(payload)!={'model','answers','usage'}: raise ValueError('invalid_envelope')
    if not isinstance(payload['model'],str) or not payload['model']: raise ValueError('invalid_model')
    if not isinstance(payload['answers'],dict) or set(payload['answers'])!={'intent'}: raise ValueError('invalid_answers')
    answer=payload['answers']['intent']
    if not isinstance(answer,dict) or set(answer)!={'type','choice','confidence','probabilities'}: raise ValueError('invalid_answer')
    probs=answer['probabilities']
    if answer['type']!='choice' or answer['choice'] not in INTENTS or not _number(answer['confidence']): raise ValueError('invalid_choice')
    if not isinstance(probs,dict) or set(probs)!=set(INTENTS) or not all(_number(v) for v in probs.values()): raise ValueError('invalid_probabilities')
    if abs(sum(probs.values())-1)>1e-6 or probs[answer['choice']]<max(probs.values()): raise ValueError('inconsistent_choice')
    usage=payload['usage']
    if not isinstance(usage,dict) or set(usage)!={'input_tokens','output_tokens'}: raise ValueError('invalid_usage')
    tokens=Usage(**usage)
    return IntentOutcome(IntentDecision(intent=answer['choice'],confidence=answer['confidence']),
                         'accepted',payload['model'],dict(probs),tokens)


class JevIntentClassifier:
    def __init__(self,settings,client: httpx.AsyncClient | None=None):
        self.settings,self.client=settings,client

    async def classify(self,query: str) -> IntentOutcome:
        if not isinstance(query,str) or not query.strip() or len(query)>4000: return failure('invalid_query')
        settings=self.settings
        if not settings.jev_api_key or not settings.jev_model: return failure('configuration_missing')
        base=settings.jev_base_url.rstrip('/')
        parsed=urlparse(base)
        if (parsed.scheme,parsed.netloc,parsed.path)!=('https','api.typesafe.ai','/v1') or parsed.query or parsed.fragment:
            return failure('configuration_invalid')
        timeout=settings.jev_timeout
        if type(timeout) not in (int,float) or not math.isfinite(timeout) or not 0<timeout<=60: return failure('configuration_invalid')
        body={'model':settings.jev_model,'state':query,'questions':{'intent':{'type':'choice','instructions':INSTRUCTIONS,'criteria':CRITERIA}}}
        try:
            async def call(client):
                from app.ch07_evaluation import evaluation_call
                with evaluation_call('jev'):
                    return await client.post(base+'/systemone',json=body,
                        headers={'Authorization':'Bearer '+settings.jev_api_key},timeout=timeout,follow_redirects=False)
            if self.client is None:
                async with httpx.AsyncClient() as client: response=await call(client)
            else: response=await call(self.client)
            if response.status_code!=200: return failure('provider_error')
            try: return parse_response(response.content)
            except (ValueError,TypeError,KeyError,UnicodeError): return failure('invalid_response')
        except httpx.TimeoutException: return failure('timeout')
        except httpx.HTTPError: return failure('provider_error')
