"""Useful self-check and reference boundaries before any answer reaches customers."""
import asyncio
from functools import wraps
import importlib
from types import SimpleNamespace
import pytest
from app.knowledge_types import ConfidenceDecision
from test_rag_evidence import result,settings
from app.evidence import select_prompt_evidence


def api(): return importlib.import_module('app.rag_generation')

def run_async(test):
    @wraps(test)
    def run(*args,**kwargs): return asyncio.run(test(*args,**kwargs))
    return run

class Model:
    def __init__(self,payload): self.payload=payload;self.calls=[]
    async def generate_knowledge(self,messages):
        self.calls.append(messages)
        if isinstance(self.payload,Exception): raise self.payload
        return self.payload

def setup(payload):
    query,retrieved=result(3);prompt=select_prompt_evidence(query,retrieved,settings())
    model=Model(payload)
    return model,query,prompt

@run_async
async def test_low_retrieval_or_empty_budget_never_calls_generation_model():
    module=api();model,query,prompt=setup({'useful':True,'answer':'有据回答[1]','reason':'证据足够'})
    answer=await module.RagGenerator(model).generate(query,prompt,ConfidenceDecision(False,'校准门槛未通过'))
    assert not answer.useful and answer.pool_source=='retrieval_low_conf' and model.calls==[]
    assert answer.answer==module.REFUSAL and answer.citations==prompt.citations
    empty=select_prompt_evidence(*result(0),settings())
    answer=await module.RagGenerator(model).generate(query,empty,ConfidenceDecision(True,'通过'))
    assert answer.pool_source=='retrieval_low_conf' and model.calls==[]

@run_async
async def test_generation_self_check_false_replaces_speculative_draft_and_preserves_full_evidence():
    module=api();model,query,prompt=setup({'useful':False,'answer':'猜测也许已经退款[99]','reason':'证据没有处理结果'})
    answer=await module.RagGenerator(model).generate(query,prompt,ConfidenceDecision(True,'通过'))
    assert answer.answer==module.REFUSAL and answer.pool_source=='self_check'
    assert answer.reason=='证据没有处理结果' and answer.citations==prompt.citations

@run_async
@pytest.mark.parametrize('payload',[
    {'useful':'true','answer':'回答[1]','reason':'足够'},
    {'useful':True,'answer':'回答[0]','reason':'足够'},
    {'useful':True,'answer':'回答[-1]','reason':'足够'},
    {'useful':True,'answer':'回答[99]','reason':'足够'},
    {'useful':True,'answer':'回答没有角标','reason':'足够'},
    {'useful':True,'answer':'已经退款[1]','reason':'足够'},
    {'useful':True,'answer':'保证退款明天到账[1]','reason':'足够'},
    {'useful':True,'answer':'保证商品绝对安全[1]','reason':'足够'},
    {'useful':True,'answer':'回答[1]','reason':'足够','extra':'未知'},
])
async def test_invalid_structure_citations_or_forbidden_promises_become_safe_self_check(payload):
    module=api();model,query,prompt=setup(payload)
    answer=await module.RagGenerator(model).generate(query,prompt,ConfidenceDecision(True,'通过'))
    assert not answer.useful and answer.pool_source=='self_check' and answer.answer==module.REFUSAL
    assert 'extra' not in answer.reason and answer.citations==prompt.citations

@run_async
async def test_valid_answer_keeps_entire_prompt_snapshot_including_uncited_chunks():
    module=api();model,query,prompt=setup({'useful':True,'answer':'仅依据原文回答[1]','reason':'覆盖全部问题'})
    answer=await module.RagGenerator(model).generate(query,prompt,ConfidenceDecision(True,'通过'))
    assert answer.useful and answer.pool_source is None and answer.citations==prompt.citations and len(answer.citations)==3
    assert model.calls==[prompt.messages]

@run_async
async def test_model_outage_is_service_failure_and_bad_json_is_self_check():
    module=api();model,query,prompt=setup(RuntimeError('secret transport diagnostic'))
    with pytest.raises(module.GenerationServiceError,match='unavailable') as error:
        await module.RagGenerator(model).generate(query,prompt,ConfidenceDecision(True,'通过'))
    assert 'secret' not in str(error.value)
    model.payload=module.GenerationFormatError('secret malformed payload')
    answer=await module.RagGenerator(model).generate(query,prompt,ConfidenceDecision(True,'通过'))
    assert answer.pool_source=='self_check' and 'secret' not in answer.reason


@run_async
@pytest.mark.parametrize('raw',[
    '{"useful":true,"answer":"依据[1]","reason":"完整"}',
    '{"useful":true,"useful":false,"answer":"依据[1]","reason":"完整"}',
    '{"useful":true,"answer":"依据[1]","reason":"完整","extra":1}',
    '{"useful":1,"answer":"依据[1]","reason":"完整"}',
    '{"useful":true,"answer":"依据[1]","reason":NaN}',
    'not json',
])
async def test_model_service_revalidates_raw_json_and_limits_generation_parameters(raw):
    from langchain_core.messages import AIMessage
    from app.model_service import ModelService
    calls=[]
    class Bound:
        async def ainvoke(self,messages):
            return {'raw':AIMessage(raw),'parsed':{'useful':True},'parsing_error':None}
    class FakeChat:
        def with_structured_output(self,schema,**kwargs): calls.append((schema,kwargs));return Bound()
    model=ModelService.__new__(ModelService);model._model=FakeChat();model._response_token_reserve=512
    assert hasattr(model,'generate_knowledge'),'Structured generation port missing'
    if raw=='{"useful":true,"answer":"依据[1]","reason":"完整"}':
        output=await model.generate_knowledge([])
        assert output.useful and output.answer=='依据[1]'
        assert calls[0][1]=={'method':'json_mode','include_raw':True,'temperature':0,'max_tokens':512}
    else:
        with pytest.raises(api().GenerationFormatError): await model.generate_knowledge([])


@run_async
async def test_parse_error_from_raw_and_transport_failure_remain_distinct():
    from app.model_service import ModelService
    from langchain_core.messages import AIMessage
    class FakeChat:
        def with_structured_output(self,*args,**kwargs): return self
        async def ainvoke(self,messages):
            if self.outage: raise RuntimeError('private network diagnostic')
            return {'raw':AIMessage('bad'),'parsed':None,'parsing_error':ValueError('private parser diagnostic')}
    model=ModelService.__new__(ModelService);model._model=FakeChat();model._response_token_reserve=512;model._model.outage=False
    assert hasattr(model,'generate_knowledge'),'Structured generation port missing'
    with pytest.raises(api().GenerationFormatError): await model.generate_knowledge([])
    model._model.outage=True
    with pytest.raises(RuntimeError,match='private network'): await model.generate_knowledge([])


@run_async
async def test_negative_warning_is_not_a_forbidden_positive_promise():
    module=api();model,query,prompt=setup({'useful':True,'answer':'超保或人为损坏可能收费，不能保证免费维修，也无法保证物流送达时间[1]。','reason':'证据覆盖规则'})
    answer=await module.RagGenerator(model).generate(query,prompt,ConfidenceDecision(True,'通过'))
    assert answer.useful and answer.pool_source is None



def test_upstream_requests_have_bounded_timeout_and_retry_budget(monkeypatch):
    import app.model_service as module
    from app.config import Settings
    captured=[]
    monkeypatch.setattr(module,'ChatOpenAI',lambda **kwargs:captured.append(kwargs))
    module.ModelService(Settings(chat_base_url='https://example.invalid',chat_model='test',chat_api_key='placeholder'))
    assert captured[0].get('timeout')==180 and captured[0].get('max_retries')==1
