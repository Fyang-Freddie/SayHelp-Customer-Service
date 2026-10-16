import asyncio
import sqlite3
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk


def test_bound_structured_and_failed_calls_share_durable_quota(tmp_path,monkeypatch):
    from app.ch07_evaluation import EvaluationBudget,EvaluationLimitExceeded,MeteredModel
    path=tmp_path/'calls.sqlite'
    meter=EvaluationBudget(path,limits={'chat':2},buckets={'dialogue':2})
    class Provider:
        calls=0
        def bind_tools(self,*args,**kwargs):return self
        def with_structured_output(self,*args,**kwargs):return self
        async def ainvoke(self,*args,**kwargs):
            self.calls+=1
            if self.calls==2:raise ConnectionError('synthetic')
            return {'raw':AIMessage('ok',usage_metadata={'input_tokens':3,'output_tokens':1,'total_tokens':4})}
    provider=Provider(); model=MeteredModel(provider,meter)
    async def run():
        await model.bind_tools([]).with_structured_output(dict).ainvoke([])
        with pytest.raises(ConnectionError):await model.ainvoke([])
        with pytest.raises(EvaluationLimitExceeded):await model.ainvoke([])
    asyncio.run(run())
    assert provider.calls==2
    with sqlite3.connect(path) as db:
        assert [r[0] for r in db.execute('SELECT outcome FROM calls ORDER BY rowid')]==['success','failed']


def test_stream_cancel_consumes_one_request(tmp_path):
    from app.ch07_evaluation import EvaluationBudget,MeteredModel
    from contextlib import aclosing
    meter=EvaluationBudget(tmp_path/'calls.sqlite',limits={'chat':1},buckets={'dialogue':1})
    class Provider:
        async def astream(self,*args,**kwargs):
            yield AIMessageChunk('first')
            await asyncio.Event().wait()
    async def run():
        async with aclosing(MeteredModel(Provider(),meter).astream([])) as stream:
            await anext(stream)
    asyncio.run(run())
    with sqlite3.connect(meter.path) as db:
        assert db.execute('SELECT outcome FROM calls').fetchone()[0]=='failed'
def test_every_production_model_operation_rejects_oversized_exact_request():
    import asyncio
    from types import SimpleNamespace
    import pytest
    from langchain_core.messages import HumanMessage
    from app.model_service import ModelService
    from app.ch06_types import Resolution
    from app.history import InputBudgetExceeded
    service=object.__new__(ModelService)
    service._settings=SimpleNamespace(layered_context=True,context_token_budget=2000)
    class NeverCalled:
        def __getattr__(self,name):raise AssertionError('oversized request reached provider binding')
    service._agent_model=NeverCalled()
    messages=[HumanMessage('长'*2100)]
    async def run():
        for operation in (
            lambda:service.resolve_query(messages),lambda:service.expand_query(messages),
            lambda:service._ch06_structured(Resolution,messages,512),
            lambda:service.select(messages,[],max_tokens=512)):
            with pytest.raises(InputBudgetExceeded):await operation()
        with pytest.raises(InputBudgetExceeded):
            async for _ in service.stream_reply(messages,max_tokens=512):pass
    asyncio.run(run())


def test_recap_provider_requires_valid_json_answer_before_public_output():
    import asyncio
    from types import SimpleNamespace
    from langchain_core.messages import HumanMessage,AIMessage
    from app.model_service import ModelService
    class Provider:
        def __init__(self):self.payload='{"answer":"订单ABC123缺件，待核实。"}'
        def with_structured_output(self,*args,**kwargs):
            assert kwargs['method']=='json_mode' and kwargs['include_raw']
            return self
        async def ainvoke(self,messages):return {'raw':AIMessage(content=self.payload),'parsing_error':None}
    service=object.__new__(ModelService)
    service._settings=SimpleNamespace(layered_context=True,context_token_budget=18000)
    provider=Provider();service._agent_model=provider
    messages=[HumanMessage('请复述订单ABC123')]
    assert asyncio.run(service.recap_reply(messages,max_tokens=512)).content=='订单ABC123缺件，待核实。'
    provider.payload='<｜｜DSML｜｜ calls>query_order(ABC123)'
    import pytest
    with pytest.raises(ValueError):asyncio.run(service.recap_reply(messages,max_tokens=512))

\n