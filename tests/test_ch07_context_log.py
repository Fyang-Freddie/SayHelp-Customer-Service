import json
from langchain_core.messages import HumanMessage, AIMessage


def test_logged_request_equals_model_spy_payload(tmp_path):
    from app.context_log import ContextLog
    path=tmp_path/'app.log'
    log=ContextLog(path)
    messages=[HumanMessage('订单123原话'),AIMessage('前次答复')]
    tools=[{'type':'function','function':{'name':'query_order','parameters':{}}}]
    log.model_request(1,'turn','select',messages,tools,{'history':100},'早期摘要全文')
    row=json.loads(path.read_text(encoding='utf8'))
    assert row['event']=='model_ctx'
    assert [m['content'] for m in row['messages']]==[m.content for m in messages]
    assert row['tools']==tools
    assert row['summary']=='早期摘要全文'
    assert row['window_count']==2


def test_context_log_preserves_text_as_data(tmp_path):
    from app.context_log import ContextLog
    path=tmp_path/'app.log';log=ContextLog(path)
    log.write('history_ctx',{'messages':[{'role':'user','content':'换行\n{"event":"伪造"}'}]})
    lines=path.read_text(encoding='utf8').splitlines()
    assert len(lines)==1
    assert json.loads(lines[0])['messages'][0]['content'].startswith('换行\n')


def test_resolver_logs_the_exact_model_adapter_messages(tmp_path):
    import asyncio
    from app.context_log import ContextLog
    from app.query_resolver import QueryResolver
    class Spy:
        sent=None
        async def resolve_query(self,messages):
            self.sent=[m.model_dump(mode='json') for m in messages]
            return AIMessage(content='{"query":"订单ABC123","changed":false,"unresolved_references":[]}')
    model=Spy();log=ContextLog(tmp_path/'app.log')
    def record(messages):
        log.model_request(1,'turn','resolve',messages,[],{'actual_input':len(messages)},'早期摘要')
    asyncio.run(QueryResolver(model).resolve('订单ABC123',[],{},on_request=record))
    row=json.loads((tmp_path/'app.log').read_text(encoding='utf8'))
    assert row['messages']==model.sent and row['phase']=='resolve'


def test_auxiliary_structured_request_callback_sees_actual_adapter_payload():
    import asyncio
    from types import SimpleNamespace
    from app.model_service import ModelService
    class Provider:
        sent=None
        def with_structured_output(self,*args,**kwargs):return self
        async def ainvoke(self,messages):
            self.sent=[m.model_dump(mode='json') for m in messages]
            return {'raw':AIMessage(content='{"question":"请提供订单号"}'),
                'parsing_error':None}
    service=object.__new__(ModelService)
    service._settings=SimpleNamespace(layered_context=True,context_token_budget=18000)
    provider=Provider();service._agent_model=provider
    captured=[]
    asyncio.run(service.ask_customer('订单待核实',on_request=lambda messages:
        captured.extend(m.model_dump(mode='json') for m in messages)))
    assert captured==provider.sent
