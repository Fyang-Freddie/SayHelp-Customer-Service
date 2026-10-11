"""Exercise provider HTTP/JSON boundaries without sending credentials over a network."""
import asyncio
from importlib import import_module
import json
from types import SimpleNamespace
import httpx
import pytest

LABELS=['物流','订单','商品咨询','退款退货','售后','投诉','闲聊','其他']


def api():
    try: return import_module('app.jev_classifier')
    except ModuleNotFoundError: pytest.fail('Jev classifier is not implemented')


def result():
    return {'model':'jev-test-version','answers':{'intent':{'type':'choice','choice':'退款退货','probabilities':{x:1.0 if x=='退款退货' else 0.0 for x in LABELS},'confidence':0.9}},'usage':{'input_tokens':100,'output_tokens':20}}


def settings():
    return SimpleNamespace(jev_api_key='test-secret',jev_model='jev-test-version',jev_base_url='https://api.typesafe.ai/v1',jev_timeout=10)


def classify(handler, query='蓝牙耳机可以退货吗？'):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await api().JevIntentClassifier(settings(),client).classify(query)
    return asyncio.run(run())


def test_state_is_only_complete_query_and_output_has_only_two_fields():
    def handle(request):
        payload=json.loads(request.content)
        assert payload['state']=='蓝牙耳机可以退货吗？'
        assert set(payload)=={'state','questions','model'}
        assert set(payload['questions'])=={'intent'}
        assert payload['questions']['intent']['type']=='choice'
        assert set(payload['questions']['intent']['criteria'])==set(LABELS)
        assert request.url==httpx.URL('https://api.typesafe.ai/v1/systemone')
        return httpx.Response(200,json=result())
    outcome=classify(handle)
    assert outcome.decision.model_dump()=={'intent':'退款退货','confidence':0.9}
    assert outcome.reason=='accepted' and outcome.usage.input_tokens==100
    assert outcome.model=='jev-test-version'


@pytest.mark.parametrize('change', ['missing_probability','bad_choice','nan','bool','extra','not_max','bad_sum','wrong_type'])
def test_malformed_results_fail_closed(change):
    payload=result(); answer=payload['answers']['intent']
    if change=='missing_probability': del answer['probabilities']['其他']
    if change=='bad_choice': answer['choice']='未定义'
    if change=='nan': answer['confidence']=float('nan')
    if change=='bool': answer['confidence']=True
    if change=='extra': answer['intent']='退款退货'
    if change=='not_max': answer['choice']='订单'
    if change=='bad_sum': answer['probabilities']['订单']=0.5
    if change=='wrong_type': answer['type']='noul'
    outcome=classify(lambda _:httpx.Response(200,content=json.dumps(payload)))
    assert outcome.reason=='invalid_response'
    assert outcome.decision.model_dump()=={'intent':'其他','confidence':0.0}


def test_duplicate_json_keys_rejected():
    body=json.dumps(result()).replace('"confidence": 0.9','"confidence": 0.9, "confidence": 1.0')
    assert classify(lambda _:httpx.Response(200,content=body)).reason=='invalid_response'


def test_http_failure_and_timeout_are_explicit_without_provider_text():
    failed=classify(lambda _:httpx.Response(401,text='test-secret'))
    assert failed.reason=='provider_error' and 'test-secret' not in repr(failed)
    def timeout(request): raise httpx.ReadTimeout('test-secret',request=request)
    failed=classify(timeout)
    assert failed.reason=='timeout' and 'test-secret' not in repr(failed)


def test_redirect_is_not_followed():
    count=[]
    def handle(request):
        count.append(request.url.host)
        return httpx.Response(307,headers={'Location':'https://example.invalid/steal'})
    assert classify(handle).reason=='provider_error'
    assert count==['api.typesafe.ai']


def test_cancellation_propagates():
    def handle(request): raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError): classify(handle)


def test_settings_hide_jev_key_and_read_central_configuration(monkeypatch,tmp_path):
    from app.config import Settings
    monkeypatch.chdir(tmp_path)
    for name,value in {'CHAT_BASE_URL':'https://example.invalid/v1','CHAT_MODEL':'test','CHAT_API_KEY':'chat-secret','DATABASE_URL':'mysql+pymysql://u:p@localhost/db','TYPESAFE_API_KEY':'jev-private','JEV_MODEL':'jev-pinned','JEV_TIMEOUT_SECONDS':'12'}.items():
        monkeypatch.setenv(name,value)
    config=Settings.from_env()
    assert config.jev_model=='jev-pinned' and config.jev_timeout==12
    assert config.jev_api_key=='jev-private' and 'jev-private' not in repr(config)
