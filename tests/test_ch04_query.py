"""One-turn normalization preserves constraints; model tests use an explicit boundary."""
import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest
from app.model_service import ModelService


def api(): return importlib.import_module('app.query_understanding')


def decision(standard='退货邮费由谁承担？',intent='knowledge',synonyms=None,clarification=None):
    return {'standard_question':standard,'synonyms':synonyms or [],'intent':intent,'clarification':clarification}


class Model:
    def __init__(self,payload=None,error=None): self.payload=payload; self.error=error; self.calls=[]
    async def understand_query(self,raw):
        self.calls.append(raw)
        if self.error: raise self.error
        return api().QueryDecision.model_validate(self.payload)


@pytest.mark.parametrize('patch',[
    {'extra':'not allowed'},{'intent':'random_product'},{'synonyms':['x']*9},
    {'synonyms':['中'*33]},{'synonyms':[1]},{'standard_question':'  '},
    {'intent':None},{'clarification':5},
],ids=['extra','intent','too_many','too_long','type','blank','null','clarification'])
def test_decision_schema_rejects_invalid_output(patch):
    from pydantic import ValidationError
    with pytest.raises(ValidationError): api().QueryDecision.model_validate({**decision(),**patch})


def test_normalization_only_receives_current_raw_question_and_expands_search_side():
    raw='亲，mh- lp50能不能用App呀？'
    model=Model(decision('MH-LP50能不能使用App？',synonyms=['应用','手机控制','应用']))
    result=asyncio.run(api().QueryUnderstanding(model).prepare(raw))
    assert model.calls==[raw] and result.raw_question==raw
    assert result.standard_question=='MH-LP50能不能使用App？'
    assert result.intent=='knowledge' and result.synonyms==['应用','手机控制']
    assert 'mh- lp50' in result.search_text and '应用' in result.search_text
    assert result.preserved_models==['mh- lp50'] and result.preserved_numbers==['50']


@pytest.mark.parametrize('raw,standard',[
    ('MH-LP50尺寸多少','猫砂盆尺寸多少'),
    ('MH-LP50尺寸多少','MH-LP100尺寸多少'),
    ('MH-LP50适合5kg猫吗','MH-LP50适合6kg猫吗'),
    ('MH-LP50没有App，适合5kg猫吗','MH-LP50支持App，适合5kg猫吗'),
    ('不是质量问题，7天内退货邮费谁出','质量问题，7天内退货邮费谁出'),
],ids=['missing_model','changed_model','number','negation','non_quality'])
def test_lost_model_number_or_negation_returns_clarification(raw,standard):
    result=asyncio.run(api().QueryUnderstanding(Model(decision(standard))).prepare(raw))
    assert result.intent=='clarify' and result.clarification


def test_quality_negation_scope_cannot_be_replaced_by_unrelated_negation():
    raw='不是质量问题，7天内退货邮费谁出'
    result=asyncio.run(api().QueryUnderstanding(Model(decision('质量问题7天退货不需要承担运费'))).prepare(raw))
    assert result.intent=='clarify'


@pytest.mark.parametrize('raw', ['那款呢？','它支持App吗？','上一个型号的质保呢？'])
def test_requires_history_is_clarified_without_coreference(raw):
    result=asyncio.run(api().QueryUnderstanding(Model(decision('商品功能是什么'))).prepare(raw))
    assert result.intent=='clarify'


def test_stock_and_current_price_never_route_to_random_product_tool():
    raw='MH-LP50现在有库存吗，多少钱？'
    result=asyncio.run(api().QueryUnderstanding(Model(decision(raw,intent='operation'))).prepare(raw))
    assert result.intent=='knowledge'


def test_unknown_explicit_model_is_kept_for_evidence_based_refusal():
    raw='MH-X999有自动清洁吗？'
    result=asyncio.run(api().QueryUnderstanding(Model(decision(raw))).prepare(raw))
    assert result.intent=='knowledge' and result.preserved_models==['MH-X999']


@pytest.mark.parametrize('payload', [None, {'standard_question':'ok','intent':'knowledge'}, {'intent':'wrong'}])
def test_invalid_model_decision_is_safe_service_error(payload):
    with pytest.raises(api().QueryUnderstandingError) as error:
        asyncio.run(api().QueryUnderstanding(Model(payload)).prepare('原话'))
    assert 'private-provider-secret' not in str(error.value)


def test_model_failure_is_safe_service_error():
    with pytest.raises(api().QueryUnderstandingError) as error:
        asyncio.run(api().QueryUnderstanding(Model(error=RuntimeError('private-provider-secret'))).prepare('原话'))
    assert 'private-provider-secret' not in str(error.value)


@pytest.mark.parametrize('raw',['not JSON','{"intent": "wrong"}',json.dumps({**decision(),'extra':'x'})])
def test_model_service_strictly_revalidates_raw_json(raw):
    class Structured:
        async def ainvoke(self,messages): return {'raw':SimpleNamespace(content=raw),'parsed':None,'parsing_error':None}
    class Client:
        def with_structured_output(self,schema,*,method,include_raw):
            assert method=='json_mode' and include_raw is True
            return Structured()
    model=ModelService.__new__(ModelService); model._model=Client()
    with pytest.raises(ValueError): asyncio.run(model.understand_query('用户原话'))


def test_model_service_prompt_contains_only_this_question_and_schema():
    captured=[]
    class Structured:
        async def ainvoke(self,messages):
            captured.extend(messages)
            return {'raw':SimpleNamespace(content=json.dumps(decision())),'parsed':None,'parsing_error':None}
    class Client:
        def with_structured_output(self,*args,**kwargs): return Structured()
    model=ModelService.__new__(ModelService); model._model=Client()
    result=asyncio.run(model.understand_query('邮费谁出'))
    assert result.intent=='knowledge' and len(captured)==2
    assert captured[1].content=='邮费谁出'
    assert 'JSON' in captured[0].content


def test_no_app_condition_cannot_be_replaced_by_other_negative_fact():
    raw='MH-LP50没有App，适合5kg猫吗'
    model=Model(decision('MH-LP50支持App，但不适合5kg猫'))
    assert asyncio.run(api().QueryUnderstanding(model).prepare(raw)).intent=='clarify'


def test_interrogative_negation_is_not_a_negative_condition():
    raw='MH-LP50能不能用App？'
    result=asyncio.run(api().QueryUnderstanding(Model(decision('MH-LP50是否支持App？'))).prepare(raw))
    assert result.intent=='knowledge'


def test_synonym_expansion_discards_new_numbers_and_other_model():
    raw='MH-LP50退货邮费谁出'
    model=Model(decision('MH-LP50退货邮费谁承担？',synonyms=['运费','MH-LP100','退款2小时到账']))
    result=asyncio.run(api().QueryUnderstanding(model).prepare(raw))
    assert result.synonyms==['运费'] and 'MH-LP100' not in result.search_text and '2小时' not in result.search_text


@pytest.mark.parametrize('raw,standard',[
    ('MH-LP50有没有App','MH-LP50和MH-LP100有没有App'),
    ('7天退货运费怎么收','7天退货运费收10元'),
    ('MH-LP50适合5kg猫吗','MH-LP50适合5g猫吗'),
    ('质量问题7天内换货运费谁出','非质量问题7天内换货运费谁出'),
],ids=['extra_model','extra_number','unit_changed','quality_flipped'])
def test_invented_constraint_is_not_normalization(raw,standard):
    result=asyncio.run(api().QueryUnderstanding(Model(decision(standard))).prepare(raw))
    assert result.intent=='clarify'


def test_colloquial_trailing_bu_is_an_interrogative():
    raw='MH-LP50和MH-LP100都能用App不？'
    model=Model(decision('MH-LP50和MH-LP100是否都支持App？'))
    result=asyncio.run(api().QueryUnderstanding(model).prepare(raw))
    assert result.intent=='knowledge' and result.preserved_negations==[]
