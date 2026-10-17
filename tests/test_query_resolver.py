"""Query resolution guards preserve the customer's facts and isolate bounded history."""
import asyncio,json
from importlib import import_module
from langchain_core.messages import AIMessage
import pytest

class Model:
    def __init__(self,payload): self.payload=payload; self.requests=[]
    async def resolve_query(self,messages):
        self.requests.append(messages)
        return AIMessage(content=json.dumps(self.payload,ensure_ascii=False),usage_metadata={'input_tokens':10,'output_tokens':5,'total_tokens':15})

def resolve(model,raw,history=None,confirmed=None):
    try: cls=import_module('app.query_resolver').QueryResolver
    except ModuleNotFoundError: pytest.fail('QueryResolver missing')
    return asyncio.run(cls(model).resolve(raw,history or [],confirmed or {}))

def test_unchanged_is_byte_for_byte_even_if_model_polishes_payload():
    raw='  订单001不要退款，只查物流。\n'
    result,usage=resolve(Model({'query':'订单001物流在哪','changed':False,'unresolved_references':[]}),raw)
    assert result.query==raw and usage.total==15

@pytest.mark.parametrize('raw,query',[
    ('订单001买了2件，能退吗','订单001买了1件，能退吗'),
    ('不退款，只查物流','查询物流'),
    ('蓝牙耳机能退吗','订单TB999999的蓝牙耳机能退吗'),
])
def test_changed_query_cannot_drop_numbers_negation_or_invent_ids(raw,query):
    model=Model({'query':query,'changed':True,'unresolved_references':[]})
    try: module=import_module('app.query_resolver')
    except ModuleNotFoundError: pytest.fail('QueryResolver missing')
    with pytest.raises(module.QueryResolutionError) as caught: resolve(model,raw)
    assert caught.value.usage.total==15

def test_unknown_reference_is_retained_for_execution_selection():
    result,_=resolve(Model({'query':'这个能退吗','changed':False,'unresolved_references':['这个']}),'这个能退吗')
    assert result.unresolved_references==['这个']

def test_history_is_bounded_complete_and_same_conversation():
    history=[{'role':'user','content':str(i)*100,'conversation_id':7,'completed':True} for i in range(30)]
    history += [{'role':'user','content':'FOREIGN','conversation_id':8}, {'role':'assistant','content':'PENDING','conversation_id':7,'completed':False}]
    model=Model({'query':'你好','changed':False,'unresolved_references':[]})
    resolve(model,'你好',history,{'conversation_id':7,'customer_id':'C01'})
    payload=json.loads(model.requests[0][-1].content)
    assert len(payload['history'])<=8
    assert 'FOREIGN' not in json.dumps(payload) and 'PENDING' not in json.dumps(payload)

def test_ids_in_assistant_history_are_not_confirmed_customer_facts():
    model=Model({'query':'订单TB123456能退吗','changed':True,'unresolved_references':[]})
    try: module=import_module('app.query_resolver')
    except ModuleNotFoundError: pytest.fail('QueryResolver missing')
    with pytest.raises(module.QueryResolutionError):
        resolve(model,'能退吗',[{'role':'assistant','content':'订单TB123456'}])

def test_user_supplied_id_can_be_resolved():
    model=Model({'query':'订单TB123456能退吗','changed':True,'unresolved_references':[]})
    result,_=resolve(model,'能退吗',[{'role':'user','content':'订单TB123456'}])
    assert result.query=='订单TB123456能退吗'


def test_id_prefix_is_not_a_confirmed_identifier():
    model=Model({'query':'订单TB1234能退吗','changed':True,'unresolved_references':[]})
    module=import_module('app.query_resolver')
    with pytest.raises(module.QueryResolutionError):
        resolve(model,'能退吗',[{'role':'user','content':'订单TB123456'}])


def test_unresolved_references_are_actual_words_not_missing_business_slots():
    model=Model({'query':'申请退款','changed':False,'unresolved_references':['退款需要的订单号']})
    with pytest.raises(import_module('app.query_resolver').QueryResolutionError):
        resolve(model,'申请退款')
