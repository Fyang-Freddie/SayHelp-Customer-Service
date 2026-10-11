import asyncio,json
import pytest
from langchain_core.messages import AIMessage
from app.model_service import ModelService
from test_aftersales_workflow import ITEM

class Chat:
    def __init__(self,payload):self.payload=payload;self.options=None;self.messages=None
    def with_structured_output(self,schema,**options):self.options=options;return self
    async def ainvoke(self,messages):
        self.messages=messages
        return {'raw':AIMessage(content=json.dumps(self.payload,ensure_ascii=False),usage_metadata={'input_tokens':12,'output_tokens':8,'total_tokens':20}),'parsing_error':None}

def model(payload):
    service=object.__new__(ModelService);service._agent_model=Chat(payload);return service

def test_eligibility_strict_json_and_actual_usage():
    m=model({'status':'eligible','explanation':'满足条款','evidence_ids':['1'],'missing_facts':[]})
    result,usage=asyncio.run(m.judge_eligibility('能退吗',ITEM,[{'id':1,'answer':'条款'}],{},max_tokens=512))
    assert result.status=='eligible' and usage.total==20
    assert m._agent_model.options['method']=='json_mode' and m._agent_model.options['include_raw']
    assert m._agent_model.options['max_tokens']==512

@pytest.mark.parametrize('payload',[{'status':'eligible','explanation':'可以','evidence_ids':['1'],'missing_facts':[],'refund_amount':999},{'status':'unknown'}])
def test_invalid_eligibility_keeps_usage_on_error(payload):
    m=model(payload)
    with pytest.raises(ValueError) as caught:asyncio.run(m.judge_eligibility('能退吗',ITEM,[],{}))
    assert caught.value.usage.total==20

def test_customer_question_uses_existing_chat_model_only():
    m=model({'question':'请提供客户ID。'})
    question,usage=asyncio.run(m.ask_customer('退货'))
    assert question=='请提供客户ID。' and usage.total==20
