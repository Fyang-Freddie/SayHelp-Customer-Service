"""Business compatibility must accept only Jev outcomes, never generated chat JSON."""
import asyncio
from dataclasses import replace
import json
import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from app.ch06_types import IntentDecision,IntentOutcome,INTENTS
from app.workflow_types import Usage
from app.intent import classify_intent,IntentClassificationError,route_intent
from app.config import Settings
from app.model_service import ModelService


class Provider:
    def __init__(self,outcome): self.outcome=outcome; self.queries=[]
    async def classify_intent(self,text):
        self.queries.append(text)
        return self.outcome


def outcome(intent='订单',reason='accepted'):
    return IntentOutcome(IntentDecision(intent=intent,confidence=.91),reason,'jev-test',{},Usage(41,9))


@pytest.mark.parametrize('intent',INTENTS)
def test_jev_result_compatible_with_two_field_business_contract(intent):
    model=Provider(outcome(intent,'valid_other' if intent=='其他' else 'accepted'))
    decision,usage=asyncio.run(classify_intent('订单001不是要退款',model=model))
    assert decision.model_dump()=={'intent':intent,'confidence':.91}
    assert usage.total==50 and model.queries==['订单001不是要退款']


@pytest.mark.parametrize('reason',['timeout','provider_error','invalid_response','low_confidence','calibration_missing','calibration_mismatch'])
def test_failure_keeps_explicit_reason_and_public_fallback(reason):
    model=Provider(outcome('其他',reason))
    with pytest.raises(IntentClassificationError) as caught:
        asyncio.run(classify_intent('问题',model=model))
    assert caught.value.reason==reason and caught.value.usage.total==50
    assert caught.value.decision.model_dump()=={'intent':'其他','confidence':.91}


def test_generated_chat_json_is_not_a_classifier():
    model=Provider(AIMessage(content='{"intent":"订单","confidence":1}'))
    with pytest.raises(IntentClassificationError):
        asyncio.run(classify_intent('问题',model=model))


@pytest.mark.parametrize('payload',[{'intent':'其他'},{'intent':'订单','confidence':True},{'intent':'新类别','confidence':1.0},{'intent':'订单','confidence':.9,'route':'business'}])
def test_public_schema_is_strict(payload):
    with pytest.raises(ValidationError): IntentDecision.model_validate(payload)


def test_other_is_explicit_fallback(): assert route_intent('其他')=='fallback'


def test_model_service_never_calls_chat_for_intent(monkeypatch):
    import app.model_service as module
    class Chat:
        def __init__(self,**kwargs): pass
        async def ainvoke(self,*args,**kwargs): raise AssertionError('Generative classification forbidden')
    monkeypatch.setattr(module,'ChatOpenAI',Chat)
    service=ModelService(Settings('https://example.invalid','test','not-real'))
    decision=asyncio.run(service.classify_intent('要退货'))
    assert isinstance(decision,IntentOutcome)
    assert decision.reason=='configuration_missing' and decision.decision.intent=='其他'
