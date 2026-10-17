"""Jev-only business adapter and deterministic routing."""
from pathlib import Path
from app.ch06_types import Intent,IntentDecision,IntentOutcome
from app.evaluation.ch06 import fingerprints
from app.intent_policy import IntentPolicy
from app.jev_classifier import JevIntentClassifier
from app.workflow_types import Usage


class IntentClassificationError(ValueError):
    """Explicit fallback metadata without provider response content."""
    def __init__(self,usage: Usage,reason='invalid_response',decision=None):
        super().__init__('Intent classification requires fallback')
        self.usage,self.reason=usage,reason
        self.decision=decision or IntentDecision(intent='其他',confidence=0.0)


class IntentClassifier:
    """Independent decision node; calibrated policy is applied only in business code."""
    def __init__(self,settings,client=None):
        self.provider=JevIntentClassifier(settings,client)
        self.policy=IntentPolicy.from_artifact(Path(settings.intent_policy_path),
            fingerprints(settings))

    async def classify(self,query: str) -> IntentOutcome:
        return self.policy.apply(await self.provider.classify(query))


async def classify_intent(text: str, *, model) -> tuple[IntentDecision,Usage]:
    outcome=await model.classify_intent(text)
    if not isinstance(outcome,IntentOutcome):
        raise IntentClassificationError(Usage())
    if outcome.reason not in ('accepted','valid_other'):
        raise IntentClassificationError(outcome.usage,outcome.reason,outcome.decision)
    return outcome.decision,outcome.usage


def resolve_reference(text: str) -> str:
    return text


def route_intent(intent: str):
    routes={'物流':'business','订单':'business','商品咨询':'knowledge',
            '退款退货':'aftersales','售后':'aftersales','投诉':'complaint','闲聊':'chitchat','其他':'fallback'}
    try: return routes[intent]
    except (KeyError,TypeError): raise ValueError('Unknown intent; routing rejected') from None
