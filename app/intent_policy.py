"""One calibrated policy owns all confidence gates and explicit intent fallbacks."""
from dataclasses import replace
import math
from pathlib import Path
from app.ch06_types import INTENTS, IntentDecision, IntentOutcome
from app.jev_classifier import strict_json


class IntentPolicy:
    def __init__(self,thresholds,expected,reason=''):
        self.thresholds,self.expected,self.reason=dict(thresholds),dict(expected),reason

    @classmethod
    def from_artifact(cls,path: Path,expected: dict):
        try:
            payload=strict_json(Path(path).read_bytes())
        except FileNotFoundError: return cls({},expected,'calibration_missing')
        except (OSError,ValueError,TypeError): return cls({},expected,'calibration_mismatch')
        try:
            values=payload['thresholds']
            if payload['version']!=1 or payload['fingerprints']!=expected or not isinstance(values,dict): raise ValueError()
            if not set(values)<=set(INTENTS)-{'其他'}: raise ValueError()
            if any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1 for v in values.values()): raise ValueError()
            return cls(values,expected)
        except (KeyError,TypeError,ValueError): return cls({},expected,'calibration_mismatch')

    def apply(self,outcome: IntentOutcome) -> IntentOutcome:
        def other(reason,confidence=None):
            return replace(outcome,decision=IntentDecision(intent='其他',confidence=outcome.decision.confidence if confidence is None else confidence),reason=reason)
        if outcome.reason!='accepted': return other(outcome.reason,0.0)
        if self.reason: return other(self.reason)
        if outcome.model!=self.expected.get('model'): return other('calibration_mismatch')
        if outcome.decision.intent=='其他': return other('valid_other')
        threshold=self.thresholds.get(outcome.decision.intent)
        if threshold is None or outcome.decision.confidence<threshold: return other('low_confidence')
        return outcome
