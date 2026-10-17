"""An absent, stale or low-confidence calibration must never admit a business route."""
from importlib import import_module
from pathlib import Path
import json
import pytest


def modules():
    try: return import_module('app.ch06_types'),import_module('app.intent_policy')
    except ModuleNotFoundError: pytest.fail('Intent policy is not implemented')


def outcome(confidence=0.6,reason='accepted'):
    types,_=modules()
    return types.IntentOutcome(types.IntentDecision(intent='退款退货',confidence=confidence),reason,'jev-test',{},types.Usage())


def policy(tmp_path, thresholds=None, fingerprint=None):
    path=tmp_path/'policy.json'
    path.write_text(json.dumps({'version':1,'fingerprints':fingerprint or {'model':'jev-test','criteria':'hash'},'thresholds':thresholds if thresholds is not None else {'退款退货':0.7}}),encoding='utf-8')
    return modules()[1].IntentPolicy.from_artifact(path,{'model':'jev-test','criteria':'hash'})


def test_low_confidence_preserves_number_and_changes_intent(tmp_path):
    decision=policy(tmp_path).apply(outcome())
    assert decision.reason=='low_confidence'
    assert decision.decision.model_dump()=={'intent':'其他','confidence':0.6}
    assert policy(tmp_path).apply(outcome(0.7)).decision.intent=='退款退货'


def test_missing_and_changed_artifacts_disable_business_routes(tmp_path):
    _,module=modules()
    missing=module.IntentPolicy.from_artifact(tmp_path/'missing.json',{'model':'jev-test'})
    assert missing.apply(outcome()).reason=='calibration_missing'
    assert policy(tmp_path,fingerprint={'model':'different'}).apply(outcome()).reason=='calibration_mismatch'
    assert policy(tmp_path,thresholds={}).apply(outcome(1.0)).decision.intent=='其他'


def test_actual_model_mismatch_and_provider_failure_remain_explicit(tmp_path):
    types,_=modules()
    changed=types.IntentOutcome(types.IntentDecision(intent='退款退货',confidence=1.0),'accepted','new-model',{},types.Usage())
    assert policy(tmp_path).apply(changed).reason=='calibration_mismatch'
    assert policy(tmp_path).apply(outcome(reason='timeout')).reason=='timeout'


@pytest.mark.parametrize('value',[True,float('nan'),-0.1,1.1])
def test_invalid_threshold_rejected(tmp_path,value):
    assert policy(tmp_path,thresholds={'退款退货':value}).apply(outcome()).reason=='calibration_mismatch'
