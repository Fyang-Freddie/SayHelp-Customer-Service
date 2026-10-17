"""Calibration must use calibration-only observations and fail closed per class."""
from importlib import import_module
import pytest

def calibrate(rows):
    try: api=import_module('app.evaluation.ch06')
    except ModuleNotFoundError: pytest.fail('Calibration missing')
    return api.calibrate_intents(rows,{'model':'test','criteria':'hash'})

def row(expected,predicted,confidence,split='calibration',query_ok=True):
    return dict(expected_intent=expected,intent=predicted,confidence=confidence,split=split,query_ok=query_ok,reason='accepted')

def test_acceptance_is_never_used_for_threshold_selection():
    with pytest.raises(ValueError): calibrate([row('订单','订单',.9,'acceptance')])

def test_high_confidence_wrong_example_disables_that_business_class():
    result=calibrate([row('订单','退款退货',1.0),row('退款退货','退款退货',.9)])
    assert '退款退货' not in result['thresholds']

def test_real_observed_cutpoint_maximizes_safe_coverage():
    result=calibrate([row('订单','售后',.3),row('售后','售后',.6),row('售后','售后',.9),row('其他','其他',1)])
    assert result['thresholds']=={'售后':.6}
    assert result['automatic_correct']==2 and result['total']==4

def test_bad_resolution_is_not_a_correct_route_even_with_correct_label():
    result=calibrate([row('售后','售后',1,query_ok=False)])
    assert result['thresholds']=={}
