"""Calibration must reject a test split before loading models or making requests."""
import asyncio
from types import SimpleNamespace
import pytest
from app.calibrate_ch04 import calibrate
from app.evaluation.dataset import load_cases
from app.ch04_ingest import ROOT


def test_formal_test_split_cannot_choose_confidence_thresholds():
    with pytest.raises(ValueError,match='calibration12'):
        asyncio.run(calibrate(SimpleNamespace(),ROOT/'eval/ch04/test.json'))


def test_thresholds_have_explicit_false_accept_false_reject_and_finite_boundaries():
    import math
    from app.evaluation.calibration import choose_threshold
    chosen=choose_threshold([(6.0,True),(8.0,False)])
    assert chosen['threshold']>8.0 and chosen['false_accept']==0 and chosen['false_reject']==1
    assert math.isfinite(chosen['threshold'])
    with pytest.raises(ValueError): choose_threshold([])
    with pytest.raises(ValueError): choose_threshold([(float('nan'),False)])
