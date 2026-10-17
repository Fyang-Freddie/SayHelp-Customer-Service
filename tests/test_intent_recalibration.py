"""Frozen Jev-only recalibration must improve coverage without unsafe routing."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.evaluation.intent_recalibration import (
    AttemptLedger, assess_thresholds, candidate_is_safer, load_frozen_cases,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'eval/ch06/intent-recalibration-v1.json'


def test_frozen_cases_are_disjoint_complete_and_include_reported_failure():
    cases, digest = load_frozen_cases(CASES)
    assert len(cases) == 220 and len(digest) == 64
    assert len({row['query'] for row in cases}) == 220
    assert sum(row['split'] == 'calibration' for row in cases) == 110
    assert sum(row['split'] == 'acceptance' for row in cases) == 110
    assert any('TB202611000001' in row['query'] and row['expected_intent'] == '投诉'
               for row in cases if row['split'] == 'calibration')


def test_frozen_cases_reject_duplicate_or_modified_schema(tmp_path):
    data = json.loads(CASES.read_text(encoding='utf8'))
    data['cases'][1]['query'] = data['cases'][0]['query']
    path = tmp_path / 'duplicate.json'
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf8')
    with pytest.raises(ValueError, match='duplicate'):
        load_frozen_cases(path)
    data['cases'][1]['query'] = '这是一个新问题'
    data['cases'][1]['unexpected'] = True
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf8')
    with pytest.raises(ValueError, match='schema'):
        load_frozen_cases(path)


def test_new_cases_must_not_reuse_previous_evaluation_prompts(tmp_path):
    data = json.loads(CASES.read_text(encoding='utf8'))
    data['cases'][0]['query'] = '谢谢你的帮助。'
    path = tmp_path / 'reused.json'
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf8')
    with pytest.raises(ValueError, match='previous'):
        load_frozen_cases(path)


def test_candidate_requires_better_correct_coverage_and_no_new_wrong_route():
    rows = [
        {'id': 'good', 'expected_intent': '投诉', 'choice': '投诉',
         'confidence': .99, 'reason': 'accepted'},
        {'id': 'unsafe', 'expected_intent': '其他', 'choice': '退款退货',
         'confidence': .98, 'reason': 'accepted'},
    ]
    baseline = assess_thresholds(rows, {'投诉': 1.0, '退款退货': 1.0})
    safer = assess_thresholds(rows, {'投诉': .99, '退款退货': 1.0})
    unsafe = assess_thresholds(rows, {'投诉': .99, '退款退货': .98})
    assert candidate_is_safer(baseline, safer) is True
    assert candidate_is_safer(baseline, unsafe) is False
    assert safer['correct'] == 2 and safer['wrong_routed_ids'] == []


def test_attempt_ledger_never_retries_or_exceeds_approved_cap(tmp_path):
    ledger = AttemptLedger(tmp_path / 'attempts.jsonl', limit=2)
    ledger.reserve('first', 'a' * 64, 'jev-test')
    with pytest.raises(ValueError, match='already reserved'):
        ledger.reserve('first', 'a' * 64, 'jev-test')
    ledger.reserve('second', 'a' * 64, 'jev-test')
    with pytest.raises(ValueError, match='exhausted'):
        ledger.reserve('third', 'a' * 64, 'jev-test')
    assert AttemptLedger(ledger.path, limit=2).used == 2


def test_cli_dry_run_freezes_hash_without_provider_or_credentials():
    result = subprocess.run([sys.executable, 'scripts/recalibrate_ch06_intent.py',
        '--split', 'calibration', '--dry-run'], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['requests'] == 110
    assert len(report['dataset_sha256']) == 64
    assert report['mode'] == 'dry_run'
