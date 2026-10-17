"""One-shot Jev intent recalibration with a durable 220-attempt ceiling."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values
from app.config import Settings
from app.evaluation.ch06 import calibrate_intents, fingerprints
from app.evaluation.intent_recalibration import (
    AttemptLedger, assess_thresholds, candidate_is_safer, load_frozen_cases,
)
from app.jev_classifier import JevIntentClassifier

CASES = ROOT / 'eval/ch06/intent-recalibration-v1.json'
LEDGER = ROOT / '.runtime/ch06/intent-recalibration-attempts.jsonl'
CANDIDATE = ROOT / 'eval/ch06/intent-policy-candidate.json'
BASELINE = ROOT / 'eval/ch06/intent-policy.json'


def _fingerprints(settings, digest):
    return {**fingerprints(settings), 'recalibration_cases': digest}


async def run(args):
    cases, digest = load_frozen_cases(CASES)
    selected = [case for case in cases if case['split'] == args.split]
    if not args.live:
        print(json.dumps({'mode': 'dry_run', 'split': args.split, 'requests': len(selected),
                          'dataset_sha256': digest}, ensure_ascii=False))
        return 0
    if args.env_file:
        for name, value in dotenv_values(args.env_file).items():
            if value is not None:
                os.environ.setdefault(name, value)
    settings = Settings.from_env()
    if not settings.jev_api_key or not settings.jev_model:
        raise SystemExit('Jev credentials and fixed JEV_MODEL are required')
    ledger = AttemptLedger(LEDGER, limit=220)
    if ledger.used + len(selected) > ledger.limit:
        raise SystemExit('Approved Jev request budget exhausted before this split')
    if any(record['dataset_sha256'] != digest or record['model'] != settings.jev_model
           for record in ledger.records):
        raise SystemExit('Existing attempt ledger belongs to another dataset or model')
    if any(record['case_id'] in {case['id'] for case in selected} for record in ledger.records):
        raise SystemExit('This split has prior attempts; no automatic retry is permitted')
    if args.split == 'acceptance':
        if not CANDIDATE.is_file():
            raise SystemExit('Calibrate and freeze a candidate before independent acceptance')
        candidate = json.loads(CANDIDATE.read_text(encoding='utf8'))
        if candidate.get('fingerprints') != _fingerprints(settings, digest):
            raise SystemExit('Candidate model, criteria, or dataset fingerprint changed')
    output = ROOT / f'eval/ch06/intent-recalibration-{args.split}.jsonl'
    if output.exists():
        raise SystemExit('Split report already exists; refusing to overwrite')
    classifier = JevIntentClassifier(settings)
    rows = []
    for case in selected:
        ledger.reserve(case['id'], digest, settings.jev_model)
        outcome = await classifier.classify(case['query'])
        row = {**case, 'choice': outcome.decision.intent,
               'confidence': outcome.decision.confidence, 'reason': outcome.reason,
               'model': outcome.model, 'probabilities': outcome.probabilities,
               'usage': outcome.usage.to_dict()}
        with output.open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        rows.append(row)
        if len(rows) % 10 == 0:
            print(json.dumps({'completed': len(rows), 'split': args.split,
                              'ledger_used': ledger.used}, ensure_ascii=False), flush=True)
        if outcome.reason != 'accepted' or outcome.model != settings.jev_model:
            raise SystemExit(f'Provider or model failed on {case["id"]}; attempts preserved, no policy promoted')
    baseline = json.loads(BASELINE.read_text(encoding='utf8'))
    baseline_score = assess_thresholds(rows, baseline['thresholds'])
    if args.split == 'calibration':
        inputs = [{**row, 'intent': row['choice'], 'query_ok': True} for row in rows]
        candidate = calibrate_intents(inputs, _fingerprints(settings, digest))
        CANDIDATE.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    candidate_score = assess_thresholds(rows, candidate['thresholds'])
    summary = {'split': args.split, 'dataset_sha256': digest, 'model': settings.jev_model,
               'baseline': baseline_score, 'candidate': candidate_score,
               'candidate_is_safer': candidate_is_safer(baseline_score, candidate_score),
               'attempts_used': ledger.used, 'candidate_path': str(CANDIDATE)}
    report = ROOT / f'eval/ch06/intent-recalibration-{args.split}-report.json'
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['calibration', 'acceptance'], required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--live', action='store_true')
    parser.add_argument('--env-file', type=Path)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
