"""Frozen, auditable Jev-only intent recalibration data and safety checks."""
from collections import Counter
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path

from app.ch06_types import INTENTS
from app.evaluation.ch06 import load_cases


def load_frozen_cases(path: Path) -> tuple[list[dict], str]:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {'version', 'description', 'cases'} or data['version'] != 1:
        raise ValueError('invalid frozen case schema')
    cases = data['cases']
    if not isinstance(cases, list) or len(cases) != 220:
        raise ValueError('frozen cases must contain exactly 220 entries')
    ids, queries = set(), set()
    previous_queries = {case['raw'] for case in load_cases()}
    counts = Counter()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {'id', 'split', 'query', 'expected_intent'}:
            raise ValueError('invalid frozen case schema')
        id, split, query, label = (case[k] for k in ('id', 'split', 'query', 'expected_intent'))
        if not all(isinstance(v, str) and v.strip() for v in (id, split, query, label)):
            raise ValueError('invalid frozen case schema')
        if id in ids or query in queries:
            raise ValueError('duplicate frozen case id or query')
        if query in previous_queries:
            raise ValueError('frozen case repeats previous evaluation prompt')
        if split not in ('calibration', 'acceptance') or label not in INTENTS:
            raise ValueError('invalid frozen case schema')
        ids.add(id)
        queries.add(query)
        counts[split] += 1
    if counts != {'calibration': 110, 'acceptance': 110}:
        raise ValueError('frozen cases require 110 per split')
    return cases, hashlib.sha256(raw).hexdigest()


def assess_thresholds(rows: list[dict], thresholds: dict[str, float]) -> dict:
    correct = 0
    fallback = 0
    wrong_routed_ids = []
    by_label = Counter()
    for row in rows:
        choice = row['choice']
        predicted = (choice if row['reason'] == 'accepted' and choice != '其他'
                     and row['confidence'] >= thresholds.get(choice, float('inf')) else '其他')
        if predicted == '其他':
            fallback += 1
        if predicted == row['expected_intent']:
            correct += 1
            by_label[predicted] += 1
        elif predicted != '其他':
            wrong_routed_ids.append(row['id'])
    return {'total': len(rows), 'correct': correct, 'fallback': fallback,
            'wrong_routed_ids': wrong_routed_ids, 'correct_by_label': dict(by_label)}


def candidate_is_safer(baseline: dict, candidate: dict) -> bool:
    return (baseline['total'] == candidate['total']
            and candidate['correct'] > baseline['correct']
            and set(candidate['wrong_routed_ids']) <= set(baseline['wrong_routed_ids']))


class AttemptLedger:
    """Reserve before every paid attempt; a reservation is never retried."""

    def __init__(self, path: Path, *, limit: int = 220):
        self.path, self.limit = Path(path), limit
        self.records = []
        if self.path.exists():
            self.records = [json.loads(line) for line in self.path.read_text(encoding='utf8').splitlines() if line]
        if len(self.records) > limit or len({r['case_id'] for r in self.records}) != len(self.records):
            raise ValueError('invalid attempt ledger')

    @property
    def used(self) -> int:
        return len(self.records)

    def reserve(self, case_id: str, digest: str, model: str) -> None:
        if any(r['case_id'] == case_id for r in self.records):
            raise ValueError('case already reserved')
        if self.used >= self.limit:
            raise ValueError('approved Jev request budget exhausted')
        record = {'at': datetime.now(UTC).isoformat(), 'case_id': case_id,
                  'dataset_sha256': digest, 'model': model}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
        self.records.append(record)
