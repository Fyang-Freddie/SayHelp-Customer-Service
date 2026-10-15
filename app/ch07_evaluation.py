"""Durable admission before paid evaluation requests, including failed attempts."""
import json
import sqlite3
from pathlib import Path
from uuid import uuid4


class EvaluationLimitExceeded(RuntimeError):
    pass


class EvaluationBudget:
    def __init__(self, path='.runtime/ch07/eval-usage.sqlite', *, limits=None, buckets=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = limits or {'chat':600, 'jev':64}
        self.buckets = buckets or {'dialogue':384, 'summary_eval':24, 'calibration':16,
            'background':48, 'browser':64, 'retry':64, 'jev_dialogue':48,
            'jev_browser':8, 'jev_retry':8}
        with self._connect() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, provider TEXT NOT NULL, bucket TEXT NOT NULL, outcome TEXT NOT NULL, usage TEXT NOT NULL)')

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def reserve(self, provider: str, bucket: str) -> str:
        if provider not in self.limits or bucket not in self.buckets:
            raise ValueError('Unknown evaluation provider or bucket')
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            total = conn.execute('SELECT count(*) FROM calls WHERE provider=?',(provider,)).fetchone()[0]
            used = conn.execute('SELECT count(*) FROM calls WHERE bucket=?',(bucket,)).fetchone()[0]
            if total >= self.limits[provider] or used >= self.buckets[bucket]:
                raise EvaluationLimitExceeded('Evaluation request allowance exhausted')
            id = uuid4().hex
            conn.execute('INSERT INTO calls VALUES (?,?,?,?,?)',(id,provider,bucket,'reserved','{}'))
            return id

    def finish(self, call_id: str, outcome: str, usage: dict) -> None:
        with self._connect() as conn:
            changed = conn.execute('UPDATE calls SET outcome=?,usage=? WHERE id=?',
                (outcome,json.dumps(usage,allow_nan=False),call_id)).rowcount
            if not changed:
                raise KeyError('Unknown evaluation call')
