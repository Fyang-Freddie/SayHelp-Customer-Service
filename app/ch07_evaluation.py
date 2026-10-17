"""Durable admission before paid evaluation requests, including failed attempts."""
import json
import sqlite3
from pathlib import Path
from uuid import uuid4
from contextlib import contextmanager, aclosing
from contextvars import ContextVar
import os

_bucket=ContextVar('ch07_evaluation_bucket',default=None)


@contextmanager
def evaluation_scope(bucket):
    token=_bucket.set(bucket)
    try:yield
    finally:_bucket.reset(token)


def configured_meter():
    path=os.environ.get('CH07_EVAL_USAGE_PATH')
    return EvaluationBudget(path) if path else None


@contextmanager
def evaluation_call(provider='chat',meter=None):
    meter=meter or configured_meter()
    usage={}
    if meter is None:
        yield usage
        return
    bucket=_bucket.get() or os.environ.get('CH07_EVAL_BUCKET','dialogue')
    if provider=='jev':bucket='jev_'+bucket
    identity=meter.reserve(provider,bucket)
    try:
        yield usage
    except BaseException:
        meter.finish(identity,'failed',usage)
        raise
    else:meter.finish(identity,'success',usage)


class MeteredModel:
    """Meter each provider request, preserving binding and streaming semantics."""
    def __init__(self,model,meter):self.model,self.meter=model,meter

    def bind_tools(self,*args,**kwargs):
        return MeteredModel(self.model.bind_tools(*args,**kwargs),self.meter)

    def with_structured_output(self,*args,**kwargs):
        return MeteredModel(self.model.with_structured_output(*args,**kwargs),self.meter)

    async def ainvoke(self,*args,**kwargs):
        with evaluation_call(meter=self.meter) as usage:
            result=await self.model.ainvoke(*args,**kwargs)
            raw=result.get('raw') if isinstance(result,dict) else result
            usage.update(getattr(raw,'usage_metadata',None) or {})
            return result

    async def astream(self,*args,**kwargs):
        with evaluation_call(meter=self.meter) as usage:
            async with aclosing(self.model.astream(*args,**kwargs)) as stream:
                async for chunk in stream:
                    usage.update(getattr(chunk,'usage_metadata',None) or {})
                    yield chunk


class EvaluationLimitExceeded(RuntimeError):
    pass


class EvaluationBudget:
    def __init__(self, path='.runtime/ch07/eval-usage.sqlite', *, limits=None, buckets=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The user approved 23 additional Jev dialogue calls after the
        # original 47/48 ledger was exhausted: 47 existing + 24 new = 71.
        self.limits = limits or {'chat':600, 'jev':71}
        self.buckets = buckets or {'dialogue':384, 'summary_eval':24, 'calibration':16,
            'background':48, 'browser':64, 'retry':64, 'jev_dialogue':48,
            'jev_browser':8, 'jev_retry':8}
        if buckets is None:
            self.buckets['jev_dialogue']=71
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

    def remaining(self,provider:str,bucket:str)->int:
        with self._connect() as conn:
            total=conn.execute('SELECT count(*) FROM calls WHERE provider=?',(provider,)).fetchone()[0]
            used=conn.execute('SELECT count(*) FROM calls WHERE bucket=?',(bucket,)).fetchone()[0]
        return max(0,min(self.limits[provider]-total,self.buckets[bucket]-used))

    def finish(self, call_id: str, outcome: str, usage: dict) -> None:
        with self._connect() as conn:
            changed = conn.execute('UPDATE calls SET outcome=?,usage=? WHERE id=?',
                (outcome,json.dumps(usage,allow_nan=False),call_id)).rowcount
            if not changed:
                raise KeyError('Unknown evaluation call')
