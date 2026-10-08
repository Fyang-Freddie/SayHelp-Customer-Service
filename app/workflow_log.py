"""Local chapter 5 events; raw weak questions never enter the flywheel database."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock

_SAFE_FIELDS = frozenset({'stop_reason', 'score', 'threshold', 'error_type', 'evidence_count',
    'intent', 'status', 'model_calls', 'tool_calls', 'conversation_id', 'turn_id',
    'route', 'node', 'name', 'tool_call_id', 'duration_ms', 'usage', 'phase',
    'attempt', 'attempts', 'reason', 'sufficient', 'evidence_ids', 'tools_bound'})

_context = ContextVar('workflow_log_context', default={})


@contextmanager
def log_context(state, node):
    token = _context.set({key: state.get(key) for key in ('conversation_id', 'turn_id', 'intent', 'route')} | {'node': node})
    try:
        yield
    finally:
        _context.reset(token)



class WorkflowLog:
    def __init__(self, directory='.runtime/ch05'):
        self.directory = Path(directory)
        self._lock = Lock()

    def _append(self, filename, row):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / filename).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    def write(self, event: str, payload: dict) -> None:
        if event not in {'weak_evidence', 'retrieval_error', 'knowledge_ready', 'turn_finished',
            'node_start', 'node_end', 'node_error', 'gate_decision', 'model_start', 'model_end',
            'model_error', 'tool_start', 'tool_attempt', 'tool_end', 'tool_result_rejected', 'turn_terminal'}:
            raise ValueError('Unknown workflow knowledge event')
        safe = {key: value for key, value in {**_context.get(), **payload}.items() if key in _SAFE_FIELDS}
        if isinstance(safe.get('usage'), dict):
            safe['usage'] = {k:v for k,v in safe['usage'].items() if k in {'input_tokens','output_tokens','estimated'}}
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            # Write the original question first. Any error propagates: callers must
            # never advertise successful recording when either append failed.
            if event == 'weak_evidence':
                self._append('low-confidence.jsonl', {'at': timestamp,
                    'question': payload['question'], **safe})
            self._append('events.jsonl', {'at': timestamp, 'event': event, 'payload': safe})
