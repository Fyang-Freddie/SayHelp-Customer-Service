"""Local chapter 5 events; raw weak questions never enter the flywheel database."""
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock

_SAFE_FIELDS = frozenset({'stop_reason', 'score', 'threshold', 'error_type', 'evidence_count',
    'intent', 'status', 'model_calls', 'tool_calls'})


class WorkflowLog:
    def __init__(self, directory='.runtime/ch05'):
        self.directory = Path(directory)
        self._lock = Lock()

    def _append(self, filename, row):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / filename).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    def write(self, event: str, payload: dict) -> None:
        if event not in {'weak_evidence', 'retrieval_error', 'knowledge_ready', 'turn_finished'}:
            raise ValueError('Unknown workflow knowledge event')
        safe = {key: value for key, value in payload.items() if key in _SAFE_FIELDS}
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            # Write the original question first. Any error propagates: callers must
            # never advertise successful recording when either append failed.
            if event == 'weak_evidence':
                self._append('low-confidence.jsonl', {'at': timestamp,
                    'question': payload['question'], **safe})
            self._append('events.jsonl', {'at': timestamp, 'event': event, 'payload': safe})
