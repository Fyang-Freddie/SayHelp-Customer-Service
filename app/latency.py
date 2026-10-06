"""Request-local duration telemetry; never include prompts or provider errors."""
import json
import logging
from contextlib import contextmanager
from time import perf_counter

logger = logging.getLogger(__name__)


class TurnTiming:
    def __init__(self):
        self.started = perf_counter()
        self.timings = {}
        self.failed_stage = None
        self.first_token_ms = None
        self.candidates = None

    @contextmanager
    def measure(self, stage):
        started = perf_counter()
        try:
            yield
        except BaseException:
            self.failed_stage = stage
            raise
        finally:
            self.timings[stage] = self.timings.get(stage, 0) + (perf_counter() - started) * 1000

    def payload(self, outcome):
        return {'outcome': outcome, 'failed_stage': self.failed_stage,
                'first_token_ms': self.first_token_ms, 'candidates': self.candidates,
                'timings_ms': {**self.timings, 'total': (perf_counter() - self.started) * 1000}}

    def log(self, outcome):
        logger.info(json.dumps(self.payload(outcome), allow_nan=False))
