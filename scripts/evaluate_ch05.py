"""Real-provider chapter 5 intent evaluation.

From the repository root:
    python scripts/evaluate_ch05.py --suite intent --output eval/ch05/intent_results.json

Labels must be authored before running. Each case makes one classification request.
The first report is preserved beside --output as <stem>.initial.json; reruns update
--output. Reports contain labels, usage and error types, never provider reasoning.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Direct script execution resolves app imports without requiring PYTHONPATH.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.intent import IntentClassificationError, IntentDecision, classify_intent
from app.intent_prompts import INTENT_SYSTEM_PROMPT
from app.model_service import ModelService
from app.workflow_types import Usage


def validate_cases(cases: list[dict]) -> None:
    """Reject incomplete or invalid benchmark data before any paid model request."""
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError('Cases must be a list of labelled objects')
    ids = []
    counts = Counter()
    boundary = 0
    for case in cases:
        if not isinstance(case.get('id'), str) or not case['id']:
            raise ValueError('Each case needs an id')
        ids.append(case['id'])
        if not isinstance(case.get('text'), str) or not case['text'].strip():
            raise ValueError('Each case needs nonempty text')
        IntentDecision.model_validate({'intent': case.get('expected')})
        if case.get('split') == 'core':
            counts[case['expected']] += 1
        elif case.get('split') == 'boundary':
            boundary += 1
        else:
            raise ValueError('Each case must be core or boundary')
        if type(case.get('mandatory', False)) is not bool:
            raise ValueError('mandatory must be a boolean')
    if len(set(ids)) != len(ids):
        raise ValueError('Case ids must be unique')
    if len(counts) != 7 or any(count < 5 for count in counts.values()) or boundary != 7:
        raise ValueError('Need at least five core cases per intent and seven boundary cases')


class _CountedModel:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    async def classify_intent(self, text):
        self.calls += 1
        return await self.model.classify_intent(text)


async def evaluate_intent(cases: list[dict], *, model) -> dict:
    validate_cases(cases)
    rows = []
    total = Usage()
    for case in cases:
        counted = _CountedModel(model)
        usage = Usage()
        prediction, error_type = None, None
        try:
            decision, usage = await classify_intent(case['text'], model=counted)
            prediction = decision.intent
        except IntentClassificationError as error:
            usage = error.usage
            error_type = type(error).__name__
        except Exception as error:
            # Provider exception messages can contain payloads/URLs; record only type.
            error_type = type(error).__name__
        success = error_type is None
        rows.append({**case, 'predicted': prediction, 'model_calls': counted.calls,
                     'success': success, 'correct': success and counted.calls == 1 and prediction == case['expected'],
                     'error_type': error_type, 'usage': usage.to_dict()})
        total = Usage(total.input_tokens + usage.input_tokens,
                      total.output_tokens + usage.output_tokens, total.estimated + usage.estimated)
    core = [row for row in rows if row['split'] == 'core']
    boundary = [row for row in rows if row['split'] == 'boundary']
    mandatory = [row for row in core if row.get('mandatory', False)]
    minimum = len(core) - 2
    return {
        'suite': 'intent', 'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'core': {'correct': sum(row['correct'] for row in core), 'total': len(core), 'minimum_correct': minimum},
        'boundary': {'correct': sum(row['correct'] for row in boundary), 'total': len(boundary)},
        'mandatory': {'correct': sum(row['correct'] for row in mandatory), 'total': len(mandatory)},
        'accepted': sum(row['correct'] for row in core) >= minimum and all(row['correct'] for row in mandatory),
        'model_calls': sum(row['model_calls'] for row in rows), 'usage': total.to_dict(), 'cases': rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['intent'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case_path = ROOT / 'eval/ch05/intent_cases.json'
    source = case_path.read_bytes()
    cases = json.loads(source.decode('utf-8-sig'))
    validate_cases(cases)
    settings = Settings.from_env()
    report = asyncio.run(evaluate_intent(cases, model=ModelService(settings)))
    report.update({'provider_model': settings.chat_model,
                   'prompt_sha256': hashlib.sha256(INTENT_SYSTEM_PROMPT.encode('utf-8')).hexdigest(),
                   'cases_sha256': hashlib.sha256(source).hexdigest(), 'provider': 'configured_real_provider'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    initial = args.output.with_name(args.output.stem + '.initial' + args.output.suffix)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if not initial.exists():
        initial.write_text(serialized, encoding='utf-8')
    args.output.write_text(serialized, encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('core', 'boundary', 'mandatory', 'accepted', 'model_calls', 'usage')},
                     ensure_ascii=False))
    print(f'Report: {args.output}; first run preserved: {initial}')
    return 0 if report['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
