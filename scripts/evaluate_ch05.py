"""Real-provider intent and read-only retrieval evaluation for chapter 5.

From the repository root:
    python scripts/evaluate_ch05.py --suite intent --output eval/ch05/intent_results.json

Labels must be authored before running. Intent uses one classification request per case.
Knowledge uses separate calibration/acceptance splits and no query or answer LLM.
Use --suite knowledge --calibrate once to fit the raw-query confidence artifact.
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



def validate_knowledge_cases(cases: list[dict]) -> None:
    from app.knowledge_types import KnowledgeFilters
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError('Knowledge cases must be labelled objects')
    ids, questions, counts = [], [], Counter()
    for case in cases:
        if not isinstance(case.get('id'), str) or not case['id']:
            raise ValueError('Knowledge case id required')
        if not isinstance(case.get('question'), str) or not case['question'].strip():
            raise ValueError('Knowledge question required')
        if case.get('split') not in {'calibration', 'acceptance', 'confirmatory'} or type(case.get('answerable')) is not bool:
            raise ValueError('Separate calibration/acceptance and boolean labels required')
        evidence = case.get('evidence', {})
        if case['answerable'] and (not isinstance(evidence, dict) or not all(
            isinstance(evidence.get(key), str) and evidence[key] for key in ('source_file', 'section_path', 'quote'))):
            raise ValueError('Answerable cases require independently labelled evidence')
        if not case['answerable'] and not case.get('missing'):
            raise ValueError('Unanswerable cases require a missing-evidence reason')
        KnowledgeFilters.model_validate(case.get('filters', {}))
        ids.append(case['id'])
        questions.append(case['question'])
        counts[(case['split'], case['answerable'])] += 1
    if len(set(ids)) != len(ids) or len(set(questions)) != len(questions):
        raise ValueError('Case ids and questions must be independent and unique')
    splits = ('calibration', 'acceptance', 'confirmatory') if any(c['split'] == 'confirmatory' for c in cases) else ('calibration', 'acceptance')
    if any(counts[(split, label)] < 6 for split in splits for label in (True, False)):
        raise ValueError('Need at least six answerable and six unanswerable cases per split')


def _has_gold(citations, case):
    gold = case['evidence']
    return any(c['source_file'] == gold['source_file'] and c['section_path'] == gold['section_path']
               and gold['quote'] in c['answer'] for c in citations)


async def evaluate_knowledge(cases: list[dict], *, gate, split='acceptance') -> dict:
    """Evaluate held-out cases only. Any false acceptance or service error fails."""
    from app.knowledge_types import KnowledgeFilters
    validate_knowledge_cases(cases)
    rows = []
    for case in cases:
        if case['split'] != split:
            continue
        outcome = await gate.prepare(case['question'], KnowledgeFilters.model_validate(case.get('filters', {})))
        passed = outcome['stop_reason'] is None and outcome['confidence']['sufficient']
        gold_present = _has_gold(outcome['citations'], case) if case['answerable'] else None
        rows.append({**case, 'passed': passed, 'gold_evidence_present': gold_present,
            'confidence': outcome['confidence'], 'stop_reason': outcome['stop_reason'],
            'citation_ids': [c['chunk_id'] for c in outcome['citations']],
            'low_confidence_recorded': outcome['low_confidence_recorded'],
            'logging_error': outcome.get('logging_error', False)})
    known = [r for r in rows if r['answerable']]
    unknown = [r for r in rows if not r['answerable']]
    false_accepts = sum(r['passed'] for r in unknown)
    false_rejects = sum(not r['passed'] for r in known)
    missing_gold = sum(r['passed'] and not r['gold_evidence_present'] for r in known)
    errors = sum(r['stop_reason'] not in (None, 'weak_evidence') or r['logging_error'] for r in rows)
    accepted_known = len(known) - false_rejects
    summary = {'answerable_total': len(known), 'unanswerable_total': len(unknown),
        'answerable_passed': accepted_known, 'false_accepts': false_accepts,
        'false_rejects': false_rejects, 'passed_without_gold_evidence': missing_gold,
        'service_or_logging_errors': errors,
        'refusal_rate': sum(not r['passed'] for r in rows) / len(rows),
        'answerable_refusal_rate': false_rejects / len(known),
        'unanswerable_refusal_rate': (len(unknown)-false_accepts) / len(unknown)}
    return {'suite': 'knowledge', 'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'evaluation_split': split, 'acceptance': summary, 'criteria': {'maximum_false_accepts': 0,
            'minimum_answerable_pass_rate': .8, 'maximum_missing_gold': 0, 'maximum_service_errors': 0},
        'accepted': false_accepts == 0 and accepted_known / len(known) >= .8 and missing_gold == 0 and errors == 0,
        'model_calls': 0, 'usage': Usage().to_dict(), 'cases': rows}


async def calibrate_knowledge(cases, *, retrieval, settings, cases_path, path):
    """Use calibration split alone; never read acceptance retrieval scores."""
    from app.evaluation.calibration import choose_threshold
    from app.knowledge_types import KnowledgeFilters, PreparedQuery
    from app.workflow_knowledge import knowledge_fingerprints, ConfidencePolicy, CALIBRATION_METHOD
    samples = []
    for case in cases:
        if case['split'] != 'calibration':
            continue
        query = PreparedQuery(case['question'], case['question'], case['question'])
        result = await asyncio.to_thread(retrieval.retrieve, query, 'hybrid_rerank',
            KnowledgeFilters.model_validate(case.get('filters', {})))
        if not result.ranked:
            raise ValueError('Calibration requires observed finite top scores')
        samples.append({'id': case['id'], 'question': case['question'],
                        'score': result.ranked[0].score, 'answerable': case['answerable']})
    choice = choose_threshold([(s['score'], s['answerable']) for s in samples])
    highest_negative = max(s['score'] for s in samples if not s['answerable'])
    lowest_positive = min(s['score'] for s in samples if s['answerable'])
    if highest_negative >= lowest_positive:
        raise ValueError('Raw-query calibration labels are not separable; no threshold written')
    # Freeze a midpoint using calibration alone, leaving equal margin to both classes.
    choice['threshold'] = highest_negative + (lowest_positive - highest_negative) / 2
    choice['method'] = CALIBRATION_METHOD
    choice['highest_negative'] = highest_negative
    choice['lowest_positive'] = lowest_positive
    expected = knowledge_fingerprints(retrieval.corpus, settings, cases_path)
    artifact = {'version': 1, 'fingerprints': expected, 'strategy': 'hybrid_rerank',
                'strategies': {'hybrid_rerank': {**choice, 'samples': samples}},
                'calibrated_at': datetime.now(timezone.utc).isoformat()}
    # Preserve the first calibration as evidence, just like evaluation reports.
    initial = path.with_name(path.stem + '.initial' + path.suffix)
    serialized = json.dumps(artifact, ensure_ascii=False, indent=2) + '\n'
    if not initial.exists(): initial.write_text(serialized, encoding='utf-8')
    path.write_text(serialized, encoding='utf-8')
    return ConfidencePolicy({'hybrid_rerank': choice['threshold']}, expected)


async def run_live_knowledge(cases, settings, cases_path, *, calibrate=False, split='acceptance'):
    from app.workflow_knowledge import (create_live_retrieval, knowledge_fingerprints,
        load_knowledge_policy, KnowledgeGate)
    validate_knowledge_cases(cases)
    with create_live_retrieval(settings) as retrieval:
        # Verify authored evidence against the existing authoritative source
        # before inference, so typos in labels cannot hide retrieval mistakes.
        chunks = list(retrieval.corpus.chunks.values())
        for case in cases:
            if case['answerable'] and not _has_gold([{'source_file': c.source_file,
                'section_path': c.section_path, 'answer': c.answer} for c in chunks], case):
                raise ValueError('Gold evidence does not match current corpus: ' + case['id'])
        path = ROOT / 'eval/ch05/confidence.json'
        if calibrate:
            policy = await calibrate_knowledge(cases, retrieval=retrieval, settings=settings,
                                               cases_path=cases_path, path=path)
        else:
            policy = load_knowledge_policy(path, knowledge_fingerprints(retrieval.corpus, settings, cases_path))
        report = await evaluate_knowledge(cases, gate=KnowledgeGate(retrieval, policy, settings), split=split)
        report.update({'provider': 'existing_read_only_knowledge_collection',
            'collection': settings.knowledge_collection, 'corpus_digest': retrieval.corpus.corpus_digest,
            'confidence_fingerprints': policy.fingerprints, 'threshold': policy.thresholds['hybrid_rerank']})
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['intent', 'knowledge'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibrate', action='store_true', help='Explicitly calibrate knowledge using calibration split alone')
    parser.add_argument('--confirmatory', action='store_true', help='Evaluate fresh confirmatory labels, preserving development acceptance results')
    args = parser.parse_args()
    if (args.calibrate or args.confirmatory) and args.suite != 'knowledge':
        parser.error('--calibrate/--confirmatory are only valid for knowledge')
    case_path = ROOT / ('eval/ch05/' + args.suite + '_cases.json')
    source = case_path.read_bytes()
    cases = json.loads(source.decode('utf-8-sig'))
    confirmation_source = None
    if args.confirmatory:
        confirmation_source = (ROOT / 'eval/ch05/knowledge_confirmation_cases.json').read_bytes()
        cases += json.loads(confirmation_source.decode('utf-8-sig'))
    (validate_cases if args.suite == 'intent' else validate_knowledge_cases)(cases)
    settings = Settings.from_env()
    if args.suite == 'intent':
        report = asyncio.run(evaluate_intent(cases, model=ModelService(settings)))
        report.update({'provider_model': settings.chat_model,
            'prompt_sha256': hashlib.sha256(INTENT_SYSTEM_PROMPT.encode('utf-8')).hexdigest(),
            'provider': 'configured_real_provider'})
        keys = ('core', 'boundary', 'mandatory', 'accepted', 'model_calls', 'usage')
    else:
        report = asyncio.run(run_live_knowledge(cases, settings, case_path, calibrate=args.calibrate,
            split='confirmatory' if args.confirmatory else 'acceptance'))
        keys = ('acceptance', 'accepted', 'model_calls', 'usage')
    report['cases_sha256'] = hashlib.sha256(source).hexdigest()
    if confirmation_source is not None:
        report['confirmation_cases_sha256'] = hashlib.sha256(confirmation_source).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    initial = args.output.with_name(args.output.stem + '.initial' + args.output.suffix)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if not initial.exists():
        initial.write_text(serialized, encoding='utf-8')
    args.output.write_text(serialized, encoding='utf-8')
    print(json.dumps({key: report[key] for key in keys}, ensure_ascii=False))
    print(f'Report: {args.output}; first run preserved: {initial}')
    return 0 if report['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
