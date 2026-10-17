"""Labeled evaluation with disjoint splits and observed, per-class confidence cuts."""
import asyncio,hashlib,json
from collections import Counter
from pathlib import Path
from app.ch06_types import INTENTS
from app.intent_criteria import criteria_fingerprint
from app.ch06_prompts import resolver_fingerprint
from app.query_resolver import QueryResolutionError

ROOT=Path(__file__).resolve().parents[2]

def load_cases():
    return sum((json.loads((ROOT/'eval/ch06'/name).read_text(encoding='utf-8')) for name in ('intent-cases.json','dialogue-cases.json')),[])

def fingerprints(settings):
    files=('intent-cases.json','dialogue-cases.json')
    digest=hashlib.sha256(b''.join((ROOT/'eval/ch06'/name).read_bytes() for name in files)).hexdigest()
    return {'model':settings.jev_model,'criteria':criteria_fingerprint(),'resolver':resolver_fingerprint(),
            'chat_model':settings.chat_model,'cases':digest}

def calibrate_intents(results: list[dict],fingerprints: dict) -> dict:
    if not results or any(r['split']!='calibration' for r in results):
        raise ValueError('Only nonempty calibration observations may select thresholds')
    thresholds={}; count=0
    for label in INTENTS:
        if label=='其他': continue
        rows=[r for r in results if r['intent']==label and r['reason']=='accepted']
        valid=[]
        for cut in sorted({r['confidence'] for r in rows}):
            accepted=[r for r in rows if r['confidence']>=cut]
            if accepted and all(r['query_ok'] and r['expected_intent']==label for r in accepted):
                valid.append((len(accepted),cut))
        if valid:
            coverage,cut=max(valid)
            thresholds[label]=cut; count+=coverage
    return {'version':1,'fingerprints':fingerprints,'thresholds':thresholds,
            'automatic_correct':count,'total':len(results),'disabled':[x for x in INTENTS if x!='其他' and x not in thresholds]}

def query_correct(case,resolution):
    if case['unchanged'] and resolution.query!=case['raw']: return False
    if any(term not in resolution.query for term in case['required_terms']): return False
    if 'unresolved' in case and set(resolution.unresolved_references)!=set(case['unresolved']): return False
    return True

async def evaluate_intents(cases,resolver,classifier,policy,output: Path):
    semaphore=asyncio.Semaphore(4)
    async def evaluate(case):
        async with semaphore:
            row={k:case[k] for k in ('id','split','raw','expected_intent')}
            try:
                resolution,usage=await resolver.resolve(case['raw'],case['history'],case['confirmed'])
            except QueryResolutionError as error:
                return {**row,'query_ok':False,'intent':'其他','confidence':0.0,'reason':error.reason,'usage':error.usage.to_dict()}
            outcome=await classifier.classify(resolution.query)
            if policy is not None: outcome=policy.apply(outcome)
            return {**row,'query':resolution.query,'query_ok':query_correct(case,resolution),
                    'resolution':resolution.model_dump(),**outcome.decision.model_dump(),'reason':outcome.reason,
                    'model':outcome.model,'probabilities':outcome.probabilities,
                    'usage':{'input_tokens':usage.input_tokens+outcome.usage.input_tokens,
                             'output_tokens':usage.output_tokens+outcome.usage.output_tokens,
                             'estimated':usage.estimated+outcome.usage.estimated}}
    results=await asyncio.gather(*(evaluate(case) for case in cases))
    correct=sum(r['query_ok'] and r['intent']==r['expected_intent'] and r['reason'] in ('accepted','valid_other') for r in results)
    matrix=Counter((r['expected_intent'],r['intent']) for r in results)
    report={'total':len(results),'correct':correct,'query_correct':sum(r['query_ok'] for r in results),
            'confusion':[{'expected':a,'actual':b,'count':n} for (a,b),n in sorted(matrix.items())],'cases':results}
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report
