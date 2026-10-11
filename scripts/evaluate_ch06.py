"""Explicit paid chapter-6 evaluation with persistent attempt budgets and no retries."""
import argparse,asyncio,json,os,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dotenv import dotenv_values
from app.config import Settings
from app.model_service import ModelService
from app.jev_classifier import JevIntentClassifier
from app.query_resolver import QueryResolver
from app.intent_policy import IntentPolicy
from app.evaluation.ch06 import load_cases,fingerprints,evaluate_intents,calibrate_intents

class Budget:
    def __init__(self,run):
        self.run=run; self.path=ROOT/'eval/ch06/request-attempts.jsonl'
        self.counts={'jev':1 if (ROOT/'eval/ch06/protocol-attempt.json').exists() else 0,'chat':0}
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                record=json.loads(line); self.counts[record['provider']]+=1
    def reserve(self,provider):
        limit={'jev':367,'chat':618}[provider]
        if self.counts[provider]>=limit: raise RuntimeError('Approved request budget exhausted')
        record={'run':self.run,'provider':provider,'at':datetime.now(timezone.utc).isoformat()}
        with self.path.open('a',encoding='utf-8') as stream: stream.write(json.dumps(record)+'\n')
        self.counts[provider]+=1
        if self.counts[provider]%10==0: print(provider,'attempts:',self.counts[provider],flush=True)

async def run(args):
    cases=[c for c in load_cases() if c['split']==args.split]
    assert len(cases)==110 and len({c['id'] for c in cases})==110
    if not args.live:
        print(json.dumps({'mode':'dry_run','suite':args.suite,'split':args.split,'cases':len(cases),'max_chat_requests':110,'max_jev_requests':110})); return 0
    if args.env_file:
        for name,value in dotenv_values(args.env_file).items():
            if value is not None: os.environ.setdefault(name,value)
    settings=Settings.from_env()
    if not settings.jev_api_key or not settings.jev_model: raise SystemExit('Jev server configuration required')
    model=ModelService(settings); provider=JevIntentClassifier(settings)
    expected=fingerprints(settings)
    run_id=f'intent-{args.split}-r{args.round}'
    if args.split=='acceptance' and args.round!=1: raise SystemExit('Acceptance may run only once')
    if args.split=='calibration' and args.round not in (1,2): raise SystemExit('At most one calibration revision')
    policy=None if args.split=='calibration' else IntentPolicy.from_artifact(ROOT/'eval/ch06/intent-policy.json',expected)
    if policy is not None and policy.reason: raise SystemExit('Freeze a valid calibration before acceptance')
    attempt=ROOT/f'eval/ch06/{run_id}.attempt.json'
    with attempt.open('x',encoding='utf-8') as stream: json.dump({'run':run_id,'fingerprints':expected,'cases':len(cases)},stream,indent=2)
    budget=Budget(run_id)
    class CountedModel:
        async def resolve_query(self,messages):
            budget.reserve('chat'); return await model.resolve_query(messages)
    class CountedJev:
        async def classify(self,query):
            budget.reserve('jev'); return await provider.classify(query)
    output=Path(args.output) if args.output else ROOT/f'eval/ch06/{run_id}.json'
    report=await evaluate_intents(cases,QueryResolver(CountedModel()),CountedJev(),policy,output)
    report['fingerprints']=expected; report['cumulative_attempts']=budget.counts
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.split=='calibration':
        artifact=calibrate_intents(report['cases'],expected)
        (ROOT/'eval/ch06/intent-policy.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in artifact.items() if k!='fingerprints'},ensure_ascii=True))
    print(json.dumps({'total':report['total'],'correct':report['correct'],'query_correct':report['query_correct'],'cumulative_attempts':budget.counts}))
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',choices=['intent'],default='intent')
    parser.add_argument('--split',choices=['calibration','acceptance'],default='calibration')
    parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.live and args.dry_run: parser.error('Choose live or dry-run')
    raise SystemExit(asyncio.run(run(args)))
