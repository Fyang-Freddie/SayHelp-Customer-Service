"""One explicit live protocol probe. The ledger prevents accidental repeat charges."""
import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dotenv import dotenv_values
from app.config import Settings, JEV_DISCOVERY_MODEL
from app.jev_classifier import JevIntentClassifier


async def run(args):
    if not args.live:
        print(json.dumps({'mode':'dry_run','decision_requests':1,'model_source':'JEV_MODEL or discovery config'}))
        return 0
    if args.env_file:
        for name,value in dotenv_values(args.env_file).items():
            if value is not None: os.environ.setdefault(name,value)
    settings=Settings.from_env()
    settings=replace(settings,jev_model=settings.jev_model or JEV_DISCOVERY_MODEL)
    if not settings.jev_api_key: raise SystemExit('TYPESAFE_API_KEY is required')
    ledger=ROOT/'eval/ch06/protocol-attempt.json'
    attempt={'started_at':datetime.now(timezone.utc).isoformat(),'query':'我想申请退货退款。','expected_intent':'退款退货','requested_model':settings.jev_model}
    try:
        with ledger.open('x',encoding='utf-8') as f: json.dump(attempt,f,ensure_ascii=False,indent=2)
    except FileExistsError: raise SystemExit('Protocol attempt already recorded; refusing an unbudgeted retry')
    result=await JevIntentClassifier(settings).classify(attempt['query'])
    report={**attempt,'reason':result.reason,'actual_model':result.model,'decision':result.decision.model_dump(),'probabilities':result.probabilities,'usage':result.usage.to_dict(),'passed':result.reason=='accepted' and result.decision.intent=='退款退货'}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--output',default=str(ROOT/'eval/ch06/protocol-check.json'))
    raise SystemExit(asyncio.run(run(parser.parse_args())))
