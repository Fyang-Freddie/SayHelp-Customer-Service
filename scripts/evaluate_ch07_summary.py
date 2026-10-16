"""Evaluate frozen synthetic facts; never rewrite labels from model output."""
import argparse,asyncio,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import HumanMessage,AIMessage
from app.context_types import TurnRecord
from app.context_summary import validate_summary


async def run(args):
    cases=json.loads(Path(args.cases).read_text(encoding='utf8'))
    if not args.live:
        print(json.dumps({'cases':len(cases),'paid_calls':0,'mode':'dry-run'}));return
    if not 1<=args.max_calls<=24:raise ValueError('Maximum is 24 evaluation calls')
    from dotenv import load_dotenv
    if args.env_file:load_dotenv(args.env_file,override=False)
    os.environ.setdefault('MODEL_CONTEXT_WINDOW','18000')
    os.environ.setdefault('MAX_OUTPUT_TOKENS','2000')
    os.environ['CH07_EVAL_USAGE_PATH']=str(Path('.runtime/ch07/eval-usage.sqlite').resolve())
    from app.ch07_evaluation import evaluation_scope
    from app.config import Settings
    from app.model_service import ModelService
    model=ModelService(Settings.from_env())
    results=[]
    for case in cases[:args.max_calls]:
        turns=(TurnRecord(case['id'],1,2,(HumanMessage(case['user']),AIMessage(case['assistant']))),)
        row={'id':case['id'],'passed':False}
        try:
            with evaluation_scope('summary_eval'):
                output=await model.summarize(background=case.get('background',''),turns=turns)
            row['output']=output
            valid=validate_summary(output,turns)
            row['missing']=[group for group in case['required'] if not any(term in valid for term in group)]
            row['forbidden']=[term for term in case.get('forbidden',[]) if term in valid]
            row['passed']=not row['missing'] and not row['forbidden'] and (not case.get('empty') or valid=='')
        except Exception as error:row['error_type']=type(error).__name__
        results.append(row)
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'cases_file':args.cases,'results':results},ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps({'id':row['id'],'passed':row['passed']},ensure_ascii=False),flush=True)
    print(json.dumps({'passed':sum(r['passed'] for r in results),'total':len(results),'output':args.output}))
    if not all(r['passed'] for r in results):raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--max-calls',type=int,default=24)
    parser.add_argument('--cases',default='eval/ch07/summary-cases.json')
    parser.add_argument('--env-file')
    parser.add_argument('--output',default='.runtime/ch07/summary-results.json')
    asyncio.run(run(parser.parse_args()))
