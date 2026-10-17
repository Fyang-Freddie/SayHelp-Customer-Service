"""Run the chapter 6 chat UI with server-only configuration and optional evaluation budgets."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dotenv import dotenv_values

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--env-file',type=Path,default=ROOT/'.env')
    parser.add_argument('--port',type=int,default=8096)
    parser.add_argument('--init-db',action='store_true')
    parser.add_argument('--count-evaluation',action='store_true')
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--case',choices=['context-switch','order-picker','refund-form'],default='order-picker')
    args=parser.parse_args(argv)
    if not args.live:
        print(json.dumps({'mode':'dry_run','case':args.case,'instructions':'Add --live to start the real browser demo server; no requests made.'}))
        return 0
    if not args.env_file.is_file():
        print('Server configuration file is missing',file=sys.stderr)
        return 2
    for name,value in dotenv_values(args.env_file).items():
        if value is not None:os.environ.setdefault(name,value)
    from app.config import Settings
    from app.model_service import ModelService
    from app.main import create_app
    try:
        settings=Settings.from_env()
        if not settings.jev_api_key or not settings.jev_model:raise ValueError()
    except (ValueError,KeyError):
        print('Server configuration requires chat provider, database and Jev key/model',file=sys.stderr)
        return 2
    if args.init_db:
        from app.init_ch05_db import initialize_ch05_database
        from app.init_ch06_db import initialize_ch06_database
        initialize_ch05_database(settings.database_url)
        initialize_ch06_database(settings.database_url)
    model=ModelService(settings)
    if args.count_evaluation:
        from scripts.evaluate_ch06 import Budget
        budget=Budget('browser-e2e')
        counts={'jev':0,'chat':0}
        for line in (budget.path.read_text(encoding='utf-8').splitlines() if budget.path.exists() else []):
            record=json.loads(line)
            if record['run']=='browser-e2e':counts[record['provider']]+=1
        def reserve(provider):
            if counts[provider]>=({'jev':36,'chat':216}[provider]):raise RuntimeError('End-to-end evaluation budget exhausted')
            budget.reserve(provider);counts[provider]+=1
        for name,provider in [('classify_intent','jev'),('resolve_query','chat'),('expand_query','chat'),('ask_customer','chat'),('judge_eligibility','chat'),('select','chat')]:
            original=getattr(model,name)
            def wrap(original,provider):
                async def call(*args,**kwargs):
                    reserve(provider);return await original(*args,**kwargs)
                return call
            setattr(model,name,wrap(original,provider))
        original_stream=model.stream_reply
        async def stream(*args,**kwargs):
            reserve('chat')
            async for chunk in original_stream(*args,**kwargs):yield chunk
        model.stream_reply=stream
    import uvicorn
    print(json.dumps({'mode':'live_browser_demo','case':args.case,'url':f'http://127.0.0.1:{args.port}/','automatic_refund_submission':False}),flush=True)
    uvicorn.run(create_app(settings,model),host='127.0.0.1',port=args.port)
    return 0

if __name__=='__main__':raise SystemExit(main())
