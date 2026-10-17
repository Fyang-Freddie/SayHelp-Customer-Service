"""Run the same frozen 24 turns over HTTP with a persistent visitor cookie."""
import argparse,asyncio,json,os,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def profile_environment(profile):
    window=18000 if profile=='demo' else 65536
    return {'MODEL_CONTEXT_WINDOW':str(window),'MAX_OUTPUT_TOKENS':'2000',
        'MAX_USER_INPUT_TOKENS':'2000','MAX_AGENT_STEPS':'3','TOOL_RESULT_MAX_TOKENS':'1200',
        'RERANK_TOP_K':'5','AGENT_TURN_TOKEN_BUDGET':str(8*window)}


def events(text):
    event=None
    for line in text.splitlines():
        if line.startswith('event: '):event=line[7:]
        elif line.startswith('data: '):yield event,json.loads(line[6:])


def preflight(settings,meter,turns):
    from app.ch07_evaluation import EvaluationLimitExceeded
    from app.context_budget import settings_budget,estimate_tokens,validate_startup
    from app.prompts import render_workflow_system_prompt
    from app.agent_runtime import _schemas
    from app.tools import build_tools
    from langchain_core.messages import SystemMessage
    async def knowledge_answer(*args,**kwargs):raise AssertionError('preflight never executes tools')
    schemas=_schemas(build_tools(None,0,None,knowledge_answer=knowledge_answer))[0]
    validate_startup(settings_budget(settings,prefix=estimate_tokens(
        [SystemMessage(render_workflow_system_prompt())],schemas)),settings.history_steady_tokens)
    if meter.remaining('jev','jev_dialogue')<turns or meter.remaining('chat','dialogue')<turns*8:
        raise EvaluationLimitExceeded('Not enough remaining allowance for a full profile; no requests started')


async def run(args):
    environment=dict(os.environ)
    output=Path(args.output or f'eval/ch07/{args.profile}-acceptance.json')
    if args.live and not args.dry_run:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps({'profile':args.profile,'passed':False,'turns':[]}),encoding='utf8')
    try:
        return await run_profile(args)
    except Exception as error:
        if args.live and not args.dry_run:
            report=json.loads(output.read_text(encoding='utf8'))
            report.update(passed=False,error_type=type(error).__name__)
            output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        raise
    finally:
        os.environ.clear();os.environ.update(environment)


async def run_profile(args):
    fixture=Path('eval/ch07/dialogue-cases.json')
    cases=json.loads(fixture.read_text(encoding='utf-8-sig'))
    config=profile_environment(args.profile)
    if args.dry_run or not args.live:
        print(json.dumps({'profile':args.profile,'turns':len(cases['turns']),'paid_calls':0,'configuration':config}));return
    from dotenv import load_dotenv
    if not Path(args.env_file).is_file():raise FileNotFoundError('Demo environment file is missing')
    load_dotenv(args.env_file,override=False)
    os.environ.update(config)
    os.environ['CH07_EVAL_USAGE_PATH']=str(Path('.runtime/ch07/eval-usage.sqlite').resolve())
    os.environ['WORKFLOW_CHECKPOINT_PATH']=str(Path(f'.runtime/ch07/{args.profile}-checkpoints.sqlite').resolve())
    from app.config import Settings
    settings=Settings.from_env()
    from app.ch07_evaluation import configured_meter
    preflight(settings,configured_meter(),len(cases['turns']))
    if args.init_db:
        from app.init_ch07_db import initialize_ch07_database
        initialize_ch07_database(settings.database_url)
    from app.main import create_app
    from app.ch07_evaluation import evaluation_scope
    import httpx
    output=Path(args.output or f'eval/ch07/{args.profile}-acceptance.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'profile':args.profile,'configuration':config,'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
        'turns':[],'passed':False}
    def save():output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    app=create_app(settings)
    async with app.router.lifespan_context(app):
        service=app.state.workflow_service
        runtime=service.context_runtime
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://ch07.local',timeout=180) as client:
            cid=None
            for index,text in enumerate(cases['turns'],1):
                started=time.perf_counter()
                with evaluation_scope('dialogue'):
                    response=await client.post('/v1/chat/stream',json={'message':text,'conversation_id':cid})
                response.raise_for_status()
                parsed=list(events(response.text))
                cid=next((data['conversation_id'] for event,data in parsed if event=='session'),cid)
                row={'round':index,'question':text,'seconds':round(time.perf_counter()-started,3),
                    'answer':''.join(data['text'] for event,data in parsed if event=='token'),
                    'completed':any(event=='done' for event,_ in parsed),
                    'summary_active_at_response':bool(runtime.worker.tasks)}
                report['conversation_id']=cid
                report['turns'].append(row);save()
                print(json.dumps({k:row[k] for k in ('round','seconds','completed','summary_active_at_response')}),flush=True)
                if not row['completed']:raise RuntimeError('Turn failed or interrupted; see saved response status')
                # Simulate reading the completed reply; this wait is outside response latency.
                if runtime.worker.tasks:
                    await asyncio.wait(list(runtime.worker.tasks.values()),timeout=60)
            history=(await client.get(f'/api/conversations/{cid}/messages')).json()
            listing=(await client.get('/api/conversations')).json()
            snapshot=await service.graph.aget_state({'configurable':{'thread_id':str(cid)}})
            summaries=runtime.context_repository.load(int(cid)).summaries
            log_rows=[json.loads(line) for line in runtime.log.path.read_text(encoding='utf8').splitlines()
                      if json.loads(line).get('conversation_id')==int(cid)]
            report.update(summary_count=len(summaries),summaries=[s['content'] for s in summaries],
                business_messages=len(history['messages']),checkpoint_messages=len(snapshot.values['messages']),
                history_logs=sum(r['event']=='history_ctx' for r in log_rows),
                model_logs=sum(r['event']=='model_ctx' for r in log_rows),
                downgrade_events=sum(r['event']=='层1 降级' for r in log_rows),
                summary_done_events=sum(r['event']=='summary done' for r in log_rows),
                listed=any(r['id']==cid for r in listing['conversations']))
            final=report['turns'][-1]['answer']
            report['recall_correct']=cases['expected_order'] in final and any(term in final for term in cases['expected_demands'])
            compression_ok=(not summaries and report['downgrade_events']==0) if args.profile=='default' else (len(summaries)>0 and report['downgrade_events']>0)
            report['passed']=compression_ok and report['history_logs']==24 and report['business_messages']==48 and report['listed'] and report['recall_correct']
            save()
    print(json.dumps({k:report[k] for k in ('profile','passed','summary_count','downgrade_events','history_logs','recall_correct')}))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['default','demo'],default='default')
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--init-db',action='store_true')
    parser.add_argument('--env-file',default='.env')
    parser.add_argument('--output')
    asyncio.run(run(parser.parse_args()))
