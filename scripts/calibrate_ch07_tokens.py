"""Bounded real usage comparison for the fixed prompt/tools and Chinese inputs."""
import argparse,asyncio,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import HumanMessage,SystemMessage
from app.context_budget import estimate_tokens
from app.prompts import render_workflow_system_prompt
from app.agent_runtime import _schemas
from app.tools import build_tools


async def main(args):
    from dotenv import load_dotenv
    load_dotenv(args.env_file,override=False)
    os.environ['CH07_EVAL_USAGE_PATH']=str(Path('.runtime/ch07/eval-usage.sqlite').resolve())
    from app.ch07_evaluation import evaluation_scope
    from app.config import Settings
    from app.model_service import ModelService
    model=ModelService(Settings.from_env())
    async def unused(*args,**kwargs):raise AssertionError('No tools executed during calibration')
    schemas=_schemas(build_tools(None,0,None,knowledge_answer=unused))[0]
    texts=['你好。','订单ABC123456需要核实，请保留手机号13800000001。',
           '商品📦需要核实；用户说法和工具事实不能混淆。'*8,
           '我要投诉配送沟通问题，尚未得到答复。'*32,
           '中文标点，。！？订单、物流与退款仅依据事实。'*64,
           'order ABC123456 pending; no refund confirmed. '*32,
           '订单ABC123456，用户要求核实。📦'*100,
           '之前订单ABC123456并未退款；现在只核实物流。'*100]
    rows=[]
    for index,text in enumerate(texts,1):
        messages=[SystemMessage(render_workflow_system_prompt()),HumanMessage(text)]
        with evaluation_scope('calibration'):
            response=await model.select(messages,schemas,max_tokens=512)
        usage=response.usage_metadata
        if not usage or not usage.get('input_tokens'):raise ValueError('Provider input usage is required')
        estimated=estimate_tokens(messages,schemas)
        rows.append({'case':index,'estimated':estimated,'provider_input':usage['input_tokens'],
            'ratio':round(estimated/usage['input_tokens'],4),
            'model':response.response_metadata.get('model_name')})
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps({'chars_per_token':1.0,'results':rows},ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(rows[-1]),flush=True)
    if any(row['estimated']<row['provider_input'] for row in rows):raise SystemExit('Estimator requires recalibration before acceptance')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',default='.env')
    parser.add_argument('--output',default='eval/ch07/token-calibration.json')
    asyncio.run(main(parser.parse_args()))
