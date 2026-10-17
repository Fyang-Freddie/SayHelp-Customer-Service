"""Print budget from the actual service prompt and bound tool schemas; no I/O tools run."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import SystemMessage
from app.agent_runtime import _schemas
from app.tools import build_tools
from app.prompts import render_workflow_system_prompt
from app.context_budget import estimate_tokens, calculate_budget, validate_startup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['default','demo'],default='default')
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    async def knowledge_answer(*a,**kw):
        raise AssertionError('Budget inspection never executes tools')
    schemas,_=_schemas(build_tools(None,0,None,knowledge_answer=knowledge_answer))
    prefix=estimate_tokens([SystemMessage(render_workflow_system_prompt())],schemas)
    ledger=calculate_budget(window=18000 if args.profile=='demo' else 65536,
        prefix=prefix,evidence=2560,summary=512,output=2000,user=2000,
        steps=3,selection=512,tool_results=1200,protocol=32,target_turns=32,steady=1024)
    validate_startup(ledger)
    print(json.dumps({'profile':args.profile, 'model_capacity_verified':False,
        'langchain_core':importlib.metadata.version('langchain-core'),
        'chars_per_token':1.0, **asdict(ledger)},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
