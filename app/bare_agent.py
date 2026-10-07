"""Run the explicit chapter 5 loop: python -m app.bare_agent --message '查询订单1001'."""
import argparse
import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent_runtime import AgentLimits, run_bare_agent
from app.config import Settings
from app.db import make_session_factory
from app.model_service import ModelService
from app.prompts import render_workflow_system_prompt
from app.repository import Repository
from app.tools import build_tools
from app.workflow_types import Usage


async def run_demo(message: str, *, settings=None, model=None):
    """Business-read demo: no conversation creation, history write, or ticket write."""
    settings = settings or Settings.from_env()
    sessions = make_session_factory(settings.database_url)
    try:
        # Reuse the existing chapter 2 business tools. The standalone demo does
        # not initialize a knowledge pipeline; chapter 5 workflow injects it.
        tools = {n:t for n,t in build_tools(Repository(sessions), 0, None).items()
                 if n in {'query_order', 'query_logistics', 'query_product'}}
        async def emit(event, payload):
            if event == 'token':
                print(payload['text'], end='', flush=True)
            else:
                print('\n' + json.dumps({'event': event, **payload}, ensure_ascii=False), flush=True)
        result = await run_bare_agent(
            [SystemMessage(content=render_workflow_system_prompt()), HumanMessage(content=message)],
            model=model or ModelService(settings), tools=tools,
            limits=AgentLimits(model_calls=settings.agent_model_calls, tool_calls=settings.agent_tool_calls,
                turn_tokens=settings.agent_turn_tokens, response_tokens=settings.response_token_reserve),
            usage=Usage(), emit=emit)
        print('\n' + json.dumps({'model_calls': result.model_calls, 'tool_calls': result.tool_calls,
            'usage': result.usage.to_dict(), 'stop_reason': result.stop_reason}, ensure_ascii=False))
        return result
    finally:
        sessions.kw['bind'].dispose()


def main():
    parser = argparse.ArgumentParser(description='SayHelp bounded naked Agent business-read demo')
    parser.add_argument('--message', required=True)
    args = parser.parse_args()
    asyncio.run(run_demo(args.message))


if __name__ == '__main__':
    main()
