"""Run the explicit chapter 5 loop: python -m app.bare_agent --message '查询订单1001'."""
import argparse
import asyncio
from contextlib import ExitStack
import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent_runtime import AgentLimits, READ_TOOLS, run_bare_agent
from app.config import Settings
from app.model_service import ModelService
from app.prompts import render_workflow_system_prompt
from app.tools import build_tools
from app.workflow_types import Usage
from app.workflow_knowledge import create_live_knowledge_gate
from app.workflow_service import knowledge_tool_answer


async def run_demo(message: str, *, settings=None, model=None):
    """Business-read demo: no conversation creation, history write, or ticket write."""
    settings = settings or Settings.from_env()
    with ExitStack() as resources:
        gate = resources.enter_context(create_live_knowledge_gate(settings))
        async def answer(keyword, filters=None):
            return await knowledge_tool_answer(gate, keyword, filters)
        tools = {name: item for name, item in build_tools(None, 0, None, knowledge_answer=answer).items()
                 if name in READ_TOOLS}
        async def emit(event, payload):
            if event == 'token':
                print(payload['text'], end='', flush=True)
            else:
                print('\n' + json.dumps({'event': event, **payload}, ensure_ascii=False), flush=True)
        result = await run_bare_agent(
            [SystemMessage(content=render_workflow_system_prompt()), HumanMessage(content=message)],
            model=model or ModelService(settings), tools=tools,
            limits=AgentLimits(model_calls=settings.agent_model_calls, tool_calls=settings.agent_tool_calls,
                turn_tokens=settings.agent_turn_tokens, response_tokens=settings.response_token_reserve,
                    input_tokens=settings.context_token_budget-settings.response_token_reserve),
            usage=Usage(), emit=emit)
        print('\n' + json.dumps({'model_calls': result.model_calls, 'tool_calls': result.tool_calls,
            'usage': result.usage.to_dict(), 'stop_reason': result.stop_reason}, ensure_ascii=False))
        return result


def main():
    parser = argparse.ArgumentParser(description='SayHelp bounded naked Agent business-read demo')
    parser.add_argument('--message', required=True)
    args = parser.parse_args()
    asyncio.run(run_demo(args.message))


if __name__ == '__main__':
    main()
