"""Standalone Chapter 5 demo using the configured provider and read-only corpus."""
import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from app.agent_runtime import AgentLimits, READ_TOOLS, run_bare_agent
from app.config import Settings
from app.intent import classify_intent
from app.model_service import ModelService
from app.prompts import render_workflow_system_prompt
from app.tools import build_tools
from app.workflow_graph import build_workflow, open_workflow_checkpointer
from app.workflow_knowledge import create_live_knowledge_gate
from app.workflow_log import WorkflowLog
from app.workflow_service import knowledge_tool_answer
from app.workflow_types import Usage

# Current CLI inputs are independent of the historical simulation acceptance run.
CASES = {
    'logistics': {'question': '请查订单1001的物流到哪里了。'},
    'multi-step': {'question': '先查询订单1001的订单状态，只有结果为已发货或已完成时，再查询它的物流；否则不要查物流。'},
    'policy': {'question': '商品发错货引起退换货，运费由谁承担？'},
    'complaint': {'question': '我要投诉你们的服务态度，太差了。'},
    'chitchat': {'question': '你好'},
    'weak-evidence': {'question': 'MH-W60是否通过NASA火星载人任务认证？'},
    'missing-info': {'question': '帮我查一下物流，我还没有提供订单号。'},
}


class CountedModel:
    """Count attempted API operations, including failures, separately from turn count."""
    def __init__(self, model):
        self.model = model
        self.calls = {'classification': 0, 'selection': 0, 'final_stream': 0}
        self.chunks = 0

    async def classify_intent(self, text):
        self.calls['classification'] += 1
        return await self.model.classify_intent(text)

    async def select(self, *args, **kwargs):
        self.calls['selection'] += 1
        return await self.model.select(*args, **kwargs)

    async def stream_reply(self, *args, **kwargs):
        self.calls['final_stream'] += 1
        async for chunk in self.model.stream_reply(*args, **kwargs):
            if chunk.text:
                self.chunks += 1
            yield chunk


async def run_demo(mode, case, *, on_token=None):
    """One complete read-only chat attempt. Caller owns its live-request budget."""
    started = perf_counter()
    settings = Settings.from_env()
    counted = CountedModel(ModelService(settings))
    nodes, events = [], []
    async def emit(event, payload):
        events.append({'event': event, 'payload': payload})
        if event == 'token' and on_token:
            on_token(payload['text'])
    with TemporaryDirectory(prefix='sayhelp_ch05_demo_') as directory, ExitStack() as resources:
        log = WorkflowLog(Path(directory)/'log')
        gate = resources.enter_context(create_live_knowledge_gate(settings, log=log))
        async def answer(keyword, filters=None):
            return await knowledge_tool_answer(gate, keyword, filters)
        # Only read tools escape; order/logistics honestly report unconnected sources.
        tools = {n: t for n, t in build_tools(None, 0, None, knowledge_answer=answer).items() if n in READ_TOOLS}
        limits = AgentLimits(model_calls=settings.agent_model_calls, tool_calls=settings.agent_tool_calls,
            turn_tokens=settings.agent_turn_tokens, response_tokens=settings.response_token_reserve,
                    input_tokens=settings.context_token_budget-settings.response_token_reserve)
        question = CASES[case]['question']
        messages = [SystemMessage(content=render_workflow_system_prompt()), HumanMessage(content=question)]
        if mode == 'bare':
            result = await run_bare_agent(messages, model=counted, tools=tools, limits=limits, usage=Usage(), emit=emit)
            state = asdict(result)
        else:
            async def classifier(text):
                return await classify_intent(text, model=counted)
            async with open_workflow_checkpointer(Path(directory)/'checkpoint.sqlite') as saver:
                graph = build_workflow(model=counted, classifier=classifier, knowledge_gate=gate,
                    tools_factory=lambda cid: tools, limits=limits, log=log, checkpointer=saver)
                state = None
                async for namespace, kind, value in graph.astream(
                    {'conversation_id': 1, 'turn_id': 'demo', 'raw_question': question, 'messages': messages},
                    {'configurable': {'thread_id': 'demo'}, 'recursion_limit': 64},
                    stream_mode=['updates', 'custom', 'values'], subgraphs=True):
                    if kind == 'updates':
                        nodes.extend(value)
                    elif kind == 'custom':
                        await emit(value['event'], value['payload'])
                    elif not namespace:
                        state = value
        summary = {'mode': mode, 'case': case, 'question': question,
            'model_source': 'live configured provider',
            'model': settings.chat_model, 'business_data': 'data source not connected',
            'knowledge_source': 'existing read-only corpus',
            'complete_chat_requests': 1, 'logical_model_requests': sum(counted.calls.values()),
            'provider_request_counting': 'logical operations; chapter 5 SDK retries disabled, not HTTP telemetry',
            'model_operations': counted.calls, 'model_calls': state['model_calls'],
            'tool_calls': state['tool_calls'], 'usage': state['usage'],
            'final_stream_chunks': counted.chunks, 'nodes': nodes, 'node_count': len(nodes),
            'tool_names': [e['payload']['name'] for e in events if e['event'] == 'tool_start'],
            'event_order': [e['event'] for e in events], 'answer': state['answer'],
            'suggestions': state.get('suggestions', []), 'citations': state.get('citations', []),
            'stop_reason': state.get('stop_reason'), 'ticket_writes': 0,
            'seconds': round(perf_counter()-started, 3)}
        return summary


def main():
    parser = argparse.ArgumentParser(description='Chapter 5 demo using configured provider and read-only corpus; never creates tickets')
    parser.add_argument('--mode', choices=['bare', 'workflow'], required=True)
    parser.add_argument('--case', choices=list(CASES), required=True)
    parser.add_argument('--output', type=Path, help='Save summary JSON; contains only the authored demo question/results')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    print('LIVE configured model / existing read-only corpus / business data source not connected', flush=True)
    result = asyncio.run(run_demo(args.mode, args.case,
        on_token=lambda text: print(text, end='', flush=True)))
    print('\n' + json.dumps(result, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
