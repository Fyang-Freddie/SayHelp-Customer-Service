"""Standalone, read-only Chapter 5 demo; no provider or business writes by default."""
import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from app.agent_runtime import AgentLimits, READ_TOOLS, run_bare_agent
from app.config import Settings
from app.evaluation.calibration import ConfidencePolicy
from app.intent import classify_intent
from app.knowledge_types import EvidenceChunk, RankedChunk, RetrievalResult
from app.model_service import ModelService
from app.prompts import render_workflow_system_prompt
from app.tools import build_tools
from app.workflow_graph import build_workflow, open_workflow_checkpointer
from app.workflow_knowledge import KnowledgeGate, create_live_knowledge_gate
from app.workflow_log import WorkflowLog
from app.workflow_service import knowledge_tool_answer
from app.workflow_types import Usage

CASES = {row['id']: row for row in json.loads((ROOT/'eval/ch05/acceptance_cases.json').read_text(encoding='utf-8'))}


class SimulationRetrieval:
    """Fixed evidence ONLY for simulation; live uses the existing read-only corpus."""
    def retrieve(self, question, strategy, filters):
        score = .1 if 'NASA' in question.raw_question else .9
        chunk = EvidenceChunk(1, '发错货退换货运费', '因发错货引起退换货，运费由平台承担。',
            '退货退款政策', None, 'policy', '退货退款政策 / 退换货运费承担',
            'knowledge_db/returns-policy.md', 1, 1, 'a'*64)
        ranked = [RankedChunk(chunk, 1, score)]
        return RetrievalResult(strategy, question, ranked, ranked, {})


class SimulationModel:
    """Authored decisions illustrate control flow, never measured model capability."""
    def __init__(self, case):
        self.case = case

    async def classify_intent(self, text):
        return AIMessage(content=json.dumps({'intent': CASES[self.case]['intent']}, ensure_ascii=False))

    async def select(self, messages, tools, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        name, args = None, {}
        if self.case == 'logistics' and not results:
            name, args = 'query_logistics', {'order_id': '1001'}
        elif self.case == 'multi-step':
            if not results:
                name, args = 'query_order', {'order_id': '1001'}
            elif len(results) == 1 and json.loads(results[0].content)['status'] in ('已发货', '已完成'):
                name, args = 'query_logistics', {'order_id': json.loads(results[0].content)['order_id']}
        elif self.case == 'policy' and not results and not any('完整检索证据' in str(m.content) for m in messages):
            name, args = 'query_faq', {'keyword': CASES[self.case]['question']}
        elif self.case == 'complaint' and not results:
            name, args = 'suggest_actions', {'actions': [{'kind': 'handoff'}, {'kind': 'create_ticket',
                'description': CASES[self.case]['question'], 'ticket_type': '投诉'}]}
        if name:
            return AIMessage(content='内部脚本草稿不输出', tool_calls=[{'name': name, 'args': args, 'id': f'demo-{len(results)}'}])
        return AIMessage(content='')

    async def stream_reply(self, messages, **kwargs):
        results = [m.content for m in messages if isinstance(m, ToolMessage) and m.name != 'suggest_actions']
        answer = {'policy': 'simulation：发错货退换货运费由平台承担。[1]',
            'missing-info': '请提供订单号，以便查询模拟物流数据。',
            'weak-evidence': '当前证据不足，请人工核实。',
            'chitchat': '您好，我是SayHelp演示助手。',
            'complaint': '很抱歉，您可以独立选择转人工或建立工单，建议尚未执行。'}.get(self.case)
        answer = answer or 'simulation：以下是随机模拟业务数据：' + '；'.join(results)
        for piece in (answer[:len(answer)//2], answer[len(answer)//2:]):
            yield AIMessageChunk(content=piece)


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


async def run_demo(mode, case, *, live=False, on_token=None):
    """One complete read-only chat attempt. Caller owns its live-request budget."""
    started = perf_counter()
    settings = Settings.from_env() if live else Settings('https://example.invalid', 'scripted-simulation', 'unused')
    counted = CountedModel(ModelService(settings) if live else SimulationModel(case))
    nodes, events = [], []
    async def emit(event, payload):
        events.append({'event': event, 'payload': payload})
        if event == 'token' and on_token:
            on_token(payload['text'])
    with TemporaryDirectory(prefix='sayhelp_ch05_demo_') as directory, ExitStack() as resources:
        log = WorkflowLog(Path(directory)/'log')
        gate = resources.enter_context(create_live_knowledge_gate(settings, log=log)) if live else KnowledgeGate(
            SimulationRetrieval(), ConfidencePolicy({'hybrid_rerank': .5}, {}), settings, log=log)
        async def answer(keyword, filters=None):
            return await knowledge_tool_answer(gate, keyword, filters)
        # Existing tools close over no business storage; only READ_TOOLS escape.
        tools = {n: t for n, t in build_tools(None, 0, None, knowledge_answer=answer).items() if n in READ_TOOLS}
        limits = AgentLimits(model_calls=settings.agent_model_calls, tool_calls=settings.agent_tool_calls,
            turn_tokens=settings.agent_turn_tokens, response_tokens=settings.response_token_reserve)
        question = CASES[case]['question']
        messages = [SystemMessage(content=render_workflow_system_prompt()), HumanMessage(content=question)]
        # Scope a deterministic RNG to ORIGINAL mock tools; even --live is simulated business data.
        with patch('app.tools.random', random.Random(5)):
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
                'model_source': 'live configured provider' if live else 'simulation',
                'model': settings.chat_model, 'business_data': 'seeded random simulation',
                'knowledge_source': 'existing read-only corpus' if live else 'fixed simulation evidence',
                'complete_chat_requests': int(live), 'provider_requests': sum(counted.calls.values()) if live else 0,
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
    parser = argparse.ArgumentParser(description='Read-only chapter 5 demo; defaults to simulation, never creates tickets')
    parser.add_argument('--mode', choices=['bare', 'workflow'], required=True)
    parser.add_argument('--case', choices=list(CASES), required=True)
    parser.add_argument('--live', action='store_true', help='Use configured provider and existing read-only retrieval')
    parser.add_argument('--output', type=Path, help='Save summary JSON; contains only the authored demo question/results')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    print('LIVE model / SIMULATION business data' if args.live else 'SIMULATION model, knowledge and business data', flush=True)
    result = asyncio.run(run_demo(args.mode, args.case, live=args.live,
        on_token=lambda text: print(text, end='', flush=True)))
    print('\n' + json.dumps(result, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
