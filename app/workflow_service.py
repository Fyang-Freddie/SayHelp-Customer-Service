"""Reconcile MySQL history, stream the graph, and own its SQLite lifecycle."""
import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager, ExitStack
from uuid import uuid4

from anyio import CancelScope
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from app.agent_runtime import AgentLimits, READ_TOOLS
from app.chat_service import ChatEvent
from app.chat_views import public_actions
from app.history import prepare_context
from app.intent import classify_intent
from app.prompts import render_workflow_system_prompt
from app.tools import build_tools
from app.workflow_graph import build_workflow, open_workflow_checkpointer
from app.workflow_knowledge import create_live_knowledge_gate
from app.workflow_log import WorkflowLog
from app.workflow_repository import WorkflowRepository


async def _settled(function, *args, **kwargs):
    """Finish started database work even under repeated task/scope cancellation."""
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        with CancelScope(shield=True):
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
        if not worker.cancelled():
            worker.exception()
        raise


class WorkflowService:
    def __init__(self, repository, workflow_repository, graph, settings):
        self.repository = repository
        self.workflow_repository = workflow_repository
        self.graph = graph
        self.settings = settings

    async def _persist_reply(self, conversation_id, turn_id, current_id, result):
        # Parent agent outputs include its input history; fixed nodes only emit final.
        messages = result['messages']
        start = next((i + 1 for i, m in enumerate(messages) if m.id == current_id), 0)
        suffix = messages[start:]
        if not suffix or not isinstance(suffix[-1], AIMessage) or suffix[-1].tool_calls:
            raise RuntimeError('Workflow has no complete final message')
        for position, message in enumerate(suffix[:-1], 1):
            if isinstance(message, AIMessage):
                calls = [*message.tool_calls, *message.invalid_tool_calls]
                if not calls:
                    raise RuntimeError('Unexpected intermediate answer')
                await _settled(self.workflow_repository.append_message_once,
                    conversation_id, turn_id, position, 'assistant', message.content,
                    tool_calls=calls)
            elif isinstance(message, ToolMessage):
                await _settled(self.workflow_repository.append_message_once,
                    conversation_id, turn_id, position, 'tool', message.content,
                    tool_call_id=message.tool_call_id)
            else:
                raise RuntimeError('Unexpected workflow history message')
        return await _settled(self.workflow_repository.commit_reply, conversation_id,
            turn_id, len(suffix), result['answer'], result.get('citations', []),
            result.get('suggestions', []))

    async def stream_turn(self, conversation_id: int, message: str, filters=None) -> AsyncIterator[ChatEvent]:
        turn_id = uuid4().hex
        committed = False
        finished = False
        started = False
        try:
            if await asyncio.to_thread(self.repository.get_conversation, conversation_id) is None:
                raise KeyError('Conversation not found')
            rows = await asyncio.to_thread(self.repository.load_messages, conversation_id)
            excluded = await asyncio.to_thread(self.workflow_repository.incomplete_message_ids, conversation_id)
            current = HumanMessage(content=message, id=turn_id + ':user')
            context = prepare_context([r for r in rows if str(r.id) not in excluded],
                SystemMessage(content=render_workflow_system_prompt(), id='workflow-system'), current, self.settings)
            # The attempt may commit before cancellation is observed; cleanup always checks it.
            started = True
            current.id = await _settled(self.workflow_repository.append_message_once,
                conversation_id, turn_id, 0, 'user', message)
            inputs = {'conversation_id': conversation_id, 'turn_id': turn_id,
                'messages': [RemoveMessage(id=REMOVE_ALL_MESSAGES), *context], 'raw_question': message,
                'filters': filters.model_dump() if hasattr(filters, 'model_dump') else (filters or {})}
            config = {'configurable': {'thread_id': str(conversation_id)}, 'recursion_limit': 64}
            final = None
            streamed = False
            # Closing settles graph/checkpointer tasks before request reservation is released.
            async with aclosing(self.graph.astream(inputs, config,
                    stream_mode=['custom', 'updates'], subgraphs=True)) as stream:
                async for namespace, mode, data in stream:
                    if mode == 'custom' and data['event'] in ('tool_start', 'tool_end'):
                        yield ChatEvent('tool_status', {'name': data['payload']['name'],
                            'state': 'running' if data['event'] == 'tool_start' else data['payload']['status']})
                    elif mode == 'custom' and data['event'] == 'token':
                        if data['event'] == 'token':
                            streamed = True
                        yield ChatEvent(data['event'], data['payload'])
                    elif mode == 'updates' and not namespace:
                        for node, result in data.items():
                            if node not in ('agent', 'fallback', 'complaint', 'chitchat'):
                                continue
                            if result.get('status') != 'done' or not result.get('answer', '').strip():
                                raise RuntimeError('Workflow did not produce a complete answer')
                            final = result
                            message_id, actions = await self._persist_reply(
                                conversation_id, turn_id, current.id, result)
                            committed = True
            # Stream exhaustion confirms ALL graph tasks and final SQLite writes succeeded.
            if final is None or not committed:
                raise RuntimeError('Workflow ended without a persisted reply')
            if not streamed:
                # Fixed route text is one literal event; model replies are never re-chunked.
                yield ChatEvent('token', {'text': final['answer']})
            if final.get('citations'):
                yield ChatEvent('citations', {'message_id': message_id, 'citations': final['citations']})
            if actions:
                yield ChatEvent('actions', {'message_id': message_id, 'actions': public_actions(actions)})
            finished = True
            yield ChatEvent('completed', {'message_id': message_id})
        except BaseException as error:
            if started and not finished:
                status = ('incomplete' if isinstance(error, (asyncio.CancelledError, GeneratorExit))
                          else 'checkpoint_failed' if committed else 'failed')
                # A failed database can prevent this best-effort marker; never hide the original error.
                with CancelScope(shield=True):
                    try:
                        await _settled(self.workflow_repository.set_turn_status, turn_id, status)
                    except (Exception, asyncio.CancelledError):
                        pass
            raise


async def knowledge_tool_answer(gate, keyword, filters=None):
    """Existing tool wrappers receive complete gate evidence, without another model."""
    result = await gate.prepare(keyword, filters)
    useful = bool(result.get('confidence', {}).get('sufficient') and result.get('evidence'))
    evidence = result.get('evidence', []) if useful else []
    return {'keyword': keyword, 'matches': evidence, 'useful': useful,
            'message': '请依据完整证据回答' if useful else '知识证据不足，请人工核实',
            'reason': result.get('confidence', {}).get('reason'),
            'citations': result.get('citations', []) if useful else []}


@asynccontextmanager
async def open_workflow_service(repository, model_service, settings, *, knowledge_gate=None):
    """One app owns retrieval warmup and one official SQLite connection."""
    with ExitStack() as resources:
        log = WorkflowLog()
        if knowledge_gate is None:
            knowledge_gate = resources.enter_context(create_live_knowledge_gate(settings, log=log))
            await _settled(knowledge_gate.retrieval.warmup)
        async def classifier(text):
            return await classify_intent(text, model=model_service)
        async def answer(keyword, filters=None):
            return await knowledge_tool_answer(knowledge_gate, keyword, filters)
        def tools_factory(conversation_id):
            return {name: tool for name, tool in build_tools(repository, conversation_id, None,
                knowledge_answer=answer).items() if name in READ_TOOLS}
        async with open_workflow_checkpointer(settings.workflow_checkpoint_path) as saver:
            compiled = build_workflow(model=model_service, classifier=classifier,
                knowledge_gate=knowledge_gate, tools_factory=tools_factory,
                limits=AgentLimits(model_calls=settings.agent_model_calls,
                    tool_calls=settings.agent_tool_calls, turn_tokens=settings.agent_turn_tokens,
                    response_tokens=settings.response_token_reserve), log=log, checkpointer=saver)
            yield WorkflowService(repository, WorkflowRepository(repository.session_factory), compiled, settings)
