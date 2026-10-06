"""Persist one customer turn and execute at most one model-selected tool."""

import asyncio
import json
from contextlib import aclosing, nullcontext
from time import perf_counter
from app.latency import TurnTiming
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.config import Settings
from app.chat_views import sources_from_tool
from app.knowledge_search import KnowledgeSearch
from app.history import UnknownConversation, prepare_context
from app.model_service import ModelService
from app.prompts import render_service_system_prompt
from app.repository import Repository
from app.tool_executor import ToolExecutor
from app.tools import build_tools
from app.evidence import select_prompt_evidence
from app.knowledge_types import KnowledgeFilters


@dataclass(frozen=True)
class ChatEvent:
    kind: Literal['tool_status', 'sources', 'token', 'retrieval_status', 'citations', 'timings', 'completed']
    data: dict


class ChatService:
    def __init__(self, repository: Repository, model_service: ModelService,
                 settings: Settings, knowledge_search: KnowledgeSearch | None = None,
                 *, query_understanding=None, rag_retrieval=None, rag_generator=None,
                 confidence_policy=None) -> None:
        self.repository = repository
        self.model_service = model_service
        self.settings = settings
        self.knowledge_search = knowledge_search
        dependencies = (query_understanding, rag_retrieval, rag_generator, confidence_policy)
        if any(item is not None for item in dependencies) and any(item is None for item in dependencies):
            raise ValueError('All evidence pipeline dependencies are required')
        if all(item is None for item in dependencies) and knowledge_search is None:
            raise ValueError('Inject an evidence pipeline or an explicit legacy search adapter')
        self.query_understanding = query_understanding
        self.rag_retrieval = rag_retrieval
        self.rag_generator = rag_generator
        self.confidence_policy = confidence_policy

    async def _settled_write(self, function, *args, **kwargs):
        """Repeated cancellation must settle the synchronous transaction first."""
        worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
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

    async def _append_message(self, conversation_id, role, content,
                              tool_calls=None, tool_call_id=None):
        return await self._settled_write(self.repository.append_message,
            conversation_id, role, content, tool_calls=tool_calls, tool_call_id=tool_call_id)

    async def _retrieve_knowledge(self, query, filters, timing):
        with timing.measure('retrieval') if timing else nullcontext():
            result = await asyncio.to_thread(self.rag_retrieval.retrieve, query, 'hybrid_rerank', filters)
        if timing:
            timing.timings.update({'retrieval.' + k: v for k, v in result.timings_ms.items()})
            timing.candidates = len(result.candidates)
        return result

    async def _generate_knowledge(self, query, result, timing):
        with timing.measure('generation') if timing else nullcontext():
            prompt = select_prompt_evidence(query, result, self.settings)
            return await self.rag_generator.generate(query, prompt, self.confidence_policy.assess(result))

    async def _knowledge_answer(self, query, filters, timing=None):
        result = await self._retrieve_knowledge(query, filters, timing)
        return await self._generate_knowledge(query, result, timing), result

    @staticmethod
    def _tool_payload(keyword, answer, result):
        categories = {str(item.chunk.id): item.chunk.category for item in result.ranked}
        matches = [{'question': c['question'], 'answer': c['answer'],
                    'category': categories[c['chunk_id']]} for c in answer.citations] if answer.useful else []
        return {'keyword': keyword, 'matches': matches, 'useful': answer.useful,
                'message': answer.answer, 'citations': answer.citations}

    async def knowledge_tool_answer(self, keyword, filters=None):
        """A pure tool shares retrieval/self-check, without writing chat or pool rows."""
        query = await self.query_understanding.prepare(keyword)
        if query.intent != 'knowledge':
            return {'keyword': keyword, 'matches': [], 'useful': False,
                    'message': query.clarification or '请提出具体知识问题。', 'citations': []}
        answer, result = await self._knowledge_answer(query, filters)
        return self._tool_payload(keyword, answer, result)

    async def _publish_knowledge(self, conversation_id, raw_question, answer, timing=None):
        with timing.measure('persistence') if timing else nullcontext():
            message_id = await self._settled_write(self.repository.commit_knowledge_answer,
                                                  conversation_id, answer, raw_question)
        yield ChatEvent('citations', {'citations': answer.citations, 'message_id': message_id})
        for start in range(0, len(answer.answer), 40):
            yield ChatEvent('token', {'text': answer.answer[start:start + 40]})
        yield ChatEvent('completed', {'message_id': message_id})

    async def stream_turn(self, conversation_id: int, message: str, filters: KnowledgeFilters | None = None) -> AsyncIterator[ChatEvent]:
        timing = TurnTiming()
        outcome = 'cancelled'
        try:
            async with aclosing(self._stream_turn(conversation_id, message, filters, timing)) as turn:
                async for event in turn:
                    if event.kind == 'token' and timing.first_token_ms is None:
                        timing.first_token_ms = (perf_counter() - timing.started) * 1000
                    if event.kind == 'completed':
                        outcome = 'complete'
                        yield ChatEvent('timings', timing.payload(outcome))
                    yield event
        except asyncio.CancelledError:
            raise
        except Exception:
            outcome = 'error'
            raise
        finally:
            timing.log(outcome)

    async def _stream_turn(self, conversation_id, message, filters, timing):
        # The HTTP composition root reserves the conversation before entering SSE.
        conversation = await asyncio.to_thread(self.repository.get_conversation, conversation_id)
        if conversation is None:
            raise UnknownConversation(conversation_id)
        rows = await asyncio.to_thread(self.repository.load_messages, conversation_id)
        system = SystemMessage(content=render_service_system_prompt())
        current = HumanMessage(content=message)
        messages = prepare_context(rows, system, current, self.settings)
        await self._append_message(conversation_id, 'user', message)

        prepared_query = None
        if self.query_understanding is not None:
            yield ChatEvent('retrieval_status', {'state': 'running', 'stage': 'understanding'})
            with timing.measure('understanding'):
                prepared_query = await self.query_understanding.prepare(message)
            if prepared_query.intent == 'knowledge':
                yield ChatEvent('retrieval_status', {'state': 'running', 'stage': 'retrieval'})
                result = await self._retrieve_knowledge(prepared_query, filters, timing)
                yield ChatEvent('retrieval_status', {'state': 'running', 'stage': 'generation'})
                answer = await self._generate_knowledge(prepared_query, result, timing)
                yield ChatEvent('retrieval_status', {'state': 'complete', 'stage': 'complete'})
                async for event in self._publish_knowledge(conversation_id, message, answer, timing):
                    yield event
                return
            if prepared_query.intent == 'clarify':
                answer = prepared_query.clarification or '请补充具体问题或型号。'
                message_id = await self._append_message(conversation_id, 'assistant', answer)
                yield ChatEvent('token', {'text': answer})
                yield ChatEvent('completed', {'message_id': message_id})
                return
            if prepared_query.intent == 'chitchat':
                parts = []
                async for text in self.model_service.stream_chat([system, current]):
                    parts.append(text)
                    yield ChatEvent('token', {'text': text})
                message_id = await self._append_message(conversation_id, 'assistant', ''.join(parts))
                yield ChatEvent('completed', {'message_id': message_id})
                return

        tools = build_tools(self.repository, conversation_id, self.knowledge_search,
                            knowledge_answer=self.knowledge_tool_answer if prepared_query is not None else None)
        selection = await self.model_service.choose_tool(messages, list(tools.values()))
        # LangChain separates invalid JSON calls; both still need matching results.
        calls = [*selection.tool_calls, *selection.invalid_tool_calls]
        raw_calls = selection.additional_kwargs.get('tool_calls', [])
        if isinstance(raw_calls, list):
            order = {call.get('id'): index for index, call in enumerate(raw_calls)
                     if isinstance(call, dict)}
            calls.sort(key=lambda call: order.get(call.get('id'), len(order)))
        suffix = []
        controlled_answer = None
        if calls:
            await self._append_message(conversation_id,
                                    'assistant', selection.text or None, calls)
            ids = [call.get('id') for call in calls]
            if any(not isinstance(id, str) or not id for id in ids) or len(set(ids)) != len(ids):
                raise ValueError('Tool calls require unique nonempty IDs')
            suffix.append(selection)
            first = calls[0]
            public_name = first.get('name') if first.get('name') in tools else 'unknown'
            yield ChatEvent('tool_status', {'name': public_name, 'state': 'running'})
            executor = ToolExecutor(tools)
            for index, call in enumerate(calls):
                if index == 0:
                    if prepared_query is not None and public_name in ('query_faq', 'query_product'):
                        yield ChatEvent('retrieval_status', {'state': 'running'})
                        controlled_answer, retrieval = await self._knowledge_answer(prepared_query, filters, timing)
                        payload = self._tool_payload(message, controlled_answer, retrieval)
                        result = ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id=call['id'], status='success')
                        yield ChatEvent('retrieval_status', {'state': 'complete'})
                    else:
                        result = await executor.execute(call)
                else:
                    result = ToolMessage(content='本轮最多执行一个工具，此调用已跳过。',
                                         tool_call_id=call['id'], status='error')
                await self._append_message(conversation_id,
                                        'tool', result.content, tool_call_id=result.tool_call_id)
                suffix.append(result)
                if index == 0:
                    yield ChatEvent('tool_status', {'name': public_name, 'state': result.status})
                    if prepared_query is None and public_name == 'query_faq' and result.status == 'success':
                        sources = await asyncio.to_thread(sources_from_tool, result.content)
                        if sources:
                            yield ChatEvent('sources', {'sources': sources})

        if controlled_answer is not None:
            async for event in self._publish_knowledge(conversation_id, message, controlled_answer, timing):
                yield event
            return

        # All prior history pruning keeps the active request/results together.
        messages = prepare_context(rows, system, current, self.settings, suffix=suffix)
        parts = []
        async for text in self.model_service.stream_chat(messages):
            parts.append(text)
            yield ChatEvent('token', {'text': text})
        message_id = await self._append_message(conversation_id,
                                'assistant', ''.join(parts))
        if message_id is not None:
            yield ChatEvent('completed', {'message_id': message_id})
