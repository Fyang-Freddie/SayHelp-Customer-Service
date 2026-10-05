"""Persist one customer turn and execute at most one model-selected tool."""

import asyncio
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


@dataclass(frozen=True)
class ChatEvent:
    kind: Literal['tool_status', 'sources', 'token']
    data: dict


class ChatService:
    def __init__(self, repository: Repository, model_service: ModelService,
                 settings: Settings, knowledge_search: KnowledgeSearch) -> None:
        self.repository = repository
        self.model_service = model_service
        self.settings = settings
        self.knowledge_search = knowledge_search

    async def _append_message(self, conversation_id: int, role: str, content: str | None,
                              tool_calls: list[dict] | None = None,
                              tool_call_id: str | None = None) -> None:
        """Settle an in-flight write before cancellation releases its reservation."""
        worker = asyncio.create_task(asyncio.to_thread(
            self.repository.append_message, conversation_id, role, content,
            tool_calls=tool_calls, tool_call_id=tool_call_id))
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Cancellation cannot stop the sync transaction. Repeated cancellation
            # must also wait so this write cannot land inside the following turn.
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not worker.cancelled():
                worker.exception()  # Retrieve any write error; preserve cancellation.
            raise

    async def stream_turn(self, conversation_id: int, message: str) -> AsyncIterator[ChatEvent]:
        # The HTTP composition root reserves the conversation before entering SSE.
        conversation = await asyncio.to_thread(self.repository.get_conversation, conversation_id)
        if conversation is None:
            raise UnknownConversation(conversation_id)
        rows = await asyncio.to_thread(self.repository.load_messages, conversation_id)
        system = SystemMessage(content=render_service_system_prompt())
        current = HumanMessage(content=message)
        messages = prepare_context(rows, system, current, self.settings)
        await self._append_message(conversation_id, 'user', message)

        tools = build_tools(self.repository, conversation_id, self.knowledge_search)
        selection = await self.model_service.choose_tool(messages, list(tools.values()))
        # LangChain separates invalid JSON calls; both still need matching results.
        calls = [*selection.tool_calls, *selection.invalid_tool_calls]
        raw_calls = selection.additional_kwargs.get('tool_calls', [])
        if isinstance(raw_calls, list):
            order = {call.get('id'): index for index, call in enumerate(raw_calls)
                     if isinstance(call, dict)}
            calls.sort(key=lambda call: order.get(call.get('id'), len(order)))
        suffix = []
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
                    result = await executor.execute(call)
                else:
                    result = ToolMessage(content='本轮最多执行一个工具，此调用已跳过。',
                                         tool_call_id=call['id'], status='error')
                await self._append_message(conversation_id,
                                        'tool', result.content, tool_call_id=result.tool_call_id)
                suffix.append(result)
                if index == 0:
                    yield ChatEvent('tool_status', {'name': public_name, 'state': result.status})
                    if public_name == 'query_faq' and result.status == 'success':
                        sources = await asyncio.to_thread(sources_from_tool, result.content)
                        if sources:
                            yield ChatEvent('sources', {'sources': sources})

        # All prior history pruning keeps the active request/results together.
        messages = prepare_context(rows, system, current, self.settings, suffix=suffix)
        parts = []
        async for text in self.model_service.stream_chat(messages):
            parts.append(text)
            yield ChatEvent('token', {'text': text})
        await self._append_message(conversation_id,
                                'assistant', ''.join(parts))
