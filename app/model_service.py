"""Configured chat model operations."""

from collections.abc import AsyncIterator
import json

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.prompts import render_extraction_prompt
from app.schemas import AfterSalesExtraction


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self._model = ChatOpenAI(
            base_url=settings.chat_base_url,
            model=settings.chat_model,
            api_key=settings.chat_api_key,
        )

    async def choose_tool(self, messages: list[BaseMessage], tools: list[BaseTool]) -> AIMessage:
        """Make one async selection request, leaving the final model unbound."""
        selection = await self._model.bind_tools(tools).ainvoke(messages)
        if not isinstance(selection, AIMessage):
            raise TypeError("Tool selection must return an AIMessage")
        return selection

    async def stream_chat(self, messages: list[BaseMessage]) -> AsyncIterator[str]:
        async for chunk in self._model.astream(messages):
            text = str(chunk.text)
            if text:
                yield text

    async def extract_after_sales(self, description: str) -> AfterSalesExtraction:
        structured_model = self._model.with_structured_output(
            AfterSalesExtraction, method="json_mode", include_raw=True
        )
        result = await structured_model.ainvoke(render_extraction_prompt(description))
        if result["parsing_error"] is not None:
            raise ValueError("Upstream extraction could not be parsed")
        raw = result["raw"].content

        def reject_non_json_constant(value: str) -> None:
            raise ValueError(f"Invalid JSON constant: {value}")

        payload = json.loads(raw, parse_constant=reject_non_json_constant)
        return AfterSalesExtraction.model_validate(payload)


    async def understand_query(self, raw_question: str):
        from app.query_understanding import QueryDecision
        from app.query_prompts import query_messages
        structured = self._model.with_structured_output(QueryDecision, method='json_mode', include_raw=True)
        result = await structured.ainvoke(query_messages(raw_question))
        if result['parsing_error'] is not None:
            raise ValueError('Query understanding returned invalid JSON')
        raw = result['raw'].content
        if not isinstance(raw,str) or len(raw)>50000:
            raise ValueError('Query understanding must return bounded JSON text')
        def reject_constant(value): raise ValueError('Invalid JSON constant')
        def unique_keys(items):
            value={}
            for key,item in items:
                if key in value: raise ValueError('Duplicate JSON key')
                value[key]=item
            return value
        try:
            return QueryDecision.model_validate(json.loads(raw,parse_constant=reject_constant,object_pairs_hook=unique_keys))
        except (ValueError,TypeError):
            raise ValueError('Query understanding returned invalid structured JSON') from None
