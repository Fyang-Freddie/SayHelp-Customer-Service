"""Configured chat model operations."""

from collections.abc import AsyncIterator
import json

from langchain_core.messages import BaseMessage
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
        payload = json.loads(raw)
        return AfterSalesExtraction.model_validate(payload)
