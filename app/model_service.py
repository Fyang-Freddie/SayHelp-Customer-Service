"""Configured chat model operations."""

from collections.abc import AsyncIterator

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.config import Settings


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
