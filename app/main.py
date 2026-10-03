"""Customer service streaming HTTP API."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.config import Settings
from app.history import ConversationStore, InputBudgetExceeded, UnknownConversation
from app.model_service import ModelService
from app.prompts import render_service_system_prompt
from app.schemas import AfterSalesExtraction, ChatRequest, ExtractRequest


PreparedChat = tuple[str, HumanMessage, list[BaseMessage]]


def create_app(
    settings: Settings | None = None, model_service: ModelService | None = None
) -> FastAPI:
    settings = settings if settings is not None else Settings.from_env()
    model_service = model_service if model_service is not None else ModelService(settings)
    history = ConversationStore(settings)
    app = FastAPI()

    def prepare_chat(request: ChatRequest) -> PreparedChat:
        conversation_id = request.conversation_id
        user = HumanMessage(content=request.message)
        system = SystemMessage(content=render_service_system_prompt())
        if conversation_id is None:
            input_limit = settings.context_token_budget - settings.response_token_reserve
            if count_tokens_approximately([system, user]) > input_limit:
                raise HTTPException(status_code=413, detail="Message exceeds input budget")
            conversation_id = history.create()
        try:
            messages = history.prepare(conversation_id, system, user)
        except UnknownConversation as error:
            raise HTTPException(status_code=404, detail="Conversation not found") from error
        except InputBudgetExceeded as error:
            raise HTTPException(status_code=413, detail="Message exceeds input budget") from error
        return conversation_id, user, messages

    @app.post("/v1/chat/stream", response_class=EventSourceResponse)
    async def stream_chat(
        prepared: Annotated[PreparedChat, Depends(prepare_chat)],
    ) -> AsyncIterator[ServerSentEvent]:
        conversation_id, user, messages = prepared
        yield ServerSentEvent(event="session", data={"conversation_id": conversation_id})
        parts: list[str] = []
        try:
            async for chunk in model_service.stream_chat(messages):
                parts.append(chunk)
                yield ServerSentEvent(event="token", data={"text": chunk})
        except Exception:
            yield ServerSentEvent(event="error", data={"message": "Upstream chat failed"})
            return
        history.commit(conversation_id, user, AIMessage(content="".join(parts)))
        yield ServerSentEvent(event="done", data={"conversation_id": conversation_id})

    @app.post("/v1/aftersales/extract")
    async def extract_after_sales(request: ExtractRequest) -> AfterSalesExtraction:
        try:
            return await model_service.extract_after_sales(request.description)
        except Exception:
            raise HTTPException(status_code=502, detail="Upstream extraction failed") from None

    return app
