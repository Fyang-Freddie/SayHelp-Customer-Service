"""Customer service streaming HTTP API with persistent conversation storage."""
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.chat_service import ChatService
from app.config import Settings
from app.embedding import BgeM3Embedder
from app.knowledge_search import KnowledgeSearch
from app.vector_store import MilvusKnowledgeStore
from app.db import make_session_factory
from app.history import InputBudgetExceeded, prepare_context
from app.model_service import ModelService
from app.prompts import render_service_system_prompt
from app.repository import Repository
from app.schemas import AfterSalesExtraction, ChatRequest, ExtractRequest

PreparedChat = tuple[int, str]


def create_app(
    settings: Settings | None = None,
    model_service: ModelService | None = None,
    session_factory: sessionmaker[Session] | None = None,
    knowledge_search: KnowledgeSearch | None = None,
) -> FastAPI:
    settings = settings if settings is not None else Settings.from_env()
    owns_engine = session_factory is None
    if session_factory is None:
        if not settings.database_url or not settings.database_url.strip():
            raise ValueError('DATABASE_URL is required unless session_factory is injected')
        session_factory = make_session_factory(settings.database_url)
    model_service = model_service if model_service is not None else ModelService(settings)
    repository = Repository(session_factory)
    if knowledge_search is None:
        # Shared across turns; BGE weights and the Milvus client remain lazy.
        knowledge_search = KnowledgeSearch(
            session_factory, BgeM3Embedder(cache_dir=settings.bge_cache_dir),
            MilvusKnowledgeStore(uri=settings.milvus_uri),
            min_score=settings.knowledge_min_score)
    service = ChatService(repository, model_service, settings, knowledge_search)
    active: set[int] = set()
    reservation_lock = RLock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            if owns_engine:
                session_factory.kw['bind'].dispose()

    app = FastAPI(lifespan=lifespan)

    @app.get('/', include_in_schema=False)
    async def chat_page() -> FileResponse:
        return FileResponse(Path(__file__).parent / 'web' / 'index.html', media_type='text/html')

    def prepare_chat(request: ChatRequest) -> Iterator[PreparedChat]:
        system = SystemMessage(content=render_service_system_prompt())
        current = HumanMessage(content=request.message)
        conversation_id = int(request.conversation_id) if request.conversation_id is not None else None
        reserved = False
        try:
            # Reject an oversized new turn before creating its conversation row.
            prepare_context([], system, current, settings)
            if conversation_id is not None and repository.get_conversation(conversation_id) is None:
                raise HTTPException(status_code=404, detail='Conversation not found')
            with reservation_lock:
                if conversation_id in active:
                    raise HTTPException(status_code=409, detail='Conversation is active')
                if len(active) >= settings.max_conversations:
                    raise HTTPException(status_code=503, detail='Conversation capacity reached')
                if conversation_id is None:
                    conversation_id = repository.create_conversation('guest-' + uuid4().hex)
                active.add(conversation_id)
                reserved = True
            prepare_context(repository.load_messages(conversation_id), system, current, settings)
            yield conversation_id, request.message
        except InputBudgetExceeded:
            raise HTTPException(status_code=413, detail='Message exceeds input budget') from None
        except SQLAlchemyError:
            raise HTTPException(status_code=503, detail='Conversation storage unavailable') from None
        finally:
            # Request scope tears down the SSE producer (including settled writes)
            # before this dependency is closed, also on cancellation/disconnect.
            if reserved:
                with reservation_lock:
                    active.discard(conversation_id)

    @app.post('/v1/chat/stream', response_class=EventSourceResponse)
    async def stream_chat(
        prepared: Annotated[PreparedChat, Depends(prepare_chat, scope='request')],
    ) -> AsyncIterator[ServerSentEvent]:
        conversation_id, message = prepared
        wire_id = str(conversation_id)
        yield ServerSentEvent(event='session', data={'conversation_id': wire_id})
        try:
            async with aclosing(service.stream_turn(conversation_id, message)) as turn:
                async for event in turn:
                    yield ServerSentEvent(event=event.kind, data=event.data)
        except Exception:
            yield ServerSentEvent(event='error', data={'message': 'Upstream chat failed'})
            return
        yield ServerSentEvent(event='done', data={'conversation_id': wire_id})

    @app.post('/v1/aftersales/extract')
    async def extract_after_sales(request: ExtractRequest) -> AfterSalesExtraction:
        try:
            return await model_service.extract_after_sales(request.description)
        except Exception:
            raise HTTPException(status_code=502, detail='Upstream extraction failed') from None

    return app
