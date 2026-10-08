"""Customer service streaming HTTP API with persistent conversation storage."""
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.chat_views import public_messages, read_document
from app.config import Settings
from app.ch04_ingest import DOCUMENTS
from app.workflow_service import open_workflow_service
from app.workflow_repository import WorkflowRepository
from app.db import make_session_factory
from app.history import InputBudgetExceeded, prepare_context
from app.model_service import ModelService
from app.prompts import render_workflow_system_prompt
from app.ticket_actions import TicketActions
from app.repository import Repository
from app.schemas import AfterSalesExtraction, ChatRequest, ConversationId, ExtractRequest, PinRequest, TicketConfirmationRequest

PreparedChat = tuple[int, str, object]


def create_app(
    settings: Settings | None = None,
    model_service: ModelService | None = None,
    session_factory: sessionmaker[Session] | None = None,
    *, knowledge_gate=None,
) -> FastAPI:
    settings = settings if settings is not None else Settings.from_env()
    owns_engine = session_factory is None
    if session_factory is None:
        if not settings.database_url or not settings.database_url.strip():
            raise ValueError('DATABASE_URL is required unless session_factory is injected')
        session_factory = make_session_factory(settings.database_url)
    model_service = model_service if model_service is not None else ModelService(settings)
    repository = Repository(session_factory)
    workflow_repository = WorkflowRepository(session_factory)
    ticket_actions = TicketActions(repository, workflow_repository)
    active: set[int] = set()
    reservation_lock = RLock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            async with open_workflow_service(repository, model_service, settings,
                    knowledge_gate=knowledge_gate) as service:
                app.state.workflow_service = service
                try:
                    yield
                finally:
                    del app.state.workflow_service
        finally:
            if owns_engine:
                session_factory.kw['bind'].dispose()

    app = FastAPI(lifespan=lifespan)

    @app.get('/', include_in_schema=False)
    async def chat_page() -> FileResponse:
        return FileResponse(Path(__file__).parent / 'web' / 'index.html', media_type='text/html')

    @app.get('/v1/conversations')
    def conversation_list(limit: Annotated[int, Query(ge=1, le=100)] = 30,
                          before: Annotated[str | None, Query(max_length=512)] = None) -> dict:
        try:
            return repository.list_conversations(limit=limit, cursor=before)
        except ValueError:
            raise HTTPException(status_code=422, detail='Invalid history cursor; refresh the history list') from None
        except SQLAlchemyError:
            raise HTTPException(status_code=503, detail='Conversation storage unavailable') from None

    @app.patch('/v1/conversations/{conversation_id}/pin')
    def pin_conversation(conversation_id: ConversationId, request: PinRequest) -> dict:
        try:
            with reservation_lock:
                id = int(conversation_id)
                if id in active:
                    raise HTTPException(status_code=409, detail='Conversation is active')
                return repository.set_pinned(id, request.is_pinned)
        except KeyError:
            raise HTTPException(status_code=404, detail='Conversation not found') from None
        except SQLAlchemyError:
            raise HTTPException(status_code=503, detail='Conversation storage unavailable') from None

    @app.delete('/v1/conversations/{conversation_id}', status_code=204)
    def delete_conversation(conversation_id: ConversationId) -> Response:
        try:
            with reservation_lock:
                id = int(conversation_id)
                if id in active:
                    raise HTTPException(status_code=409, detail='Conversation is active')
                repository.soft_delete_conversation(id)
            return Response(status_code=204)
        except KeyError:
            raise HTTPException(status_code=404, detail='Conversation not found') from None
        except SQLAlchemyError:
            raise HTTPException(status_code=503, detail='Conversation storage unavailable') from None

    @app.get('/v1/conversations/{conversation_id}/messages')
    def conversation_messages(conversation_id: ConversationId) -> dict:
        try:
            if repository.get_conversation(int(conversation_id)) is None:
                raise HTTPException(status_code=404, detail='Conversation not found')
            return {'conversation_id': conversation_id,
                    'messages': public_messages(repository.load_messages(int(conversation_id)),
                        workflow_repository.load_actions(int(conversation_id)))}
        except SQLAlchemyError:
            raise HTTPException(status_code=503, detail='Conversation storage unavailable') from None

    @app.post('/v1/conversations/{conversation_id}/tickets')
    async def confirm_ticket(conversation_id: ConversationId, request: TicketConfirmationRequest) -> dict:
        id = int(conversation_id)
        reserved = False
        try:
            with reservation_lock:
                if id in active:
                    raise HTTPException(status_code=409, detail='Conversation is active')
                active.add(id)
                reserved = True
            # RLock is thread based; release it before awaiting, keep active claimed.
            return await ticket_actions.confirm(id, request.action_id,
                request.description, request.ticket_type)
        except KeyError:
            raise HTTPException(status_code=404, detail='Conversation or action not found') from None
        except ValueError:
            raise HTTPException(status_code=409, detail='Ticket action or request conflicts') from None
        except (SQLAlchemyError, TimeoutError):
            raise HTTPException(status_code=503,
                detail='Ticket result unavailable; retry this action with the same details') from None
        finally:
            if reserved:
                with reservation_lock:
                    active.discard(id)

    @app.get('/v1/knowledge/documents/{filename}')
    def knowledge_document(filename: str) -> dict:
        try:
            if 'knowledge_db/' + filename not in DOCUMENTS:
                raise FileNotFoundError('Knowledge document not found')
            return read_document(filename)
        except (OSError, UnicodeError):
            raise HTTPException(status_code=404, detail='Knowledge document not found') from None

    def prepare_chat(request: ChatRequest) -> Iterator[PreparedChat]:
        system = SystemMessage(content=render_workflow_system_prompt())
        current = HumanMessage(content=request.message)
        conversation_id = int(request.conversation_id) if request.conversation_id is not None else None
        reserved = False
        try:
            # Reject an oversized new turn before creating its conversation row.
            prepare_context([], system, current, settings)
            with reservation_lock:
                if conversation_id is not None and repository.get_conversation(conversation_id) is None:
                    raise HTTPException(status_code=404, detail='Conversation not found')
                if conversation_id in active:
                    raise HTTPException(status_code=409, detail='Conversation is active')
                if len(active) >= settings.max_conversations:
                    raise HTTPException(status_code=503, detail='Conversation capacity reached')
                if conversation_id is None:
                    conversation_id = repository.create_conversation('guest-' + uuid4().hex)
                active.add(conversation_id)
                reserved = True
            prepare_context(repository.load_messages(conversation_id), system, current, settings)
            yield conversation_id, request.message, request.filters
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
        conversation_id, message, filters = prepared
        message_id = None
        wire_id = str(conversation_id)
        yield ServerSentEvent(event='session', data={'conversation_id': wire_id})
        try:
            async with aclosing(app.state.workflow_service.stream_turn(conversation_id, message, filters)) as turn:
                async for event in turn:
                    if event.kind == 'completed':
                        message_id = event.data['message_id']
                    else:
                        yield ServerSentEvent(event=event.kind, data=event.data)
        except Exception:
            yield ServerSentEvent(event='error', data={'message': 'Upstream chat failed'})
            return
        if message_id is None:
            yield ServerSentEvent(event='error', data={'message': 'Workflow did not complete'})
            return
        yield ServerSentEvent(event='done', data={'conversation_id': wire_id, 'message_id': message_id})

    @app.post('/v1/aftersales/extract')
    async def extract_after_sales(request: ExtractRequest) -> AfterSalesExtraction:
        try:
            return await model_service.extract_after_sales(request.description)
        except Exception:
            raise HTTPException(status_code=502, detail='Upstream extraction failed') from None

    return app
