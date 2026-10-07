"""Short database transactions for conversations, literal FAQ search, and tickets."""

from uuid import uuid4
from hashlib import sha256
import json

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import Session, sessionmaker, undefer
from sqlalchemy.exc import IntegrityError

from app.ch05_db import WorkflowTicketRequest

from app.db import Conversation, Faq, Message, Ticket
from app.ch04_db import LowConfidenceQuestion
from app.knowledge_types import KnowledgeAnswer
from app.history_cursor import decode_cursor, encode_cursor


class Repository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def create_conversation(self, user_id: str) -> int:
        with self.session_factory.begin() as session:
            row = Conversation(user_id=user_id)
            session.add(row)
            session.flush()
            return row.id

    def get_conversation(self, id: int) -> Conversation | None:
        with self.session_factory() as session:
            return session.scalar(select(Conversation).options(undefer('*')).where(Conversation.id == id, Conversation.deleted_at.is_(None)))

    def append_message(self, id: int, role: str, content: str | None,
                       tool_calls: list[dict] | None = None,
                       tool_call_id: str | None = None) -> str:
        with self.session_factory.begin() as session:
            row = Message(conversation_id=id, role=role, content=content,
                          tool_calls=tool_calls, tool_call_id=tool_call_id)
            session.add(row)
            session.flush()
            message_id = str(row.id)
        return message_id

    def commit_knowledge_answer(self, conversation_id: int, answer: KnowledgeAnswer, raw_question: str) -> str:
        if (answer.useful and answer.pool_source is not None) or (not answer.useful and answer.pool_source not in ('retrieval_low_conf', 'self_check')):
            raise ValueError('Inconsistent knowledge answer and pool source')
        with self.session_factory.begin() as session:
            row = Message(conversation_id=conversation_id, role='assistant', content=answer.answer, citations=answer.citations)
            session.add(row)
            if not answer.useful:
                session.add(LowConfidenceQuestion(conversation_id=conversation_id, raw_question=raw_question,
                            source=answer.pool_source, reason=answer.reason))
            session.flush()
            message_id = str(row.id)
        return message_id

    def load_messages(self, id: int) -> list[Message]:
        with self.session_factory() as session:
            return list(session.scalars(select(Message).options(undefer(Message.citations)).where(Message.conversation_id == id).order_by(Message.id)))

    def set_pinned(self, id: int, is_pinned: bool) -> dict:
        if type(is_pinned) is not bool:
            raise ValueError('Pin state must be a boolean')
        with self.session_factory.begin() as session:
            row = session.get(Conversation, id, options=[undefer('*')], with_for_update=True)
            if row is None or row.deleted_at is not None:
                raise KeyError('Conversation not found')
            if bool(row.is_pinned) != is_pinned:
                row.is_pinned = int(is_pinned)
                row.pinned_at = func.now() if is_pinned else None
                session.flush()
            result = {'id': str(row.id), 'is_pinned': bool(row.is_pinned),
                      'pinned_at': row.pinned_at.isoformat() if row.pinned_at else None}
        return result

    def soft_delete_conversation(self, id: int) -> None:
        with self.session_factory.begin() as session:
            row = session.get(Conversation, id, options=[undefer('*')], with_for_update=True)
            if row is None:
                raise KeyError('Conversation not found')
            if row.deleted_at is None:
                row.deleted_at = func.now()

    def list_conversations(self, *, limit: int = 30, cursor: str | None = None) -> dict:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('History limit must be between 1 and 100')
        key = decode_cursor(cursor) if cursor is not None else None
        first_question = (select(Message.content).where(Message.conversation_id == Conversation.id,
                          Message.role == 'user').order_by(Message.id).limit(1).scalar_subquery())
        activity = (select(Message.conversation_id, func.max(Message.id).label('cursor'),
                    func.max(Message.created_at).label('last_activity'))
                    .group_by(Message.conversation_id).subquery())
        # Activity never changes the order within the pinned group.
        normal_activity = case((Conversation.is_pinned == 0, activity.c.cursor), else_=0)
        statement = (select(Conversation, first_question, activity.c.cursor, activity.c.last_activity)
                     .options(undefer('*')).join(activity, activity.c.conversation_id == Conversation.id)
                     .where(Conversation.deleted_at.is_(None))
                     .order_by(Conversation.is_pinned.desc(), Conversation.pinned_at.desc(),
                               normal_activity.desc(), Conversation.id.desc()).limit(limit + 1))
        if key is not None:
            if key['pin_group'] == 1:
                remaining = or_(Conversation.is_pinned == 0, and_(Conversation.is_pinned == 1,
                    or_(Conversation.pinned_at < key['pinned_at'],
                        and_(Conversation.pinned_at == key['pinned_at'], Conversation.id < key['id']))))
            else:
                remaining = and_(Conversation.is_pinned == 0, or_(activity.c.cursor < key['activity_id'],
                    and_(activity.c.cursor == key['activity_id'], Conversation.id < key['id'])))
            statement = statement.where(remaining)
        with self.session_factory() as session:
            rows = session.execute(statement).all()
            visible = rows[:limit]
            next_cursor = None
            if len(rows) > limit:
                row, _, activity_id, _ = visible[-1]
                next_cursor = encode_cursor(int(row.is_pinned), row.pinned_at, activity_id, row.id)
            return {'conversations': [{'id': str(row.id), 'title': (title or '新对话')[:64],
                     'status': row.status, 'updated_at': updated.isoformat(),
                     'is_pinned': bool(row.is_pinned),
                     'pinned_at': row.pinned_at.isoformat() if row.pinned_at else None}
                     for row, title, activity_id, updated in visible], 'next_cursor': next_cursor}

    def find_faq(self, keyword: str, limit: int = 5) -> list[Faq]:
        if not 1 <= limit <= 100:
            raise ValueError('FAQ limit must be between 1 and 100')
        if not keyword.strip():
            return []
        with self.session_factory() as session:
            statement = select(Faq).where(Faq.question.contains(keyword, autoescape=True)).order_by(Faq.id).limit(limit)
            return list(session.scalars(statement))

    def create_ticket(self, conversation_id: int, description: str, ticket_type: str, *,
                      request_key: str | None = None) -> str:
        if ticket_type not in ('售后', '投诉', '咨询'):
            raise ValueError('Unsupported ticket type')
        if request_key is not None and (not isinstance(request_key, str) or not request_key.strip() or len(request_key) > 128):
            raise ValueError('Invalid ticket request key')
        payload = json.dumps([conversation_id, description, ticket_type], ensure_ascii=False, separators=(',', ':'))
        digest = sha256(payload.encode('utf-8')).hexdigest()

        def active_conversation(session):
            conversation = session.get(Conversation, conversation_id, options=[undefer('*')], with_for_update=True)
            if conversation is None:
                raise ValueError('Conversation does not exist')
            if conversation.deleted_at is not None:
                raise KeyError('Conversation not found')

        def original_ticket(session):
            existing = session.get(WorkflowTicketRequest, request_key)
            if existing is not None:
                if existing.conversation_id != conversation_id or existing.payload_digest != digest:
                    raise ValueError('Ticket request payload conflict')
                return existing.ticket_no
            return None

        try:
            with self.session_factory.begin() as session:
                # Same lock as deletion/history: a stale confirmation cannot write
                # to a conversation deleted before this transaction acquired it.
                active_conversation(session)
                if request_key is not None:
                    original = original_ticket(session)
                    if original is not None:
                        return original
                number = 'T' + uuid4().hex[:31]
                session.add(Ticket(ticket_no=number, conversation_id=conversation_id,
                                   description=description, ticket_type=ticket_type))
                session.flush()
                if request_key is not None:
                    session.add(WorkflowTicketRequest(request_key=request_key, conversation_id=conversation_id,
                                                      payload_digest=digest, ticket_no=number))
                    session.flush()
            return number
        except IntegrityError as error:
            if request_key is None or getattr(error.orig, 'args', (None,))[0] != 1062:
                raise
            # Unique-key arbitration has completed at the DB. The failed session
            # is closed/rolled back; a fresh transaction reads the committed winner.
            with self.session_factory.begin() as session:
                active_conversation(session)
                original = original_ticket(session)
                if original is None:
                    raise error
                return original
