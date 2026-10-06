"""Short database transactions for conversations, literal FAQ search, and tickets."""

from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker, undefer

from app.db import Conversation, Faq, Message, Ticket
from app.ch04_db import LowConfidenceQuestion
from app.knowledge_types import KnowledgeAnswer


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
            return session.get(Conversation, id)

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

    def list_conversations(self, *, limit: int = 30, before: int | None = None) -> dict:
        first_question = (select(Message.content).where(Message.conversation_id == Conversation.id,
                          Message.role == 'user').order_by(Message.id).limit(1).scalar_subquery())
        activity = (select(Message.conversation_id, func.max(Message.id).label('cursor'),
                    func.max(Message.created_at).label('last_activity'))
                    .group_by(Message.conversation_id).subquery())
        statement = (select(Conversation, first_question, activity.c.cursor, activity.c.last_activity)
                     .join(activity, activity.c.conversation_id == Conversation.id)
                     .order_by(activity.c.cursor.desc()).limit(limit + 1))
        if before is not None:
            statement = statement.where(activity.c.cursor < before)
        with self.session_factory() as session:
            rows = session.execute(statement).all()
            return {'conversations': [{'id': str(row.id), 'title': (title or '新对话')[:64],
                     'status': row.status, 'updated_at': updated.isoformat()}
                     for row, title, cursor, updated in rows[:limit]],
                    'next_cursor': str(rows[limit - 1][2]) if len(rows) > limit else None}

    def find_faq(self, keyword: str, limit: int = 5) -> list[Faq]:
        if not 1 <= limit <= 100:
            raise ValueError('FAQ limit must be between 1 and 100')
        if not keyword.strip():
            return []
        with self.session_factory() as session:
            statement = select(Faq).where(Faq.question.contains(keyword, autoescape=True)).order_by(Faq.id).limit(limit)
            return list(session.scalars(statement))

    def create_ticket(self, conversation_id: int, description: str, ticket_type: str) -> str:
        if ticket_type not in ('售后', '投诉', '咨询'):
            raise ValueError('Unsupported ticket type')
        number = 'T' + uuid4().hex[:31]
        with self.session_factory.begin() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise ValueError('Conversation does not exist')
            session.add(Ticket(ticket_no=number, conversation_id=conversation_id,
                               description=description, ticket_type=ticket_type))
            conversation.status = '已转人工'
        return number
