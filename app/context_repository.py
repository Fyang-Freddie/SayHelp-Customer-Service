"""Short row-locked transactions; no model or checkpoint work under a SQL lock."""
from datetime import datetime, timedelta, UTC
from uuid import uuid4
from sqlalchemy import select, func
from sqlalchemy.orm import aliased
from app.ch07_db import ConversationContext, ConversationSummary
from app.context_types import ContextSnapshot, SummaryClaim
from app.db import Message
from app.db import Conversation


class ContextRepository:
    def __init__(self, session_factory):
        self.session_factory=session_factory

    def expired_conversations(self,limit=100):
        active=aliased(Conversation)
        with self.session_factory() as session:
            return list(session.scalars(select(ConversationContext.id).join(
                active,active.id==ConversationContext.id).where(
                active.deleted_at.is_(None),
                ConversationContext.summary_lease_token.is_not(None),
                ConversationContext.summary_lease_until<=datetime.now(UTC).replace(tzinfo=None))
                .order_by(ConversationContext.summary_lease_until).limit(limit)))

    def _row(self, session, cid, lock=False):
        row=session.get(ConversationContext,cid,with_for_update=lock)
        if row is None: raise KeyError('Conversation not found')
        return row

    def load(self, conversation_id: int) -> ContextSnapshot:
        with self.session_factory.begin() as session:
            row=self._row(session,conversation_id,True)
            summaries=session.scalars(select(ConversationSummary).where(
                ConversationSummary.conversation_id==conversation_id).order_by(ConversationSummary.ordinal)).all()
            return ContextSnapshot(conversation_id,row.context_version,row.summary_upto_msg_id,
                row.layer1_from_msg_id,tuple({k:getattr(s,k) for k in
                    ('ordinal','from_msg_id','upto_msg_id','content')} for s in summaries))

    def _message(self, session, cid, id, role):
        message=session.get(Message,id)
        if message is None or message.conversation_id!=cid or message.role!=role:
            raise ValueError('Invalid context boundary')
        return message

    def advance_layer1(self, conversation_id: int, expected_version: int, from_msg_id: int) -> bool:
        with self.session_factory.begin() as session:
            row=self._row(session,conversation_id,True)
            if row.context_version!=expected_version: return False
            self._message(session,conversation_id,from_msg_id,'user')
            if row.layer1_from_msg_id is not None and from_msg_id<row.layer1_from_msg_id:
                raise ValueError('Context boundary cannot move backwards')
            if row.summary_upto_msg_id is not None and from_msg_id<=row.summary_upto_msg_id:
                raise ValueError('Layer 1 overlaps summaries')
            if row.layer1_from_msg_id!=from_msg_id:
                row.layer1_from_msg_id=from_msg_id
                row.context_version+=1
            return True

    def claim(self, conversation_id: int, from_msg_id: int, upto_msg_id: int, digest: str,
              *,excluded_message_ids=()) -> SummaryClaim | None:
        with self.session_factory.begin() as session:
            row=self._row(session,conversation_id,True)
            now=datetime.now(UTC).replace(tzinfo=None)
            if row.summary_lease_until and row.summary_lease_until>now: return None
            if row.summary_upto_msg_id is not None and upto_msg_id<=row.summary_upto_msg_id: return None
            self._message(session,conversation_id,from_msg_id,'user')
            end=self._message(session,conversation_id,upto_msg_id,'assistant')
            if end.tool_calls or not row.layer1_from_msg_id or not from_msg_id<=upto_msg_id<row.layer1_from_msg_id:
                raise ValueError('Summary must cover complete layer 2 turns')
            first=session.scalar(select(Message.id).where(Message.conversation_id==conversation_id,
                Message.role=='user',Message.id>(row.summary_upto_msg_id or 0),
                Message.id.not_in([int(id) for id in excluded_message_ids])).order_by(Message.id).limit(1))
            if first!=from_msg_id: raise ValueError('Summary would skip uncovered history')
            token=uuid4().hex
            expires=now+timedelta(seconds=120)
            ordinal=1+(session.scalar(select(func.max(ConversationSummary.ordinal)).where(
                ConversationSummary.conversation_id==conversation_id)) or 0)
            row.summary_lease_token, row.summary_lease_until=token,expires
            row.summary_from_id,row.summary_to_id,row.summary_source_digest=from_msg_id,upto_msg_id,digest
            return SummaryClaim(conversation_id,ordinal,from_msg_id,upto_msg_id,
                row.summary_upto_msg_id,token,digest,expires)

    @staticmethod
    def _valid(row, claim):
        return (row.summary_lease_token==claim.lease_token and
            row.summary_lease_until is not None and row.summary_lease_until>datetime.now(UTC).replace(tzinfo=None) and
            row.summary_upto_msg_id==claim.previous_summary_upto and
            row.summary_from_id==claim.from_msg_id and row.summary_to_id==claim.upto_msg_id and
            row.summary_source_digest==claim.source_digest)

    def complete(self, claim: SummaryClaim, content: str, estimated_tokens: int) -> bool:
        with self.session_factory.begin() as session:
            row=self._row(session,claim.conversation_id,True)
            if not self._valid(row,claim): return False
            session.add(ConversationSummary(conversation_id=claim.conversation_id,
                ordinal=claim.ordinal,from_msg_id=claim.from_msg_id,upto_msg_id=claim.upto_msg_id,
                content=content,estimated_tokens=estimated_tokens,source_digest=claim.source_digest))
            row.summary_upto_msg_id=claim.upto_msg_id
            row.context_version+=1
            row.summary_lease_token=None
            row.summary_lease_until=None
            session.flush()
            return True

    def fail(self, claim: SummaryClaim, reason: str) -> None:
        with self.session_factory.begin() as session:
            row=self._row(session,claim.conversation_id,True)
            if row.summary_lease_token==claim.lease_token:
                row.summary_lease_token=None
                row.summary_lease_until=None

    def renew(self, claim: SummaryClaim) -> bool:
        with self.session_factory.begin() as session:
            row=self._row(session,claim.conversation_id,True)
            if not self._valid(row,claim): return False
            row.summary_lease_until=datetime.now(UTC).replace(tzinfo=None)+timedelta(seconds=120)
            return True
