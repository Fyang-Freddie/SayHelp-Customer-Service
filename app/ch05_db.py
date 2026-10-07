"""Chapter 5 metadata only; the earlier chapter tables retain their DDL."""
from sqlalchemy import Enum, ForeignKey, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base, bigint, table_options


def identity_string(length):
    # Opaque server identities are case-sensitive even on a case-insensitive DB.
    return String(length, collation='utf8mb4_bin')


class WorkflowTurn(Base):
    __tablename__ = 'workflow_turns'
    __table_args__ = (table_options('ch05 工作流轮次'),)
    turn_id: Mapped[str] = mapped_column(identity_string(64), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(bigint(), ForeignKey('conversations.id', name='fk_workflow_turn_conversation'), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default=text("'running'"))
    final_message_id: Mapped[int | None] = mapped_column(bigint(), ForeignKey('messages.id', name='fk_workflow_turn_final_message'), nullable=True)


class WorkflowMessageKey(Base):
    __tablename__ = 'workflow_message_keys'
    __table_args__ = (table_options('ch05 轮次消息幂等键'),)
    turn_id: Mapped[str] = mapped_column(identity_string(64), ForeignKey('workflow_turns.turn_id', name='fk_workflow_message_turn'), primary_key=True)
    position: Mapped[int] = mapped_column(bigint(), primary_key=True, autoincrement=False)
    message_id: Mapped[int] = mapped_column(bigint(), ForeignKey('messages.id', name='fk_workflow_message_message'), nullable=False)


class WorkflowAction(Base):
    __tablename__ = 'workflow_actions'
    __table_args__ = (table_options('ch05 无副作用动作建议'),)
    action_id: Mapped[str] = mapped_column(identity_string(64), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(bigint(), ForeignKey('conversations.id', name='fk_workflow_action_conversation'), nullable=False)
    turn_id: Mapped[str] = mapped_column(identity_string(64), ForeignKey('workflow_turns.turn_id', name='fk_workflow_action_turn'), nullable=False)
    message_id: Mapped[int] = mapped_column(bigint(), ForeignKey('messages.id', name='fk_workflow_action_message'), nullable=False)
    kind: Mapped[str] = mapped_column(Enum('handoff', 'create_ticket', validate_strings=True), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    ticket_type: Mapped[str | None] = mapped_column(Enum('售后', '投诉', '咨询', validate_strings=True), nullable=True)


class WorkflowTicketRequest(Base):
    __tablename__ = 'workflow_ticket_requests'
    __table_args__ = (table_options('ch05 工单确认幂等键'),)
    request_key: Mapped[str] = mapped_column(identity_string(128), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(bigint(), ForeignKey('conversations.id', name='fk_workflow_ticket_conversation'), nullable=False)
    payload_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    ticket_no: Mapped[str] = mapped_column(String(32), ForeignKey('tickets.ticket_no', name='fk_workflow_ticket_ticket'), nullable=False)
