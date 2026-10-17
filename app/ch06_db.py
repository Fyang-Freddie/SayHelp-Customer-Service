"""Chapter 6 durable interaction and refund metadata; no payment execution."""
from datetime import datetime
from decimal import Decimal
from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, JSON, DECIMAL, String, text
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base, bigint, table_options
from app.ch05_db import identity_string


class WorkflowInteraction(Base):
    __tablename__='workflow_interactions'
    __table_args__=(table_options('ch06 等待用户交互'),)
    interrupt_id: Mapped[str]=mapped_column(identity_string(64),primary_key=True)
    conversation_id: Mapped[int]=mapped_column(bigint(),ForeignKey('conversations.id',name='fk_interaction_conversation'))
    turn_id: Mapped[str]=mapped_column(identity_string(64),ForeignKey('workflow_turns.turn_id',name='fk_interaction_turn'))
    message_id: Mapped[int]=mapped_column(bigint(),ForeignKey('messages.id',name='fk_interaction_message'))
    kind: Mapped[str]=mapped_column(Enum('order_selection','customer_input','clarification',validate_strings=True))
    payload: Mapped[dict]=mapped_column(JSON)
    checkpoint_id: Mapped[str]=mapped_column(identity_string(64))
    status: Mapped[str]=mapped_column(Enum('waiting','processing','consumed','cancelled',validate_strings=True),server_default='waiting')
    created_at: Mapped[datetime]=mapped_column(DateTime,server_default=text('CURRENT_TIMESTAMP'))


class WorkflowResumeRequest(Base):
    __tablename__='workflow_resume_requests'
    __table_args__=(table_options('ch06 恢复请求幂等记录'),)
    request_id: Mapped[str]=mapped_column(identity_string(128),primary_key=True)
    conversation_id: Mapped[int]=mapped_column(bigint(),ForeignKey('conversations.id',name='fk_resume_conversation'))
    turn_id: Mapped[str]=mapped_column(identity_string(64),ForeignKey('workflow_turns.turn_id',name='fk_resume_turn'))
    interrupt_id: Mapped[str]=mapped_column(identity_string(64),ForeignKey('workflow_interactions.interrupt_id',name='fk_resume_interaction'))
    payload_digest: Mapped[str]=mapped_column(String(64))
    status: Mapped[str]=mapped_column(Enum('processing','complete','cancelled',validate_strings=True),server_default='processing')
    result: Mapped[dict|None]=mapped_column(JSON(none_as_null=True),nullable=True)


class RefundAction(Base):
    __tablename__='refund_actions'
    __table_args__=(table_options('ch06 待确认退款动作'),)
    action_id: Mapped[str]=mapped_column(identity_string(64),primary_key=True)
    conversation_id: Mapped[int]=mapped_column(bigint(),ForeignKey('conversations.id',name='fk_refund_action_conversation'))
    turn_id: Mapped[str]=mapped_column(identity_string(64),ForeignKey('workflow_turns.turn_id',name='fk_refund_action_turn'))
    item: Mapped[dict]=mapped_column(JSON)
    eligibility: Mapped[dict]=mapped_column(JSON)
    fingerprints: Mapped[dict]=mapped_column(JSON)
    status: Mapped[str]=mapped_column(Enum('active','submitted','cancelled',validate_strings=True),server_default='active')
    created_at: Mapped[datetime]=mapped_column(DateTime,server_default=text('CURRENT_TIMESTAMP'))


class RefundRequest(Base):
    __tablename__='refund_requests'
    __table_args__=(Index('uq_refund_action','action_id',unique=True),Index('uq_refund_no','refund_no',unique=True),Index('uq_refund_item','customer_id','order_id','item_id',unique=True),table_options('ch06 本地待处理退款申请'))
    request_id: Mapped[str]=mapped_column(identity_string(128),primary_key=True)
    action_id: Mapped[str]=mapped_column(identity_string(64),ForeignKey('refund_actions.action_id',name='fk_refund_request_action'))
    conversation_id: Mapped[int]=mapped_column(bigint(),ForeignKey('conversations.id',name='fk_refund_request_conversation'))
    refund_no: Mapped[str]=mapped_column(identity_string(64))
    customer_id: Mapped[str]=mapped_column(identity_string(64))
    order_id: Mapped[str]=mapped_column(identity_string(64))
    item_id: Mapped[str]=mapped_column(identity_string(64))
    amount: Mapped[Decimal]=mapped_column(DECIMAL(12,2))
    quantity: Mapped[int]=mapped_column(Integer)
    reason_code: Mapped[str]=mapped_column(identity_string(64))
    payload_digest: Mapped[str]=mapped_column(String(64))
    status: Mapped[str]=mapped_column(Enum('pending',validate_strings=True),server_default='pending')
    created_at: Mapped[datetime]=mapped_column(DateTime,server_default=text('CURRENT_TIMESTAMP'))
