"""Additional mapping over conversations without changing legacy ORM loaders."""
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from app.db import bigint, table_options


class ContextBase(DeclarativeBase):
    pass


class ConversationContext(ContextBase):
    __tablename__='conversations'
    id: Mapped[int] = mapped_column(bigint(), primary_key=True)
    summary_upto_msg_id: Mapped[int | None] = mapped_column(bigint(), nullable=True)
    layer1_from_msg_id: Mapped[int | None] = mapped_column(bigint(), nullable=True)
    context_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default='0')
    summary_lease_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    summary_lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    summary_from_id: Mapped[int | None] = mapped_column(bigint(), nullable=True)
    summary_to_id: Mapped[int | None] = mapped_column(bigint(), nullable=True)
    summary_source_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)


class ConversationSummary(ContextBase):
    __tablename__='conversation_summaries'
    __table_args__=(UniqueConstraint('conversation_id','ordinal',name='uq_summary_ordinal'),
                   UniqueConstraint('conversation_id','from_msg_id','upto_msg_id',name='uq_summary_range'),
                   table_options('ch07 只追加会话事实摘要'))
    id: Mapped[int] = mapped_column(bigint(),primary_key=True,autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(bigint(),ForeignKey('conversations.id'),nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer,nullable=False)
    from_msg_id: Mapped[int] = mapped_column(bigint(),nullable=False)
    upto_msg_id: Mapped[int] = mapped_column(bigint(),nullable=False)
    content: Mapped[str] = mapped_column(Text,nullable=False)
    estimated_tokens: Mapped[int] = mapped_column(Integer,nullable=False)
    source_digest: Mapped[str] = mapped_column(String(64),nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime,nullable=False,server_default=text('CURRENT_TIMESTAMP'))
