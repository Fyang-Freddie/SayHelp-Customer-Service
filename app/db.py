"""SQLAlchemy mappings matching the authoritative Chapter 2 MySQL DDL."""

from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, JSON, String, Text, create_engine, text
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


def bigint():
    return BIGINT(unsigned=True).with_variant(Integer, 'sqlite')


def table_options(comment: str):
    return {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4', 'comment': comment}


class Conversation(Base):
    __tablename__ = 'conversations'
    __table_args__ = (Index('idx_user_id', 'user_id'), table_options('客服会话'))
    id: Mapped[int] = mapped_column(bigint(), primary_key=True, autoincrement=True, comment='会话主键')
    user_id: Mapped[str] = mapped_column(String(64), comment='用户标识')
    status: Mapped[str] = mapped_column(Enum('进行中', '已转人工', '已结束', validate_strings=True), server_default='进行中', comment='处理状态')
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP'), comment='开启时间')
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP'), comment='更新时间')


class Message(Base):
    __tablename__ = 'messages'
    __table_args__ = (Index('idx_conversation_id', 'conversation_id'), table_options('会话消息流水'))
    id: Mapped[int] = mapped_column(bigint(), primary_key=True, autoincrement=True, comment='消息主键')
    conversation_id: Mapped[int] = mapped_column(bigint(), ForeignKey('conversations.id', name='fk_messages_conversation'), comment='所属会话')
    role: Mapped[str] = mapped_column(Enum('user', 'assistant', 'tool', validate_strings=True), comment='角色:用户/助手/工具结果')
    content: Mapped[str | None] = mapped_column(Text, comment='消息正文,assistant 纯工具调用时可为空')
    tool_calls: Mapped[list[dict] | None] = mapped_column(JSON(none_as_null=True), comment='assistant 消息带的工具调用申请单')
    tool_call_id: Mapped[str | None] = mapped_column(String(64), comment='tool 消息对应的申请单 id,回灌时对号入座')
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP'), comment='产生时间')


class Faq(Base):
    __tablename__ = 'faq'
    __table_args__ = (Index('idx_category', 'category'), table_options('常见问答'))
    id: Mapped[int] = mapped_column(bigint(), primary_key=True, autoincrement=True, comment='FAQ 主键')
    question: Mapped[str] = mapped_column(String(512), comment='问题')
    answer: Mapped[str] = mapped_column(Text, comment='答案')
    category: Mapped[str] = mapped_column(String(64), comment='分类')
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP'), comment='创建时间')
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP'), comment='更新时间')


class Ticket(Base):
    __tablename__ = 'tickets'
    __table_args__ = (Index('idx_conversation_id', 'conversation_id'), table_options('人工工单'))
    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True, comment='工单号,如 T20260701008')
    conversation_id: Mapped[int] = mapped_column(bigint(), ForeignKey('conversations.id', name='fk_tickets_conversation'), comment='关联会话,可倒查当时聊了什么')
    description: Mapped[str] = mapped_column(Text, comment='问题描述')
    ticket_type: Mapped[str] = mapped_column(Enum('售后', '投诉', '咨询', validate_strings=True), comment='工单类型')
    status: Mapped[str] = mapped_column(Enum('待处理', '已处理', validate_strings=True), server_default='待处理', comment='处理状态')
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text('CURRENT_TIMESTAMP'), comment='创建时间')


def make_session_factory(database_url: str) -> sessionmaker[Session]:
    engine = create_engine(database_url, pool_pre_ping=True)
    return sessionmaker(engine, expire_on_commit=False)
