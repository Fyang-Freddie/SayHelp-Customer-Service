"""Mappings for the supplied Chapter 4 tables and independent index progress."""
from datetime import datetime
from sqlalchemy import DateTime, Enum, ForeignKey, Index, JSON, String, Text, text
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base, bigint, table_options
from app.knowledge_db import KnowledgeChunk

class LowConfidenceQuestion(Base):
    __tablename__ = 'low_confidence_questions'
    __table_args__ = (
        Index('idx_source', 'source'),
        Index('idx_created_at', 'created_at'),
        table_options('低置信度问题池'),
    )
    id: Mapped[int] = mapped_column(bigint(), primary_key=True, autoincrement=True, nullable=False, comment='主键')
    conversation_id: Mapped[int | None] = mapped_column(bigint(), ForeignKey('conversations.id', name='fk_lcq_conversation'), nullable=True, comment='来源会话')
    raw_question: Mapped[str] = mapped_column(Text, nullable=False, comment='用户原话,带情绪口语')
    source: Mapped[str] = mapped_column(Enum('retrieval_low_conf','self_check','user_feedback', validate_strings=True), nullable=False, comment='入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决')
    reason: Mapped[str | None] = mapped_column(Text, nullable=True, comment='判不能的原因,留作复盘')
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text('CURRENT_TIMESTAMP'), comment='入池时间')


class FaithCase(Base):
    __tablename__ = 'faith_cases'
    __table_args__ = (
        Index('uk_eval_id', 'eval_id', unique=True),
        Index('idx_status', 'status'),
        Index('idx_last_seen_at', 'last_seen_at'),
        table_options('ch04 忠实度编造个案台账'),
    )
    id: Mapped[int] = mapped_column(bigint(), primary_key=True, autoincrement=True, nullable=False, comment='主键')
    eval_id: Mapped[str] = mapped_column(String(16), nullable=False, comment='评估集题号,如 A43;一题一行')
    bucket: Mapped[str] = mapped_column(String(24), nullable=False, comment='题目所属桶:A_policy / B_model / C_colloquial / E_multi')
    query: Mapped[str] = mapped_column(String(512), nullable=False, comment='用户问题原文')
    strategy: Mapped[str] = mapped_column(String(24), nullable=False, server_default=text("'hybrid_rerank'"), comment='产出这条答案的检索策略')
    answer: Mapped[str] = mapped_column(Text, nullable=False, comment='被判编造的那版生成答案原文')
    reason: Mapped[str] = mapped_column(Text, nullable=False, comment='裁判给的理由:编在哪一句')
    citations: Mapped[list[dict] | None] = mapped_column(JSON(none_as_null=True), nullable=True, comment='这一轮喂给模型的 Top-K 证据全集快照:[{n,chunk_id,section_path,question,answer}];答案里的角标 [n] 就是这份列表的序号,答案通常只引用其中两三条;老数据没记为 NULL')
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True, comment='判这条的裁判模型')
    status: Mapped[str] = mapped_column(Enum('未解决','已解决','无需解决', validate_strings=True), nullable=False, server_default=text("'未解决'"), comment='处置状态,人工点按钮改')
    seen_count: Mapped[int] = mapped_column(INTEGER(unsigned=True), nullable=False, server_default=text('1'), comment='被判编造的累计次数(跨轮)')
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text('CURRENT_TIMESTAMP'), comment='第一次被判编造的时间')
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text('CURRENT_TIMESTAMP'), comment='最近一次被判编造的时间')
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True, comment='处置说明:标已解决要写清怎么解决的,标无需解决要写清为什么不用改;退回未解决时清空。空着的处置在台账上等于没有交代')
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment='最近一次被标为已解决/无需解决的时间;复发后仍保留,用来标「复发」')


class KnowledgeIndexState(Base):
    __tablename__ = 'knowledge_index_states'
    __table_args__ = (table_options('ch04 独立集合构建进度'),)
    collection_name: Mapped[str] = mapped_column(String(128), primary_key=True, comment='索引集合')
    chunk_id: Mapped[int] = mapped_column(bigint(), ForeignKey('knowledge_chunks.id', name='fk_index_chunk'), primary_key=True, comment='权威 chunk')
    payload_digest: Mapped[str] = mapped_column(String(64), nullable=False, comment='文本与过滤元数据指纹')
    status: Mapped[str] = mapped_column(Enum('pending','done', validate_strings=True), nullable=False, server_default=text("'pending'"), comment='新集合写入状态')
    error: Mapped[str | None] = mapped_column(Text, nullable=True, comment='安全错误原因')
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP'), comment='状态更新时间')
