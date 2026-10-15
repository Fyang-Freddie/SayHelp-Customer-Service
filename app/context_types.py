"""Value objects separating persisted history from transient model context."""
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class BudgetLedger:
    window: int
    prefix: int
    evidence: int
    summary: int
    output: int
    user: int
    safety: int
    react_peak: int
    history: int
    layer1: int
    layer2: int
    estimated: bool = True


@dataclass(frozen=True)
class TurnRecord:
    turn_id: str
    first_msg_id: int
    last_msg_id: int
    messages: tuple[BaseMessage, ...]
    completed: bool = True


@dataclass(frozen=True)
class ContextSnapshot:
    conversation_id: int
    version: int = 0
    summary_upto_msg_id: int | None = None
    layer1_from_msg_id: int | None = None
    summaries: tuple[dict, ...] = ()


@dataclass(frozen=True)
class ContextView:
    snapshot: ContextSnapshot
    layer1: tuple[BaseMessage, ...]
    layer2: tuple[BaseMessage, ...]
    summary_text: str
    ledger: BudgetLedger
    layer1_from_msg_id: int | None
    pending_turns: tuple[TurnRecord, ...] = ()


@dataclass(frozen=True)
class SummaryClaim:
    conversation_id: int
    ordinal: int
    from_msg_id: int
    upto_msg_id: int
    previous_summary_upto: int | None
    lease_token: str
    source_digest: str
    expires_at: datetime
