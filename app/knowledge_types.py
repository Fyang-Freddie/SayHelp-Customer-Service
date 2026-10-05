"""Shared contracts for retrieval, immutable evidence and online/evaluation routing."""
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

RetrievalStrategy = Literal['dense', 'bm25', 'hybrid', 'hybrid_rerank']
Intent = Literal['knowledge', 'operation', 'chitchat', 'clarify']


class KnowledgeFilters(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    category: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)] | None = None
    product_category: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)] | None = None
    content_type: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)] | None = None

    @field_validator('category', 'product_category', 'content_type', mode='before')
    @classmethod
    def reject_explicit_null(cls, value):
        if value is None:
            raise ValueError('Filter values must be nonempty strings')
        return value


@dataclass(frozen=True)
class PreparedQuery:
    raw_question: str
    standard_question: str
    search_text: str
    synonyms: list[str] = field(default_factory=list)
    intent: Intent = 'knowledge'
    clarification: str | None = None
    preserved_models: list[str] = field(default_factory=list)
    preserved_numbers: list[str] = field(default_factory=list)
    preserved_negations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceChunk:
    id: int
    question: str
    answer: str
    category: str
    product_category: str | None
    content_type: str | None
    section_path: str | None
    source_file: str | None
    source_start_line: int | None
    source_end_line: int | None
    source_digest: str | None


class Citation(TypedDict):
    n: int
    chunk_id: str
    section_path: str | None
    question: str
    answer: str
    source_file: str | None
    source_start_line: int | None
    source_end_line: int | None
    source_digest: str | None
    source_url: str | None


@dataclass(frozen=True)
class StageHit:
    chunk_id: int
    rank: int
    score: float
    stage: str


@dataclass(frozen=True)
class RankedChunk:
    chunk: EvidenceChunk
    rank: int
    score: float


@dataclass(frozen=True)
class StoreResult:
    hits: list[StageHit]
    stage_hits: dict[str, list[StageHit]]
    entities: dict[int, dict]


@dataclass(frozen=True)
class CorpusSnapshot:
    files: dict[str, str]
    chunks: dict[int, EvidenceChunk]
    payload_digests: dict[int, str]
    corpus_digest: str


@dataclass(frozen=True)
class BuildSummary:
    indexed: int
    reused: int
    removed: int
    verified: int
    corpus_digest: str


@dataclass(frozen=True)
class RetrievalResult:
    strategy: RetrievalStrategy
    query: PreparedQuery
    candidates: list[RankedChunk]
    ranked: list[RankedChunk]
    stage_hits: dict[str, list[StageHit]]


@dataclass(frozen=True)
class PromptEvidence:
    messages: list
    citations: list[Citation]
    relevance_order: list[int]
    prompt_tokens: int


@dataclass(frozen=True)
class ConfidenceDecision:
    sufficient: bool
    reason: str


@dataclass(frozen=True)
class KnowledgeAnswer:
    useful: bool
    answer: str
    reason: str
    pool_source: Literal['retrieval_low_conf', 'self_check'] | None
    citations: list[Citation]
