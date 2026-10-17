"""Strict public decisions and private metadata for chapter 6."""
from dataclasses import dataclass, field
from typing import Literal, get_args
from pydantic import BaseModel, ConfigDict, Field, field_validator
from app.workflow_types import Usage

Intent = Literal['物流','订单','商品咨询','退款退货','售后','投诉','闲聊','其他']
INTENTS = get_args(Intent)


class StrictModel(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,frozen=True)


class IntentDecision(StrictModel):
    intent: Intent
    confidence: float = Field(ge=0,le=1,allow_inf_nan=False)


@dataclass(frozen=True)
class IntentOutcome:
    decision: IntentDecision
    reason: str
    model: str
    probabilities: dict[str,float]
    usage: Usage = field(default_factory=Usage)


class Resolution(StrictModel):
    query: str = Field(min_length=1,max_length=4000)
    changed: bool
    unresolved_references: list[str] = Field(max_length=8)


class Expansion(StrictModel):
    queries: list[str] = Field(min_length=3,max_length=3)

    @field_validator('queries')
    @classmethod
    def validate_queries(cls, values):
        if any(not x.strip() or len(x)>1000 for x in values) or len({x.strip() for x in values})!=3:
            raise ValueError('Expansion requires three distinct nonempty queries')
        return values


class Eligibility(StrictModel):
    status: Literal['eligible','ineligible','needs_information']
    explanation: str = Field(min_length=1,max_length=4000)
    evidence_ids: list[str]
    missing_facts: list[str]


class CustomerQuestion(StrictModel):
    question: str = Field(min_length=1,max_length=300)
