"""Application configuration loaded from process variables and a local .env file."""

from dataclasses import dataclass, field
import os
import math
import json
import re
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class Settings:
    chat_base_url: str
    chat_model: str
    chat_api_key: str = field(repr=False)
    database_url: str | None = field(default=None, repr=False)
    workflow_checkpoint_path: str = ".runtime/ch05/checkpoints.sqlite"
    agent_model_calls: int = 6
    agent_tool_calls: int = 6
    agent_turn_tokens: int = 12000
    context_token_budget: int = 4096
    response_token_reserve: int = 512
    max_conversations: int = 100
    max_turns_per_conversation: int = 20
    milvus_uri: str = 'http://127.0.0.1:19530'
    bge_cache_dir: str | None = None
    knowledge_min_score: float = 0.55
    knowledge_collection: str = 'knowledge_ch04'
    knowledge_confidence_path: str = 'eval/ch04/confidence.json'
    knowledge_confidence_overrides: dict[str,float] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        file_values = dotenv_values(Path.cwd() / ".env")

        def value(name: str) -> str | None:
            return os.environ.get(name, file_values.get(name))

        def required(name: str) -> str:
            result = value(name)
            if result is None or not result.strip():
                raise ValueError(f"{name} is required")
            return result.strip()

        def positive_int(name: str, default: int) -> int:
            raw = value(name)
            if raw is None:
                return default
            try:
                result = int(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be a positive integer") from error
            if result <= 0:
                raise ValueError(f"{name} must be a positive integer")
            return result

        def agent_name(public: str, legacy: str) -> str:
            # Public Chapter 5 names override the earlier demonstration aliases.
            return public if value(public) is not None else legacy

        def bounded_calls(name: str) -> int:
            count = positive_int(name, 6)
            if count > 6:
                raise ValueError(f'{name} must be at most 6')
            return count

        def minimum_score() -> float:
            try:
                result = float(value('KNOWLEDGE_MIN_SCORE') or '0.55')
            except ValueError:
                raise ValueError('KNOWLEDGE_MIN_SCORE must be finite and within [-1, 1]') from None
            if not math.isfinite(result) or not -1 <= result <= 1:
                raise ValueError('KNOWLEDGE_MIN_SCORE must be finite and within [-1, 1]')
            return result

        collection = value('KNOWLEDGE_COLLECTION') or 'knowledge_ch04'
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}',collection) or collection=='knowledge':
            raise ValueError('KNOWLEDGE_COLLECTION must name an independent chapter 4 collection')
        try:
            overrides=json.loads(value('KNOWLEDGE_CONFIDENCE_OVERRIDES') or '{}')
            if not isinstance(overrides,dict) or not set(overrides)<={'dense','bm25','hybrid','hybrid_rerank'} or any(type(v) not in (int,float) or not math.isfinite(v) for v in overrides.values()): raise ValueError()
        except (TypeError,ValueError):
            raise ValueError('KNOWLEDGE_CONFIDENCE_OVERRIDES must contain explicit finite strategy thresholds') from None
        base_url = required("CHAT_BASE_URL")
        model = required("CHAT_MODEL")
        api_key = required("CHAT_API_KEY")
        budget = positive_int("CONTEXT_TOKEN_BUDGET", 4096)
        reserve = positive_int("RESPONSE_TOKEN_RESERVE", 512)
        if reserve >= budget:
            raise ValueError("RESPONSE_TOKEN_RESERVE must be below CONTEXT_TOKEN_BUDGET")

        token_name = agent_name('AGENT_TURN_TOKEN_BUDGET', 'AGENT_TURN_TOKENS')
        turn_tokens = positive_int(token_name, 12000)
        if turn_tokens <= reserve:
            raise ValueError(f'{token_name} must exceed RESPONSE_TOKEN_RESERVE')
        checkpoint_path = value('WORKFLOW_CHECKPOINT_PATH') or '.runtime/ch05/checkpoints.sqlite'
        if not checkpoint_path.strip():
            raise ValueError('WORKFLOW_CHECKPOINT_PATH must be nonempty')

        return cls(
            chat_base_url=base_url,
            chat_model=model,
            chat_api_key=api_key,
            workflow_checkpoint_path=checkpoint_path,
            agent_model_calls=bounded_calls(agent_name('AGENT_MAX_MODEL_CALLS', 'AGENT_MODEL_CALLS')),
            agent_tool_calls=bounded_calls(agent_name('AGENT_MAX_TOOL_CALLS', 'AGENT_TOOL_CALLS')),
            agent_turn_tokens=turn_tokens,
            context_token_budget=budget,
            response_token_reserve=reserve,
            max_conversations=positive_int("MAX_CONVERSATIONS", 100),
            max_turns_per_conversation=positive_int("MAX_TURNS_PER_CONVERSATION", 20),
            database_url=required("DATABASE_URL"),
            milvus_uri=value("MILVUS_URI") or 'http://127.0.0.1:19530',
            bge_cache_dir=value("BGE_CACHE_DIR"),
            knowledge_min_score=minimum_score(),
            knowledge_collection=collection,
            knowledge_confidence_path=value("KNOWLEDGE_CONFIDENCE_PATH") or "eval/ch04/confidence.json",
            knowledge_confidence_overrides=overrides,
        )
