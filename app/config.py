"""Application configuration loaded from process variables and a local .env file."""

from dataclasses import dataclass, field
import os
import math
import json
import re
import logging
from pathlib import Path

from dotenv import dotenv_values


JEV_DISCOVERY_MODEL = "jev-latest"
POLICY_SEARCH_CONCURRENCY = 2
POLICY_RRF_K = 60
POLICY_CANDIDATES = 50
POLICY_TOP_K = 10


@dataclass(frozen=True)
class Settings:
    chat_base_url: str
    chat_model: str
    chat_api_key: str = field(repr=False)
    database_url: str | None = field(default=None, repr=False)
    jev_api_key: str = field(default='', repr=False)
    jev_model: str = ''
    jev_base_url: str = 'https://api.typesafe.ai/v1'
    jev_timeout: float = 10.0
    intent_policy_path: str = 'eval/ch06/intent-policy.json'
    ch06_knowledge_collection: str = 'knowledge_ch06'
    ch06_knowledge_manifest: str = 'eval/ch06/corpus.json'
    ch06_knowledge_policy: str = 'eval/ch06/knowledge-policy.json'
    workflow_checkpoint_path: str = ".runtime/ch05/checkpoints.sqlite"
    agent_model_calls: int = 6
    agent_tool_calls: int = 6
    agent_turn_tokens: int = 12000
    context_token_budget: int = 4096
    response_token_reserve: int = 512
    max_user_input_tokens: int = 2000
    max_agent_steps: int = 3
    tool_result_max_tokens: int = 1200
    rerank_top_k: int = 5
    summary_token_reserve: int = 512
    history_target_turns: int = 32
    history_steady_tokens: int = 1024
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
        def context_name(public, legacy):
            if value(legacy) is not None:
                logging.getLogger(__name__).info('context_config legacy=%s selected=%s',
                    legacy, public if value(public) is not None else legacy)
            return public if value(public) is not None else legacy
        budget = positive_int(context_name('MODEL_CONTEXT_WINDOW', 'CONTEXT_TOKEN_BUDGET'), 65536)
        reserve = positive_int(context_name('MAX_OUTPUT_TOKENS', 'RESPONSE_TOKEN_RESERVE'), 2000)
        steps = positive_int('MAX_AGENT_STEPS', 3)
        if steps > 5:
            raise ValueError('MAX_AGENT_STEPS must be at most 5 (plus final generation)')
        if reserve >= budget:
            raise ValueError("RESPONSE_TOKEN_RESERVE must be below CONTEXT_TOKEN_BUDGET")

        token_name = agent_name('AGENT_TURN_TOKEN_BUDGET', 'AGENT_TURN_TOKENS')
        turn_tokens = positive_int(token_name, (steps + 5) * budget)
        if turn_tokens <= reserve:
            raise ValueError(f'{token_name} must exceed RESPONSE_TOKEN_RESERVE')
        checkpoint_path = value('WORKFLOW_CHECKPOINT_PATH') or '.runtime/ch05/checkpoints.sqlite'
        if not checkpoint_path.strip():
            raise ValueError('WORKFLOW_CHECKPOINT_PATH must be nonempty')

        return cls(
            chat_base_url=base_url,
            chat_model=model,
            chat_api_key=api_key,
            jev_api_key=value('TYPESAFE_API_KEY') or '',
            jev_model=value('JEV_MODEL') or '',
            jev_base_url=value('JEV_BASE_URL') or 'https://api.typesafe.ai/v1',
            jev_timeout=positive_int('JEV_TIMEOUT_SECONDS',10),
            intent_policy_path=value('INTENT_POLICY_PATH') or 'eval/ch06/intent-policy.json',
            ch06_knowledge_collection=value('CH06_KNOWLEDGE_COLLECTION') or 'knowledge_ch06',
            ch06_knowledge_manifest=value('CH06_KNOWLEDGE_MANIFEST') or 'eval/ch06/corpus.json',
            ch06_knowledge_policy=value('CH06_KNOWLEDGE_POLICY') or 'eval/ch06/knowledge-policy.json',
            workflow_checkpoint_path=checkpoint_path,
            agent_model_calls=bounded_calls(agent_name('AGENT_MAX_MODEL_CALLS', 'AGENT_MODEL_CALLS')),
            agent_tool_calls=bounded_calls(agent_name('AGENT_MAX_TOOL_CALLS', 'AGENT_TOOL_CALLS')),
            agent_turn_tokens=turn_tokens,
            context_token_budget=budget,
            response_token_reserve=reserve,
            max_user_input_tokens=positive_int('MAX_USER_INPUT_TOKENS', 2000),
            max_agent_steps=steps,
            tool_result_max_tokens=positive_int('TOOL_RESULT_MAX_TOKENS', 1200),
            rerank_top_k=positive_int('RERANK_TOP_K', 5),
            summary_token_reserve=positive_int('SUMMARY_TOKEN_RESERVE', 512),
            history_target_turns=positive_int('HISTORY_TARGET_TURNS', 32),
            history_steady_tokens=positive_int('HISTORY_STEADY_TOKENS', 1024),
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
