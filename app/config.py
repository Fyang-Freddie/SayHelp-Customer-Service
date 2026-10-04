"""Application configuration loaded from process variables and a local .env file."""

from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class Settings:
    chat_base_url: str
    chat_model: str
    chat_api_key: str = field(repr=False)
    database_url: str | None = field(default=None, repr=False)
    context_token_budget: int = 4096
    response_token_reserve: int = 512
    max_conversations: int = 100
    max_turns_per_conversation: int = 20
    milvus_uri: str = 'http://127.0.0.1:19530'
    bge_cache_dir: str | None = None

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

        base_url = required("CHAT_BASE_URL")
        model = required("CHAT_MODEL")
        api_key = required("CHAT_API_KEY")
        budget = positive_int("CONTEXT_TOKEN_BUDGET", 4096)
        reserve = positive_int("RESPONSE_TOKEN_RESERVE", 512)
        if reserve >= budget:
            raise ValueError("RESPONSE_TOKEN_RESERVE must be below CONTEXT_TOKEN_BUDGET")

        return cls(
            chat_base_url=base_url,
            chat_model=model,
            chat_api_key=api_key,
            context_token_budget=budget,
            response_token_reserve=reserve,
            max_conversations=positive_int("MAX_CONVERSATIONS", 100),
            max_turns_per_conversation=positive_int("MAX_TURNS_PER_CONVERSATION", 20),
            database_url=required("DATABASE_URL"),
            milvus_uri=value("MILVUS_URI") or 'http://127.0.0.1:19530',
            bge_cache_dir=value("BGE_CACHE_DIR"),
        )
