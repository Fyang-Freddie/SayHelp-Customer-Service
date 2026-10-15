"""One Chinese-aware estimate and auditable per-request window accounting."""
from dataclasses import asdict
from math import ceil
from collections.abc import Sequence
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately
from app.context_types import BudgetLedger
from app.history import InputBudgetExceeded


def estimate_tokens(messages: Sequence[BaseMessage], tools: Sequence[dict] = ()) -> int:
    return ceil(count_tokens_approximately(list(messages), chars_per_token=1.0,
                extra_tokens_per_message=3.0, tools=list(tools) or None))


def calculate_budget(*, window: int, prefix: int, evidence: int, summary: int,
                     output: int, user: int, steps: int, selection: int,
                     tool_results: int, protocol: int, target_turns: int,
                     steady: int) -> BudgetLedger:
    values = locals()
    if any(type(v) is not int or v < 0 for v in values.values()):
        raise ValueError('Budget values must be nonnegative integers')
    if min(window, output, steps, selection, tool_results, target_turns, steady) < 1:
        raise ValueError('Window, output, steps and historical target must be positive')
    safety = ceil(window * .05)
    peak = steps * (selection + tool_results + max(32, protocol))
    available = max(0, window - prefix - evidence - summary - output - user - safety - peak)
    history = min(target_turns * steady, available)
    layer1 = history * 7 // 10
    return BudgetLedger(window, prefix, evidence, summary, output, user,
                        safety, peak, history, layer1, history-layer1)


def validate_startup(ledger: BudgetLedger, steady: int = 1024) -> None:
    if type(steady) is not int or steady<1:
        raise ValueError('Steady turn estimate must be positive')
    if ledger.history < steady:
        raise InputBudgetExceeded('上下文预算不足: ' + str(asdict(ledger)))


def validate_request(messages, tools, output: int, window: int) -> int:
    tokens = estimate_tokens(messages, tools)
    if tokens + output + ceil(window * .05) > window:
        raise InputBudgetExceeded('上下文预算不足: actual request exceeds model window')
    return tokens


def settings_budget(settings, *, prefix: int, evidence: int | None = None,
                    summary: int = 0, user: int | None = None, protocol: int = 32):
    return calculate_budget(window=settings.context_token_budget, prefix=prefix,
        evidence=settings.rerank_top_k * 512 if evidence is None else evidence,
        summary=max(settings.summary_token_reserve, summary),
        output=settings.response_token_reserve,
        user=settings.max_user_input_tokens if user is None else user,
        steps=settings.max_agent_steps, selection=512,
        tool_results=settings.tool_result_max_tokens, protocol=protocol,
        target_turns=settings.history_target_turns, steady=settings.history_steady_tokens)
