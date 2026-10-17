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


def validate_provider_capacity(settings, models: dict) -> int:
    aliases={'deepseek-v4-flash':'deepseek-flash',
             'deepseek-v4-flash-vision-exp':'deepseek-flash'}
    name=aliases.get(settings.chat_model,settings.chat_model)
    rows=models.get('data',[]) if isinstance(models,dict) else []
    row=next((item for item in rows if isinstance(item,dict) and item.get('id')==name),None)
    if row is None or type(row.get('context_window')) is not int or row['context_window']<1:
        raise InputBudgetExceeded('上下文预算不足: 模型容量不可验证，请配置 MODEL_PROVIDER_CONTEXT_WINDOW')
    capacity=row['context_window']
    output=row.get('max_output_tokens')
    if settings.context_token_budget>capacity or (type(output) is int and settings.response_token_reserve>output):
        raise InputBudgetExceeded(f'上下文预算不足: 配置窗口 {settings.context_token_budget} / 输出 {settings.response_token_reserve} 超过模型容量 {capacity} / {output}')
    return capacity


async def check_provider_capacity(settings) -> tuple[int,str]:
    declared=settings.provider_context_window
    if declared is not None:
        capacity=validate_provider_capacity(settings,{'data':[{
            'id':{'deepseek-v4-flash':'deepseek-flash',
                  'deepseek-v4-flash-vision-exp':'deepseek-flash'}.get(settings.chat_model,settings.chat_model),
            'context_window':declared}]})
        return capacity,'configured_unverified'
    import httpx
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response=await client.get(settings.chat_base_url.rstrip('/')+'/models',
                headers={'Authorization':'Bearer '+settings.chat_api_key})
            response.raise_for_status()
            capacity=validate_provider_capacity(settings,response.json())
    except (httpx.HTTPError,ValueError,KeyError,TypeError) as error:
        raise InputBudgetExceeded('上下文预算不足: 无法取得模型容量，请检查 /models 或配置 MODEL_PROVIDER_CONTEXT_WINDOW') from error
    return capacity,'provider_models_api'


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
