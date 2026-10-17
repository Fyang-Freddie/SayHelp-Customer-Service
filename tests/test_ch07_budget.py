"""Catch inconsistent window accounting and evaluation quota overspending."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def budget(window=18000, **changes):
    from app.context_budget import calculate_budget
    values = dict(window=window, prefix=3666, evidence=2560, summary=512,
                  output=2000, user=2000, steps=3, selection=512,
                  tool_results=1200, protocol=32, target_turns=32, steady=1024)
    return calculate_budget(**(values | changes))


@pytest.mark.parametrize('window,want', [(18000, (1130, 791, 339)),
                                      (65536, (32768, 22937, 9831))])
def test_budget_ledger(window, want):
    ledger = budget(window)
    assert (ledger.history, ledger.layer1, ledger.layer2) == want


def test_budget_uses_actual_input_and_overhead():
    assert budget(user=50, evidence=1500).history == 4140
    assert budget(protocol=64).history == 1034
    assert budget(summary=1512).history == 130
    assert budget(window=8000).history == 0


def test_counting_includes_chinese_emoji_and_tool_arguments():
    from app.context_budget import estimate_tokens, validate_request
    from app.history import InputBudgetExceeded
    short = [HumanMessage('订单📦')]
    long = [HumanMessage('订单📦' * 100)]
    assert estimate_tokens(long) > estimate_tokens(short) + 290
    calls = [AIMessage('', tool_calls=[dict(name='query_order', args={'order_id':'123'*200}, id='call-1', type='tool_call')]),
             ToolMessage('查询完成', tool_call_id='call-1')]
    assert estimate_tokens(calls) > 600
    with pytest.raises(InputBudgetExceeded):
        validate_request(long, [], output=100, window=200)


def test_startup_budget_rejects_zero_history():
    from app.context_budget import validate_startup
    from app.history import InputBudgetExceeded
    with pytest.raises(InputBudgetExceeded, match='上下文预算不足'):
        validate_startup(budget(window=8000))
    with pytest.raises(InputBudgetExceeded, match='上下文预算不足'):
        validate_startup(budget(summary=1512))


def test_evaluation_budget_is_atomic_and_survives_reopen(tmp_path):
    from app.ch07_evaluation import EvaluationBudget, EvaluationLimitExceeded
    path = tmp_path/'usage.sqlite'
    meter = EvaluationBudget(path, limits={'chat':3}, buckets={'test':3})
    def reserve(_):
        try:
            return meter.reserve('chat','test')
        except EvaluationLimitExceeded:
            return None
    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = [x for x in executor.map(reserve, range(12)) if x]
    assert len(ids) == 3
    meter.finish(ids[0], 'failed', {})
    reopened = EvaluationBudget(path, limits={'chat':3}, buckets={'test':3})
    with pytest.raises(EvaluationLimitExceeded):
        reopened.reserve('chat','test')


def test_new_config_overrides_legacy_without_loading_secrets(monkeypatch, tmp_path):
    from app.config import Settings
    monkeypatch.chdir(tmp_path)
    for key,value in dict(CHAT_BASE_URL='http://localhost', CHAT_MODEL='test', CHAT_API_KEY='test',
                          DATABASE_URL='sqlite://', MODEL_CONTEXT_WINDOW='18000',
                          CONTEXT_TOKEN_BUDGET='4096', MAX_OUTPUT_TOKENS='2000',
                          RESPONSE_TOKEN_RESERVE='512', MAX_AGENT_STEPS='3').items():
        monkeypatch.setenv(key,value)
    settings=Settings.from_env()
    assert settings.context_token_budget == 18000
    assert settings.response_token_reserve == 2000
    assert settings.max_agent_steps == 3


def test_startup_rejects_provider_capacity_below_configured_window():
    from app.context_budget import validate_provider_capacity
    from app.config import Settings
    from app.history import InputBudgetExceeded
    settings=Settings('https://api.deepseek.com','deepseek-v4-flash','test',
        context_token_budget=18000,response_token_reserve=2000,layered_context=True)
    models={'data':[{'id':'deepseek-flash','context_window':8192,'max_output_tokens':2048}]}
    with pytest.raises(InputBudgetExceeded,match='上下文预算不足'):
        validate_provider_capacity(settings,models)
    good={'data':[{'id':'deepseek-flash','context_window':1000000,'max_output_tokens':2048}]}
    assert validate_provider_capacity(settings,good)==1000000


def test_capacity_probe_uses_model_endpoint_and_checks_declared_output(monkeypatch):
    import asyncio
    import httpx
    from app.context_budget import check_provider_capacity
    from app.config import Settings
    from app.history import InputBudgetExceeded
    seen=[]
    def handler(request):
        seen.append((str(request.url),request.headers.get('authorization')))
        return httpx.Response(200,json={'data':[{'id':'deepseek-flash',
            'context_window':32768,'max_output_tokens':1024}]})
    original=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(
        transport=httpx.MockTransport(handler),**kwargs))
    settings=Settings('https://api.deepseek.com/v1','deepseek-v4-flash','fixture-key',
        context_token_budget=18000,response_token_reserve=2000,layered_context=True)
    with pytest.raises(InputBudgetExceeded,match='上下文预算不足'):
        asyncio.run(check_provider_capacity(settings))
    assert seen==[('https://api.deepseek.com/v1/models','Bearer fixture-key')]


def test_explicit_capacity_override_is_labeled_unverified():
    import asyncio
    from app.context_budget import check_provider_capacity
    from app.config import Settings
    settings=Settings('https://example.invalid/v1','fixture-model','fixture-key',
        context_token_budget=18000,response_token_reserve=2000,
        provider_context_window=32768,layered_context=True)
    assert asyncio.run(check_provider_capacity(settings))==(32768,'configured_unverified')
