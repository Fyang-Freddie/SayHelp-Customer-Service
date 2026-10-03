import pytest
from pydantic import ValidationError

from app.config import Settings
from app.prompts import render_extraction_prompt, render_service_system_prompt
from app.schemas import AfterSalesExtraction, ChatRequest, ExtractRequest


@pytest.mark.parametrize("missing", ["CHAT_BASE_URL", "CHAT_MODEL", "CHAT_API_KEY"])
def test_missing_provider_setting_is_rejected(monkeypatch, tmp_path, missing):
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "CHAT_BASE_URL": "https://example.test/v1",
        "CHAT_MODEL": "test-model",
        "CHAT_API_KEY": "top-secret",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv(missing)

    with pytest.raises(ValueError, match=missing) as error:
        Settings.from_env()
    assert "top-secret" not in str(error.value)


def test_env_file_loads_provider_settings_without_overriding_process_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "CHAT_BASE_URL=https://file.example/v1\nCHAT_MODEL=file-model\nCHAT_API_KEY=file-key\n",
        encoding="utf-8",
    )
    for key in ("CHAT_BASE_URL", "CHAT_MODEL", "CHAT_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CHAT_MODEL", "process-model")

    settings = Settings.from_env()

    assert settings.chat_base_url == "https://file.example/v1"
    assert settings.chat_model == "process-model"
    assert settings.chat_api_key == "file-key"
    assert settings.context_token_budget == 4096
    assert settings.response_token_reserve == 512
    assert settings.max_conversations == 100
    assert settings.max_turns_per_conversation == 20


@pytest.mark.parametrize("reserve", ["4096", "4097", "0", "-1"])
def test_response_reserve_must_be_positive_and_below_context(monkeypatch, tmp_path, reserve):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAT_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("CHAT_MODEL", "test-model")
    monkeypatch.setenv("CHAT_API_KEY", "secret")
    monkeypatch.setenv("CONTEXT_TOKEN_BUDGET", "4096")
    monkeypatch.setenv("RESPONSE_TOKEN_RESERVE", reserve)

    with pytest.raises(ValueError, match="RESPONSE_TOKEN_RESERVE"):
        Settings.from_env()


@pytest.mark.parametrize("field,value", [
    ("CONTEXT_TOKEN_BUDGET", "0"),
    ("MAX_CONVERSATIONS", "0"),
    ("MAX_TURNS_PER_CONVERSATION", "-1"),
])
def test_numeric_settings_must_be_positive(monkeypatch, tmp_path, field, value):
    monkeypatch.chdir(tmp_path)
    for key, item in {
        "CHAT_BASE_URL": "https://example.test/v1",
        "CHAT_MODEL": "test-model",
        "CHAT_API_KEY": "secret",
    }.items():
        monkeypatch.setenv(key, item)
    monkeypatch.setenv(field, value)

    with pytest.raises(ValueError, match=field):
        Settings.from_env()


@pytest.mark.parametrize("model,field", [
    (ChatRequest, "message"),
    (ExtractRequest, "description"),
])
def test_whitespace_only_request_text_is_rejected(model, field):
    with pytest.raises(ValidationError):
        model(**{field: " \t\n "})


@pytest.mark.parametrize("request_type", ["退货", "换货", "退款", "维修", "补发", "其他"])
def test_extraction_accepts_each_supported_request_type(request_type):
    value = AfterSalesExtraction(request_type=request_type, order_id=None, expected_solution=None)
    assert value.request_type == request_type
    assert value.order_id is None
    assert value.expected_solution is None


def test_extraction_rejects_unsupported_request_type():
    with pytest.raises(ValidationError):
        AfterSalesExtraction(request_type="投诉", order_id=None, expected_solution=None)


def test_service_prompt_sets_role_and_order_fact_boundaries():
    prompt = render_service_system_prompt()
    assert "客服" in prompt
    assert "订单" in prompt
    assert "不得编造" in prompt or "不要编造" in prompt
    assert "退款" in prompt
    assert "已完成" in prompt or "已处理" in prompt


def test_extraction_prompt_requests_json_with_exact_fields_and_source_description():
    prompt = render_extraction_prompt("订单A123的商品坏了，希望维修")
    assert "JSON" in prompt
    for field in ("order_id", "request_type", "expected_solution"):
        assert field in prompt
    assert "订单A123的商品坏了，希望维修" in prompt
    assert "不要编造" in prompt or "不得编造" in prompt
    for request_type in ("退货", "换货", "退款", "维修", "补发", "其他"):
        assert request_type in prompt


def test_settings_repr_does_not_expose_api_key():
    settings = Settings(
        chat_base_url="https://example.test/v1",
        chat_model="test-model",
        chat_api_key="top-secret-repr-marker",
    )

    rendered = repr(settings)
    assert "top-secret-repr-marker" not in rendered
    assert "chat_base_url" in rendered
