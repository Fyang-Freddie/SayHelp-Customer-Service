"""HTTP and model-boundary behavior for after-sales extraction."""

import asyncio
import json
from functools import partial
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import httpx
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from app.config import Settings
from app.main import create_app
from app.model_service import ModelService
from app.schemas import AfterSalesExtraction


@pytest.fixture(autouse=True)
def inject_test_storage(monkeypatch):
    """Extraction never touches storage, but app composition requires it."""
    engine = create_engine('sqlite:///:memory:')
    monkeypatch.setattr(__import__(__name__), 'create_app',
                        partial(create_app, session_factory=sessionmaker(engine)))
    try:
        yield
    finally:
        engine.dispose()


def settings() -> Settings:
    return Settings(
        chat_base_url="https://example.invalid/v1",
        chat_model="test-model",
        chat_api_key="test-secret",
    )


async def post(app, payload: dict) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        return await client.post("/v1/aftersales/extract", json=payload)


def test_extract_returns_three_validated_fields() -> None:
    class FakeModel:
        async def extract_after_sales(self, description):
            return AfterSalesExtraction(
                order_id="A-123", request_type="退款", expected_solution="原路退款"
            )

    response = asyncio.run(post(
        create_app(settings(), FakeModel()), {"description": "订单 A-123，请原路退款"}
    ))

    assert response.status_code == 200
    assert response.json() == {
        "order_id": "A-123", "request_type": "退款", "expected_solution": "原路退款"
    }


def test_extract_preserves_null_for_missing_order_and_solution() -> None:
    class FakeModel:
        async def extract_after_sales(self, description):
            return AfterSalesExtraction(request_type="其他")

    response = asyncio.run(post(
        create_app(settings(), FakeModel()), {"description": "包裹有问题"}
    ))

    assert response.status_code == 200
    assert response.json() == {
        "order_id": None, "request_type": "其他", "expected_solution": None
    }


def test_extract_rejects_whitespace_description_before_calling_model() -> None:
    class FakeModel:
        async def extract_after_sales(self, description):
            raise AssertionError("model should not be called")

    response = asyncio.run(post(
        create_app(settings(), FakeModel()), {"description": " \t\n "}
    ))

    assert response.status_code == 422


def test_extract_accepts_valid_structured_model_result(monkeypatch) -> None:
    import app.model_service as module

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema, *, method, include_raw):
            assert schema is AfterSalesExtraction
            assert method == "json_mode"
            assert include_raw is True
            return self

        async def ainvoke(self, prompt):
            assert "JSON" in prompt
            assert "A-123" in prompt
            raw = '{"order_id":"A-123","request_type":"\u9000\u6b3e","expected_solution":"\u539f\u8def\u9000\u6b3e"}'
            return {
                "raw": AIMessage(content=raw),
                "parsed": AfterSalesExtraction(
                    order_id="A-123", request_type="\u9000\u6b3e",
                    expected_solution="\u539f\u8def\u9000\u6b3e",
                ),
                "parsing_error": None,
            }

    monkeypatch.setattr(module, "ChatOpenAI", FakeChatOpenAI)
    response = asyncio.run(post(
        create_app(settings(), ModelService(settings())),
        {"description": "order A-123: refund"},
    ))

    assert response.status_code == 200
    assert response.json() == {
        "order_id": "A-123", "request_type": "\u9000\u6b3e",
        "expected_solution": "\u539f\u8def\u9000\u6b3e",
    }


@pytest.mark.parametrize("upstream_content", [
    "", "not JSON", "{}", '{"request_type":"unknown"}',
])
def test_extract_rejects_empty_or_malformed_upstream_result(upstream_content, monkeypatch) -> None:
    import app.model_service as module

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema, *, method, include_raw):
            assert schema is AfterSalesExtraction
            assert method == "json_mode"
            assert include_raw is True
            return self

        async def ainvoke(self, prompt):
            assert "JSON" in prompt
            return {
                "raw": AIMessage(content=upstream_content),
                "parsed": None,
                "parsing_error": ValueError("invalid upstream"),
            }

    monkeypatch.setattr(module, "ChatOpenAI", FakeChatOpenAI)
    response = asyncio.run(post(
        create_app(settings(), ModelService(settings())), {"description": "package issue"}
    ))

    assert response.status_code == 502
    assert response.json() == {"detail": "Upstream extraction failed"}


def test_extract_rejects_upstream_json_mode_error_without_leaking_secret(monkeypatch) -> None:
    import app.model_service as module

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema, *, method, include_raw):
            assert schema is AfterSalesExtraction
            assert method == "json_mode"
            assert include_raw is True
            return self

        async def ainvoke(self, prompt):
            raise RuntimeError("test-secret JSON mode unsupported")

    monkeypatch.setattr(module, "ChatOpenAI", FakeChatOpenAI)
    response = asyncio.run(post(
        create_app(settings(), ModelService(settings())), {"description": "package issue"}
    ))

    assert response.status_code == 502
    assert response.json() == {"detail": "Upstream extraction failed"}
    assert "test-secret" not in response.text


@pytest.mark.parametrize(("raw_content", "expected_status"), [
    ('{"order_id":"A-123","request_type":"\u9000\u6b3e","expected_solution":"\u539f\u8def\u9000\u6b3e"', 502),
    ('{"order_id":"A-123","request_type":"\u9000\u6b3e","expected_solution":"\u539f\u8def\u9000\u6b3e"}', 200),
    ('{"request_type":"invalid"}', 502),
    ('{"request_type":"\u9000\u6b3e",}', 502),
    ('{"request_type":"\u9000\u6b3e","noise":NaN}', 502),
    ('{"request_type":"\u9000\u6b3e","noise":Infinity}', 502),
    ('{"request_type":"\u9000\u6b3e","noise":-Infinity}', 502),
])
def test_extract_validates_raw_json_from_real_chat_pipeline(
    monkeypatch, raw_content: str, expected_status: int
) -> None:
    import app.model_service as module

    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": raw_content,
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    async def run() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            monkeypatch.setattr(module, "ChatOpenAI", lambda **kwargs: ChatOpenAI(
                **kwargs, http_async_client=client, http_socket_options=(),
            ))
            return await post(
                create_app(settings(), ModelService(settings())),
                {"description": "订单 A-123，请原路退款"},
            )

    response = asyncio.run(run())
    assert response.status_code == expected_status
    if expected_status == 200:
        assert response.json() == {
            "order_id": "A-123", "request_type": "\u9000\u6b3e",
            "expected_solution": "\u539f\u8def\u9000\u6b3e",
        }
    else:
        assert response.json() == {"detail": "Upstream extraction failed"}
