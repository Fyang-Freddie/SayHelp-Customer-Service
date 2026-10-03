"""HTTP and model-boundary behavior for after-sales extraction."""

import asyncio

import httpx
import pytest

from app.config import Settings
from app.main import create_app
from app.model_service import ModelService
from app.schemas import AfterSalesExtraction


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

        def with_structured_output(self, schema, *, method):
            assert schema is AfterSalesExtraction
            assert method == "json_mode"
            return self

        async def ainvoke(self, prompt):
            assert "JSON" in prompt
            assert "A-123" in prompt
            return AfterSalesExtraction(
                order_id="A-123", request_type="退款", expected_solution="原路退款"
            )

    monkeypatch.setattr(module, "ChatOpenAI", FakeChatOpenAI)
    response = asyncio.run(post(
        create_app(settings(), ModelService(settings())),
        {"description": "订单 A-123，请原路退款"},
    ))

    assert response.status_code == 200
    assert response.json() == {
        "order_id": "A-123", "request_type": "退款", "expected_solution": "原路退款"
    }


@pytest.mark.parametrize("upstream_result", [
    None, "", "not JSON", {}, {"request_type": "未知"},
])
def test_extract_rejects_empty_or_malformed_upstream_result(upstream_result, monkeypatch) -> None:
    import app.model_service as module

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema, *, method):
            assert schema is AfterSalesExtraction
            assert method == "json_mode"
            return self

        async def ainvoke(self, prompt):
            assert "JSON" in prompt
            assert "包裹有问题" in prompt
            return upstream_result

    monkeypatch.setattr(module, "ChatOpenAI", FakeChatOpenAI)
    response = asyncio.run(post(
        create_app(settings(), ModelService(settings())), {"description": "包裹有问题"}
    ))

    assert response.status_code == 502
    assert response.json() == {"detail": "Upstream extraction failed"}


def test_extract_rejects_upstream_json_mode_error_without_leaking_secret(monkeypatch) -> None:
    import app.model_service as module

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, schema, *, method):
            assert schema is AfterSalesExtraction
            assert method == "json_mode"
            return self

        async def ainvoke(self, prompt):
            raise RuntimeError("test-secret JSON mode unsupported")

    monkeypatch.setattr(module, "ChatOpenAI", FakeChatOpenAI)
    response = asyncio.run(post(
        create_app(settings(), ModelService(settings())), {"description": "包裹有问题"}
    ))

    assert response.status_code == 502
    assert response.json() == {"detail": "Upstream extraction failed"}
    assert "test-secret" not in response.text
