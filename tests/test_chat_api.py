"""HTTP behavior for incremental customer service chat."""

import asyncio
import json

import httpx
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

from app.config import Settings
from app.main import create_app
from app.model_service import ModelService


def settings(**overrides: int) -> Settings:
    values = dict(
        chat_base_url="https://example.invalid/v1",
        chat_model="test-model",
        chat_api_key="test-secret",
    )
    values.update(overrides)
    return Settings(**values)


def parse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.replace("\r\n", "\n").strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


async def post(app, payload: dict) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        return await client.post("/v1/chat/stream", json=payload)


def test_stream_emits_session_each_upstream_chunk_and_done_in_order() -> None:
    class FakeModel:
        async def stream_chat(self, messages):
            yield "你"
            yield "好，"
            yield "请问有什么可以帮您？"

    response = asyncio.run(post(create_app(settings(), FakeModel()), {"message": "你好"}))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_events(response.text)
    assert [name for name, _ in events] == ["session", "token", "token", "token", "done"]
    conversation_id = events[0][1]["conversation_id"]
    assert isinstance(conversation_id, str) and conversation_id
    assert [data for _, data in events[1:4]] == [
        {"text": "你"}, {"text": "好，"}, {"text": "请问有什么可以帮您？"}
    ]
    assert events[-1][1] == {"conversation_id": conversation_id}


def test_second_turn_uses_completed_first_turn_context() -> None:
    class ContextModel:
        async def stream_chat(self, messages):
            if messages[-1].content == "第一问":
                yield "第一答"
            else:
                history = [message.content for message in messages[1:-1]]
                yield "记得" if history == ["第一问", "第一答"] else "忘了"

    app = create_app(settings(), ContextModel())
    first = parse_events(asyncio.run(post(app, {"message": "第一问"})).text)
    conversation_id = first[0][1]["conversation_id"]
    second = parse_events(asyncio.run(post(app, {
        "message": "继续", "conversation_id": conversation_id,
    })).text)

    assert second == [
        ("session", {"conversation_id": conversation_id}),
        ("token", {"text": "记得"}),
        ("done", {"conversation_id": conversation_id}),
    ]


def test_unknown_conversation_id_returns_404_before_stream() -> None:
    class FakeModel:
        async def stream_chat(self, messages):
            yield "unused"

    response = asyncio.run(post(create_app(settings(), FakeModel()), {
        "message": "你好", "conversation_id": "missing",
    }))

    assert response.status_code == 404
    assert "text/event-stream" not in response.headers["content-type"]


def test_oversized_current_input_returns_413_before_stream() -> None:
    class FakeModel:
        async def stream_chat(self, messages):
            yield "unused"

    response = asyncio.run(post(
        create_app(settings(context_token_budget=20, response_token_reserve=10), FakeModel()),
        {"message": "你好"},
    ))

    assert response.status_code == 413
    assert "text/event-stream" not in response.headers["content-type"]


def test_upstream_failure_after_stream_start_emits_safe_error() -> None:
    class FailingModel:
        async def stream_chat(self, messages):
            yield "部分回复"
            raise RuntimeError("test-secret provider credentials")

    response = asyncio.run(post(create_app(settings(), FailingModel()), {"message": "你好"}))

    assert response.status_code == 200
    events = parse_events(response.text)
    assert [name for name, _ in events] == ["session", "token", "error"]
    assert events[1][1] == {"text": "部分回复"}
    assert isinstance(events[-1][1]["message"], str)
    assert "test-secret" not in response.text
    assert "credentials" not in response.text


def test_incomplete_stream_leaves_no_turn_in_next_request() -> None:
    class InterruptedModel:
        def __init__(self):
            self.first = True

        async def stream_chat(self, messages):
            if self.first:
                self.first = False
                yield "partial"
                raise RuntimeError("interrupted")
            yield "clean" if len(messages) == 2 else "dirty"

    app = create_app(settings(), InterruptedModel())
    first = parse_events(asyncio.run(post(app, {"message": "first"})).text)
    conversation_id = first[0][1]["conversation_id"]
    second = parse_events(asyncio.run(post(app, {
        "message": "second", "conversation_id": conversation_id,
    })).text)

    assert [name for name, _ in first] == ["session", "token", "error"]
    assert second[1] == ("token", {"text": "clean"})


def test_model_service_forwards_nonempty_text_chunks_with_configured_model(monkeypatch) -> None:
    import app.model_service as module

    class FakeChatOpenAI:
        def __init__(self, *, base_url, model, api_key):
            assert (base_url, model, api_key) == (
                "https://example.invalid/v1", "test-model", "test-secret"
            )

        async def astream(self, messages):
            yield AIMessageChunk(content="a")
            yield AIMessageChunk(content="")
            yield AIMessageChunk(content="bc")

    monkeypatch.setattr(module, "ChatOpenAI", FakeChatOpenAI)

    async def collect():
        return [chunk async for chunk in ModelService(settings()).stream_chat([])]

    assert asyncio.run(collect()) == ["a", "bc"]

def test_done_boundary_has_already_committed_turn() -> None:
    class ContextModel:
        async def stream_chat(self, messages):
            current = messages[-1].content
            if current == "first":
                yield "first-answer"
            elif current == "second":
                yield "second-answer"
            else:
                history = [message.content for message in messages[1:-1]]
                yield "saved" if history == [
                    "first", "first-answer", "second", "second-answer"
                ] else "lost"

    app = create_app(settings(), ContextModel())
    first = parse_events(asyncio.run(post(app, {"message": "first"})).text)
    conversation_id = first[0][1]["conversation_id"]
    route = next(route for route in app.routes if route.path == "/v1/chat/stream")

    async def receive_done_then_disconnect():
        user = HumanMessage(content="second")
        stream = route.endpoint(
            (conversation_id, user, [SystemMessage(content="system"), user])
        )
        events = [await stream.__anext__() for _ in range(3)]
        await stream.aclose()
        return [event.event for event in events]

    assert asyncio.run(receive_done_then_disconnect()) == ["session", "token", "done"]
    third = parse_events(asyncio.run(post(app, {
        "message": "third", "conversation_id": conversation_id,
    })).text)
    assert third[1] == ("token", {"text": "saved"})


def test_oversized_new_request_does_not_evict_existing_conversation() -> None:
    class FakeModel:
        async def stream_chat(self, messages):
            yield "ok"

    app = create_app(settings(max_conversations=1), FakeModel())
    first = parse_events(asyncio.run(post(app, {"message": "first"})).text)
    conversation_id = first[0][1]["conversation_id"]

    rejected = asyncio.run(post(app, {"message": "x" * 20000}))
    resumed = asyncio.run(post(app, {
        "message": "second", "conversation_id": conversation_id,
    }))

    assert rejected.status_code == 413
    assert resumed.status_code == 200
    assert parse_events(resumed.text)[1] == ("token", {"text": "ok"})
