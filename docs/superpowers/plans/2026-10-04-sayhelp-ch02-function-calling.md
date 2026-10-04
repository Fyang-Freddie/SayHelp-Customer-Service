# SayHelp Chapter 2 Function Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one customer message in the existing SSE chat invoke at most one model-selected business tool, persist its complete trace in MySQL, and stream the final answer with a visible tool badge.

**Architecture:** The existing FastAPI route delegates to a chat service that loads SQLAlchemy-backed history, asks the configured LangChain model for a tool choice, executes one registered `@tool`, persists request and result messages, and streams an unbound final model response. Docker Compose supplies MySQL; the existing page consumes one additional SSE status event.

**Tech Stack:** Python 3.14, FastAPI, SQLAlchemy 2.0, MySQL 8 via Docker Compose, PyMySQL, LangChain `@tool`, `langchain-openai`, Pydantic 2, pytest, HTTPX.

**Spec:** `docs/superpowers/specs/2026-10-04-sayhelp-ch02-function-calling-design.md`

## Global Constraints

- Use the user's exact DDL from the spec for all four tables: unsigned BIGINT conversation/message ids, the given enum values, JSON `tool_calls`, FKs, InnoDB, and utf8mb4. Do not create commerce or logistics tables.
- Keep the existing `POST /v1/chat/stream` and `session`, `token`, `error`, `done` events; add `tool_status`. Preserve `POST /v1/aftersales/extract`.
- Execute no more than one actual business tool per customer turn; return a matching ToolMessage for every model-issued call ID. Never enter an Agent Loop.
- Order, product, and logistics values are randomly generated demo results and identified as such. FAQ is literal SQL `LIKE` on `faq.question`; no semantic synonym expansion.
- Read-only tools can retry transient failures within a timeout; `create_ticket` never retries. Keep errors and credentials out of SSE payloads.
- Preserve approximate context-token budgeting. Excluding old turns from the model prompt must not delete their database rows.
- Check current framework/API use with Context7 before implementation. Use only the fixed stack; stop and ask the user if it cannot be made to work.
- After every task, append user wording, output/review, corrections, and failures/rework to `dev-notes/ch02.md`. Never commit `.env`, credentials, or virtual environments. Verify, commit, and push completed work to the configured GitHub remote.
- The chat page is the user's Vibe Coding exception: implement directly and inspect it in the browser; do not apply brainstorm, TDD, or code review to that page.

## File map

- `compose.yaml`: localhost-only MySQL service, healthcheck, and persistent volume.
- `db/schema.sql`: the user's DDL verbatim, in dependency order.
- `app/config.py`, `requirements.txt`: `DATABASE_URL` and SQLAlchemy/PyMySQL dependencies.
- `app/db.py`: SQLAlchemy engine/session factory and four mappings.
- `app/repository.py`: conversation/message persistence, FAQ lookup, and ticket transaction.
- `app/init_db.py`: apply `db/schema.sql` and repeatable FAQ seed.
- `app/tools.py`: the five typed LangChain `@tool` definitions and registry.
- `app/tool_executor.py`: allowlist, schema validation, timeout, safe retries, and ToolMessage conversion.
- `app/chat_service.py`: one-tool turn orchestration, context reconstruction/budgeting, persistence, and chat events.
- `app/model_service.py`, `app/prompts.py`: model tool selection and final streaming, service instructions.
- `app/main.py`, `app/schemas.py`: database-backed SSE route, ID parsing, event framing, and unchanged extraction route.
- `app/web/index.html`: status text and a small tool badge inside the current assistant bubble.
- `tests/test_db.py`, `tests/test_tools.py`, `tests/test_chat_service.py`, `tests/test_chat_api.py`: focused deterministic checks; update obsolete in-memory assumptions in `tests/test_history.py`.
- `tests/fixtures/ch02_cases.json`: labeled FAQ wording and live acceptance prompts.
- `README.md`, `dev-notes/ch02.md`: setup, demonstration, observed results, and phase record.

## Review Focus

1. A model emits two tool calls in one response: Task 3 tests that one executes and both IDs receive ToolMessages, so the final model call stays protocol-valid.
2. A ticket commit completes just before a timeout: Task 2 tests no automatic retry and only one ticket row, preventing duplicate human work.
3. A previous turn ends with an incomplete user or tool-call suffix: Task 3 tests that the next prompt contains only complete call/result pairs.
4. A request resumes a nonexistent, malformed, or concurrently active numeric conversation ID: Task 4 tests 404, 422, or 409 before SSE as appropriate.
5. A model or tool fails after the user message is committed: Task 3 and Task 4 tests safe error framing, retained audit rows, and no fabricated final answer.

---

### Task 1: MySQL persistence and repeatable FAQ setup

**Files:** Create `compose.yaml`, `db/schema.sql`, `app/db.py`, `app/repository.py`, `app/init_db.py`, `tests/test_db.py`; modify `app/config.py`, `requirements.txt`, `dev-notes/ch02.md`.

**Interfaces:**
- `Settings.database_url: str | None`; `Settings.from_env()` requires `DATABASE_URL` for a running Chapter 2 app. Direct test construction may omit it only when a session factory is injected.
- `make_session_factory(database_url: str) -> sessionmaker[Session]`; mappings `Conversation`, `Message`, `Faq`, `Ticket` mirror the DDL.
- `Repository(session_factory)` produces `create_conversation(user_id: str) -> int`, `get_conversation(id: int) -> Conversation | None`, `append_message(id: int, role: str, content: str | None, tool_calls: list[dict] | None = None, tool_call_id: str | None = None) -> None`, `load_messages(id: int) -> list[Message]`, `find_faq(keyword: str, limit: int = 5) -> list[Faq]`, and `create_ticket(conversation_id: int, description: str, ticket_type: str) -> str`.
- `python -m app.init_db` applies the exact SQL schema if absent and idempotently seeds FAQ rows. Partial schema is a clear error; no silent drop/recreate.

- [ ] **Step 1: Write failing `tests/test_db.py` cases** for all DDL columns/enum/checkable FKs in MySQL, conversation and message round trips, bounded literal FAQ lookup, ticket insert plus conversation status update, seed rerun without duplicates, and partial-schema rejection. SQLite may test repository behavior, but MySQL-specific DDL assertions run against Docker MySQL.
- [ ] **Step 2: Run** `py -3.14 -m pytest tests/test_db.py -q`; expect missing modules or failing behavior.
- [ ] **Step 3: Implement** the listed files and signatures. Copy the SQL in the spec exactly to `db/schema.sql`; keep `DATABASE_URL` secrets in ignored `.env`.
- [ ] **Step 4: Run** `py -3.14 -m pytest tests/test_db.py -q` with the test MySQL service available; expect all cases to pass. Run `docker compose config` and `python -m app.init_db` twice; expect valid Compose and no seed duplication.
- [ ] **Step 5: Record Task 1 evidence** in `dev-notes/ch02.md`, commit only Task 1 files, and push the reviewed branch.

### Task 2: Five tools and bounded execution

**Files:** Create `app/tools.py`, `app/tool_executor.py`, `tests/test_tools.py`; modify `app/prompts.py`, `dev-notes/ch02.md`.

**Interfaces:**
- `build_tools(repository: Repository, conversation_id: int) -> dict[str, BaseTool]` returns exactly `query_order(order_id: str)`, `query_product(product_query: str)`, `query_logistics(order_id: str)`, `query_faq(keyword: str)`, and `create_ticket(description: str, ticket_type: Literal["售后", "投诉", "咨询"])`. FAQ keyword is literal customer wording; the ticket uses the captured conversation ID.
- `ToolExecutor(tools: dict[str, BaseTool], timeout_seconds: float = 8, max_read_attempts: int = 2)` has `async execute(call: dict) -> ToolMessage`, preserving `call["id"]` as `tool_call_id`. Invalid names/arguments and exhausted failures return safe error content. Retry only the four read-only tools.

- [ ] **Step 1: Write failing `tests/test_tools.py` cases** for registry names and Pydantic schemas, random-demo result labels, literal FAQ hit/miss, persisted ticket, invalid name and argument handling, read-only timeout retry, and a ticket timeout that inserts once despite the timeout.
- [ ] **Step 2: Run** `py -3.14 -m pytest tests/test_tools.py -q`; expect failures.
- [ ] **Step 3: Implement** the exact tool factory and executor interfaces. Use LangChain `@tool`; validate supplied arguments before invocation and bound each attempt with `asyncio.wait_for`.
- [ ] **Step 4: Run** `py -3.14 -m pytest tests/test_tools.py -q`; expect all Task 2 cases to pass.
- [ ] **Step 5: Record Task 2 evidence**, commit only Task 2 files, and push.

### Task 3: Single-tool chat orchestration

**Files:** Create `app/chat_service.py`, `tests/test_chat_service.py`; modify `app/model_service.py`, `app/history.py` or replace its in-memory behavior behind the new service, `tests/test_history.py`, `dev-notes/ch02.md`.

**Interfaces:**
- `ModelService.choose_tool(messages: list[BaseMessage], tools: list[BaseTool]) -> AIMessage` binds tools and invokes the configured model once. Existing `stream_chat(messages: list[BaseMessage]) -> AsyncIterator[str]` remains the unbound final stream.
- `ChatService(repository: Repository, model_service: ModelService, settings: Settings)` exposes `async stream_turn(conversation_id: int, message: str) -> AsyncIterator[ChatEvent]`. `ChatEvent(kind: Literal["tool_status","token"], data: dict)` is consumed by the HTTP layer. The service loads complete history, prunes old complete turns only in prompt context, saves the user message, tool-call request, every matching tool result, and final assistant message.
- When the model makes no call, the final answer still comes from `stream_chat` so all user-visible answers stream. When it emits several calls, execute only the first; generate skipped ToolMessages for the others. The final model invocation has no tools bound.

- [ ] **Step 1: Write failing `tests/test_chat_service.py` cases** for no-tool streaming, one tool status/result/final stream, multiple-call pairing with one execution, persisted row order and JSON, completed-turn context after restart, budget pruning without DB deletion, incomplete suffix removal, and upstream failure retaining an audit trail without a final assistant row.
- [ ] **Step 2: Run** `py -3.14 -m pytest tests/test_chat_service.py tests/test_history.py -q`; expect failures.
- [ ] **Step 3: Implement** the listed interfaces, including model instructions not to invent tool facts or expand the FAQ keyword. Use labeled FAQ wording cases for prompt behavior rather than tests that merely match prompt text.
- [ ] **Step 4: Run** `py -3.14 -m pytest tests/test_chat_service.py tests/test_history.py -q`; expect all Task 3 cases to pass.
- [ ] **Step 5: Record Task 3 evidence**, commit only Task 3 files, and push.

### Task 4: Persisted SSE chat endpoint

**Files:** Modify `app/main.py`, `app/schemas.py`, `tests/test_chat_api.py`, `dev-notes/ch02.md`.

**Interfaces:**
- `create_app(settings: Settings | None = None, model_service: ModelService | None = None, session_factory: sessionmaker[Session] | None = None) -> FastAPI`.
- `POST /v1/chat/stream` accepts the existing `message` and optional string `conversation_id`; emits `session`, zero or more `tool_status`, final `token` events, and `done`, or a safe `error`. The ID is a decimal string on the wire and unsigned BIGINT in the DB. Pre-stream validation gives 422 for malformed ID, 404 for unknown ID, 409 for active conversation, and 413 for oversized current input.
- `POST /v1/aftersales/extract` remains behavior-compatible.

- [ ] **Step 1: Write failing API tests** for tool status before final tokens, badge-safe public status data, persisted final before `done`, unknown/malformed/active ID behavior, oversized request with no new conversation row, selection/final errors after stream start, and after-sales extraction regression.
- [ ] **Step 2: Run** `py -3.14 -m pytest tests/test_chat_api.py tests/test_extract_api.py -q`; expect failures.
- [ ] **Step 3: Implement** route integration and dependency lifetime, releasing active reservations even if streaming raises or the client disconnects.
- [ ] **Step 4: Run** `py -3.14 -m pytest tests/test_chat_api.py tests/test_extract_api.py -q`; expect all Task 4 cases to pass.
- [ ] **Step 5: Record Task 4 evidence**, commit only Task 4 files, and push.

### Task 5: Vibe Coding page and live acceptance

**Files:** Modify `app/web/index.html`, `README.md`, `dev-notes/ch02.md`; create `tests/fixtures/ch02_cases.json`.

**Interfaces:** The page reads `tool_status` for the current response, shows a transient execution state, and leaves a compact named tool badge in the assistant bubble. It continues to use the existing fetch-based SSE stream, scrolling, and new-conversation action.

- [ ] **Step 1: Edit the page directly** as requested; do not add a page-specific TDD or code-review gate. Do not render raw tool arguments or errors in the badge.
- [ ] **Step 2: Run** `py -3.14 -m pytest -q`, `docker compose config`, and browser inspection of the page at `http://127.0.0.1:8000/`; record actual results.
- [ ] **Step 3: Run the labeled live cases** with the configured DeepSeek model: order 1001 logistics invokes `query_logistics` and answers from its mock result; “退货政策是什么” invokes `query_faq` and matches; “邮费是多少” invokes `query_faq` and misses. Inspect corresponding MySQL rows and the visible badge. If the fixed stack cannot support this, stop and ask the user instead of changing technology.
- [ ] **Step 4: Document exact setup/demo commands**, test outcomes, and the intentional FAQ miss in `README.md` and `dev-notes/ch02.md`.
- [ ] **Step 5: Verify the complete changed tree**, commit only Chapter 2 files, push to GitHub, and perform the Superpowers whole-branch review for backend changes.
