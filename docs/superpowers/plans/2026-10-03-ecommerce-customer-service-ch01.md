# Ecommerce Customer Service Chapter 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a DeepSeek verified FastAPI service with SSE customer service chat, bounded multi-turn history, and validated after sales extraction.

**Architecture:** One OpenAI compatible LangChain chat model is configured by three existing `.env` keys. FastAPI routes call a small model service; an in memory conversation store keeps only completed turns and trims oldest turns to an approximate input budget. A separate extraction route uses `with_structured_output(method="json_mode")` and a Pydantic schema.

**Tech Stack:** Python 3.14, FastAPI, LangChain core, `langchain-openai`, Pydantic, pytest, HTTPX, uvicorn.

**Spec:** `docs/superpowers/specs/2026-10-03-ecommerce-customer-service-ch01-design.md`

## Global Constraints

- Check current official API details through Context7 MCP before implementing each FastAPI, LangChain, Pydantic, pytest, or HTTPX call. `resolve-library-id` precedes `query-docs`.
- Keep `CHAT_BASE_URL`, `CHAT_MODEL`, and `CHAT_API_KEY` as the only settings that must change to switch an OpenAI compatible chat endpoint. Never print the key.
- Do not add external tools, an agent loop, native provider SDKs, provider specific configuration requirements, or a chat page.
- Full live acceptance targets the existing DeepSeek `.env` model. Chat is transport compatible with other OpenAI style endpoints; extraction capability is reported clearly when unsupported.
- Deliver `session`, incremental `token`, and `done` SSE events, or an `error` event after stream start. Commit history only for a completed stream.
- Preserve the System Prompt and current user turn while dropping oldest complete turns to meet an approximate budget. Missing or evicted IDs return 404; oversized current input returns 413.
- After **each task**, append a contemporaneous four part entry to `dev-notes/ch01.md`: user key words, output and review, corrections or refusals, failures and rework. Do the same after plan approval, code review, and finish.

## File map

- `requirements.txt`: runtime and test dependencies.
- `app/config.py`: validated environment settings and budget defaults.
- `app/prompts.py`: `PromptTemplate` based customer service and extraction prompts.
- `app/schemas.py`: request and extraction Pydantic models.
- `app/history.py`: conversation IDs, completed turns, caps, and approximate budget trimming.
- `app/model_service.py`: LangChain OpenAI compatible model creation, streaming, and structured extraction.
- `app/main.py`: FastAPI routes and SSE framing.
- `tests/test_config_prompts.py`, `tests/test_history.py`, `tests/test_chat_api.py`, `tests/test_extract_api.py`: focused deterministic tests.
- `tests/fixtures/aftersales_cases.json`: labeled extraction examples used for prompt validation.
- `README.md`: setup, `.env` variables, API payloads, and demonstration commands.

## Review Focus

1. Empty or whitespace only `message` or `description` returns 422; Task 1 and Task 4 tests.
2. Unknown or evicted `conversation_id` returns 404; Task 2 and Task 3 tests.
3. A current turn too large for the input budget returns 413 instead of deleting it; Task 2 and Task 3 tests.
4. A failed or disconnected stream leaves history unchanged; Task 3 tests.
5. Empty or malformed model JSON never becomes a successful extraction; Task 4 tests and labeled examples.

---

### Task 1: Configuration, schemas, and prompts

**Files:** Create `requirements.txt`, `app/__init__.py`, `app/config.py`, `app/prompts.py`, `app/schemas.py`, `tests/test_config_prompts.py`.

**Interfaces:**
- Produces `Settings.from_env() -> Settings` with `chat_base_url`, `chat_model`, `chat_api_key`, `context_token_budget` (default 4096), `response_token_reserve` (default 512), `max_conversations` (default 100), and `max_turns_per_conversation` (default 20). Optional budget keys may be set in `.env`; provider switching still changes only the three `CHAT_*` keys.
- Produces `ChatRequest(message: str, conversation_id: str | None)`, `ExtractRequest(description: str)`, and `AfterSalesExtraction(order_id: str | None, request_type: Literal[...], expected_solution: str | None)`.
- Produces `render_service_system_prompt() -> str` using LangChain `PromptTemplate`; `render_extraction_prompt(description: str) -> str` also uses `PromptTemplate` and states JSON and the exact fields.

- [ ] **Step 1: Write failing tests** for missing `CHAT_*` keys, budget reserve below total, whitespace input rejection, allowed request type values, and rendered prompt constraints (role, no invented order facts, JSON instruction).
- [ ] **Step 2: Run** `python -m pytest tests/test_config_prompts.py -q`; expect failures caused by missing modules or unimplemented behavior.
- [ ] **Step 3: Implement** the listed interfaces. Load `.env` without logging secrets; reject nonpositive budgets and reserve greater than or equal to total. Keep prompt text in this file rather than embedding it in routes.
- [ ] **Step 4: Run** `python -m pytest tests/test_config_prompts.py -q`; expect all Task 1 tests to pass.
- [ ] **Step 5: Append Task 1 entry** to `dev-notes/ch01.md`, then commit only Task 1 files and that note with `feat: add customer service configuration and prompts`.

### Task 2: Bounded conversation history

**Files:** Create `app/history.py`, `tests/test_history.py`.

**Interfaces:**
- Consumes `Settings` from Task 1.
- Produces `ConversationStore(settings: Settings)` with `create() -> str`, `get(conversation_id: str) -> list[BaseMessage]` (raises `UnknownConversation`), `prepare(conversation_id: str, system: SystemMessage, current: HumanMessage) -> list[BaseMessage]` (raises `InputBudgetExceeded`), and `commit(conversation_id: str, user: HumanMessage, assistant: AIMessage) -> None`.
- `prepare` uses `count_tokens_approximately` on candidate messages and removes oldest `HumanMessage`/`AIMessage` pairs. Its limit is `context_token_budget - response_token_reserve`. `commit` caps completed turns; least recently used conversations are evicted above `max_conversations`.

- [ ] **Step 1: Write failing tests** for a new empty conversation, returning completed turns on the second request, dropping oldest full pairs while retaining System and current messages, 413 condition when System plus current exceeds budget, unknown and evicted IDs, and turn/conversation caps.
- [ ] **Step 2: Run** `python -m pytest tests/test_history.py -q`; expect failures.
- [ ] **Step 3: Implement** the store and exception types with the exact interfaces above. Keep the model and HTTP concerns out of this file.
- [ ] **Step 4: Run** `python -m pytest tests/test_history.py -q`; expect all Task 2 tests to pass.
- [ ] **Step 5: Append Task 2 entry** to `dev-notes/ch01.md`, then commit Task 2 files and that note with `feat: add bounded conversation history`.

### Task 3: Streaming conversation API

**Files:** Create `app/model_service.py`, `app/main.py`, `tests/test_chat_api.py`.

**Interfaces:**
- Consumes `Settings`, `render_service_system_prompt`, `ChatRequest`, and `ConversationStore`.
- Produces `ModelService(settings: Settings)` with `stream_chat(messages: list[BaseMessage]) -> AsyncIterator[str]`. Build a `ChatOpenAI` instance with the configured URL, model, and key; forward nonempty text chunks from `astream` without buffering.
- Produces `create_app(settings: Settings | None = None, model_service: ModelService | None = None) -> FastAPI` so tests can inject a fake model. `POST /v1/chat/stream` sends SSE `session` JSON, each `token` JSON, then `done` JSON. Complete history is committed only after the model iterator ends normally.

- [ ] **Step 1: Write failing API tests** using a fake async model for event order, multiple immediate chunks, second turn receiving first turn history, 404 unknown ID, 413 oversized input, upstream failure as safe `error` event, and cancellation or incomplete iterator leaving no committed assistant turn.
- [ ] **Step 2: Run** `python -m pytest tests/test_chat_api.py -q`; expect failures.
- [ ] **Step 3: Implement** the LangChain service and FastAPI route, using the currently documented SSE response interface. Do not buffer the full model response before sending `token` events. Use HTTP errors before stream start and an SSE `error` event after start.
- [ ] **Step 4: Run** `python -m pytest tests/test_chat_api.py -q`; expect all Task 3 tests to pass.
- [ ] **Step 5: Append Task 3 entry** to `dev-notes/ch01.md`, then commit Task 3 files and that note with `feat: stream customer service chat`.

### Task 4: Validated after sales extraction

**Files:** Modify `app/model_service.py`, `app/main.py`; create `tests/test_extract_api.py`, `tests/fixtures/aftersales_cases.json`.

**Interfaces:**
- Consumes `AfterSalesExtraction`, `ExtractRequest`, and `render_extraction_prompt` from Task 1.
- Adds `ModelService.extract_after_sales(description: str) -> AfterSalesExtraction` using `ChatOpenAI.with_structured_output(AfterSalesExtraction, method="json_mode")` and Pydantic validation.
- Adds `POST /v1/aftersales/extract`; returns the three schema fields or a clear nonsecret error when upstream JSON mode is unsupported, empty, or malformed.

- [ ] **Step 1: Create labeled cases** for an explicit order number and refund request, a return request without an order number, a repair request with a requested solution, and an ambiguous request mapped to `其他`; fix expected JSON fields by hand.
- [ ] **Step 2: Write failing API tests** for valid JSON shape, null missing fields, whitespace description 422, and empty, malformed, or rejected upstream output returning a clear server error.
- [ ] **Step 3: Run** `python -m pytest tests/test_extract_api.py -q`; expect failures.
- [ ] **Step 4: Implement** extraction with `with_structured_output(..., method="json_mode")` and the JSON instructing prompt. Do not introduce internal function calling, tools, or provider specific environment knobs.
- [ ] **Step 5: Run** `python -m pytest tests/test_extract_api.py -q`; expect all Task 4 tests to pass. Run the labeled examples against the configured DeepSeek model and record which expected fields matched; for this prompt focused check, the labeled examples serve as the user requested evaluation set.
- [ ] **Step 6: Append Task 4 entry** to `dev-notes/ch01.md`, then commit Task 4 files and that note with `feat: extract after sales requests`.

### Task 5: End to end acceptance and user instructions

**Files:** Create `README.md`; modify only product files required by a failed acceptance check; append `dev-notes/ch01.md`.

**Interfaces:** No new service interface. README documents installation, runtime command, `curl -N` streaming request, two turn reuse of `conversation_id`, extraction request, budget settings, and the DeepSeek versus other provider scope.

- [ ] **Step 1: Write README and exact PowerShell friendly `curl.exe` commands** for each of the three acceptance checks.
- [ ] **Step 2: Run** `python -m pytest -q`; expect all deterministic tests to pass.
- [ ] **Step 3: Start the API** with the existing `.env` and execute the three `curl` acceptance checks against the configured DeepSeek model. Record streamed event order, second turn context result, and validated JSON. Do not expose the key or log full sensitive user input.
- [ ] **Step 4: If acceptance fails, debug from the observed boundary, add a reproducing test, make the minimal fix, rerun focused and full checks, and update the relevant task note with the failure and rework.**
- [ ] **Step 5: Append Task 5 entry** to `dev-notes/ch01.md`, then commit README and accepted fixes with `docs: add chapter 1 run and acceptance guide`.

## Final gates

- Run `superpowers:requesting-code-review` on the complete branch, address actionable findings through `superpowers:receiving-code-review`, then append a code review conclusion to `dev-notes/ch01.md` immediately.
- Run `superpowers:verification-before-completion` with fresh test and live acceptance output. Use `superpowers:finishing-a-development-branch` for the integration decision, then append the finish entry to `dev-notes/ch01.md`.
- Final response lists demonstration commands, actual test results, and the absolute `dev-notes/ch01.md` path. Distinguish tested DeepSeek behavior from untested providers.
