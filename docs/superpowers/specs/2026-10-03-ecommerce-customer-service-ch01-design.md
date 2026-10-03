# Chapter 1: Ecommerce customer service conversation

## Goal and scope

Build a Python FastAPI service for the first chapter of an ecommerce customer service system. The chapter delivers a plain model conversation with streaming, short term context, and a separate after sales extraction endpoint. It does not add external tools, an agent loop, persistence, or a chat page.

The current configured DeepSeek model is the full acceptance target. Conversation requests use one OpenAI compatible client configured by `CHAT_BASE_URL`, `CHAT_MODEL`, and `CHAT_API_KEY` in `.env`. Users can change those three values to try GPT, Claude, DeepSeek, or Ollama compatible endpoints. Structured extraction is guaranteed for the configured DeepSeek endpoint after the live acceptance check. Other endpoints may reject the extraction request if they do not support the selected JSON output mode. No provider specific native API or internal function calling is introduced in this chapter.

## HTTP contract

- `POST /v1/chat/stream` accepts JSON `{"message": "...", "conversation_id": "..."}`; the ID may be omitted to start a new in memory conversation. A supplied ID resumes that conversation. Unknown IDs return HTTP 404 rather than silently starting a different conversation.
- The response is `text/event-stream`. A `session` event contains JSON `{"conversation_id": "..."}`. `token` events contain JSON `{"text": "..."}` for each upstream incremental text chunk as soon as it arrives; a chunk may contain more than one model token. A final `done` event contains the conversation ID and marks successful completion. An `error` event contains a safe error message if failure occurs after streaming has begun.
- `POST /v1/aftersales/extract` accepts JSON `{"description": "..."}` and returns JSON with exactly `order_id`, `request_type`, and `expected_solution`. Missing order IDs or solutions are `null`. `request_type` is one of `退货`, `换货`, `退款`, `维修`, `补发`, `其他`. Extraction must not invent an order number.
- Invalid request bodies return validation errors. Upstream failures before stream start and extraction or parsing failures return a clear server error without exposing secrets.

## Conversation flow

The customer service System Prompt states the agent's role, helpful and concise tone, order related boundaries, and prohibitions on inventing order status, policy, refunds, or completed actions. A LangChain `PromptTemplate` formats the System Prompt. A chat prompt places the formatted System message, trimmed conversation history, and current user message in order. The model uses LangChain's OpenAI compatible chat integration and streams its output through FastAPI SSE.

History is held in process memory by conversation ID. It is lost on restart. The service keeps only completed user and assistant turns. Failed or disconnected streams do not commit partial replies. For a request, it preserves the System Prompt and current user message, then drops oldest complete turns until the approximate input budget fits. The total context budget and reserved response budget are configurable with safe defaults; their difference is the input budget. Counts are approximate because one exact tokenizer cannot serve all configurable model families under the common interface. If the System Prompt and current message alone exceed the input budget, the request returns HTTP 413. In memory conversations and retained turns have finite caps; an evicted ID subsequently returns HTTP 404.

## After sales extraction

The extraction prompt explicitly requests JSON and describes each field and allowed value. The code uses a Pydantic schema with `with_structured_output(..., method="json_mode")` and validates the parsed result. The official DeepSeek JSON mode requires a JSON instruction in the prompt and can occasionally return empty content, so the endpoint handles empty or invalid output as a visible failure. It does not silently convert invalid text into a successful JSON object.

## Modules and boundaries

- Configuration loads and validates environment variables without logging the API key.
- Prompt definitions own the customer service and extraction instructions.
- Conversation history owns IDs, completed turns, and pruning.
- Model service owns the OpenAI compatible LangChain model and its streaming and extraction calls.
- FastAPI routes own request validation, SSE framing, and HTTP errors.

No module should need to read route internals to use another module. The existing `test.py` is a read only connection probe and is not the application entry point.

## Verification

Use TDD for product code. Tests cover SSE event order and incremental chunks, a second turn that receives the first turn's context, oldest turn trimming under a small budget, no partial history after a failed stream, schema validation, and missing fields. Use a model fake for deterministic behavior. After unit checks, start the service with the existing `.env` and run the three user acceptance probes with `curl`: visible streamed output, two turn continuity, and valid after sales JSON. Keep the live result separate from fake model tests. Record each completed task and review result in `dev-notes/ch01.md` as it happens.

## Documentation checked

Context7 MCP was used before API design. Relevant official pages: [FastAPI SSE](https://fastapi.tiangolo.com/tutorial/server-sent-events), [LangChain prompt reference](https://reference.langchain.com/python/langchain-core/prompts/prompt/PromptTemplate), [LangChain structured output reference](https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI/with_structured_output), [LangChain message trimming](https://reference.langchain.com/python/langchain-core/messages/utils/trim_messages), and [DeepSeek JSON output](https://api-docs.deepseek.com/guides/json_mode). [Claude's OpenAI compatibility page](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk) says `response_format` is ignored, which is why cross provider extraction is outside this chapter's guarantee.
