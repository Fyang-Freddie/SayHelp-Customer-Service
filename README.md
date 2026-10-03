# Ecommerce customer service, Chapter 1

This FastAPI service provides a streamed customer service conversation and a separate after-sales information extractor. The conversation history is kept in the API process memory and is lost when the process restarts.

## Set up and run

Use PowerShell from the repository root. Python 3.14 is the version used for this chapter.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a local `.env` file in the repository root. It is ignored by Git. Set the following values to your own OpenAI-compatible chat endpoint; do not commit or print the API key:

```dotenv
CHAT_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-chat
CHAT_API_KEY=your-key-here
```

Start the API in a PowerShell window, from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

`CHAT_BASE_URL`, `CHAT_MODEL`, and `CHAT_API_KEY` are the only settings needed to change chat providers. Chat uses an OpenAI-compatible endpoint. The configured DeepSeek model is the live acceptance target for both routes. Other compatible providers may handle chat, but extraction requires their support for the selected JSON output mode and is not guaranteed for them.

## Acceptance checks

Run these in a second PowerShell window from the repository root. The commands write only demonstration request bodies to temporary files and use `curl.exe` so they work in Windows PowerShell as well as PowerShell 7. `-N` disables curl's output buffering, making each server-sent event visible as it arrives. The examples do not require your API key in any command.

### 1. Stream a new conversation

```powershell
$firstBody = Join-Path $env:TEMP 'customer-service-first.json'
'{"message":"My callback code is maple-731. Please remember it."}' | Set-Content -Path $firstBody -Encoding ascii
curl.exe -sS -N -H 'Content-Type: application/json' --data-binary "@$firstBody" http://127.0.0.1:8000/v1/chat/stream
```

The expected event sequence is `session`, one or more `token` events, then `done`. Copy the `conversation_id` from the `session` event for the next check. Each `token` event carries a text chunk, which may contain more than one model token. An `error` event indicates that streaming failed after it began.

### 2. Continue with the same conversation ID

```powershell
$conversationId = 'paste-conversation-id-from-session-event'
$secondBody = Join-Path $env:TEMP 'customer-service-second.json'
@{ message = 'What is my callback code?'; conversation_id = $conversationId } | ConvertTo-Json -Compress | Set-Content -Path $secondBody -Encoding ascii
curl.exe -sS -N -H 'Content-Type: application/json' --data-binary "@$secondBody" http://127.0.0.1:8000/v1/chat/stream
```

The `session` and `done` events should contain the same conversation ID, and the reply should refer to `maple-731`. An unknown or evicted ID returns HTTP 404. A concurrent request using a conversation ID with an active stream returns HTTP 409; retry after that stream ends. If all conversation slots are active, a new conversation returns HTTP 503; retry after an active stream ends. History is local to one process, so use the same running API for both turns.

### 3. Extract after-sales information

```powershell
$extractBody = Join-Path $env:TEMP 'customer-service-extract.json'
'{"description":"Order A-123: the item arrived damaged. I want a refund to my original payment method."}' | Set-Content -Path $extractBody -Encoding ascii
curl.exe -sS -H 'Content-Type: application/json' --data-binary "@$extractBody" http://127.0.0.1:8000/v1/aftersales/extract
```

The response is a JSON object with exactly `order_id`, `request_type`, and `expected_solution`. `request_type` is one of `退货`, `换货`, `退款`, `维修`, `补发`, or `其他`. Missing order IDs or solutions are `null`; the service must not invent an order ID. An invalid or unsupported upstream JSON response returns HTTP 502 with a safe error message.

## Budget and retention settings

Optional `.env` values are `CONTEXT_TOKEN_BUDGET=4096`, `RESPONSE_TOKEN_RESERVE=512`, `MAX_CONVERSATIONS=100`, and `MAX_TURNS_PER_CONVERSATION=20`. The input limit is the context budget minus the reserved response budget. The service estimates token use across providers, keeps the system prompt and current message, and drops the oldest complete turns as needed. A current message too large for that input limit returns HTTP 413. Conversation and turn caps bound in-memory retention; evicted IDs return HTTP 404.

## Deterministic tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
