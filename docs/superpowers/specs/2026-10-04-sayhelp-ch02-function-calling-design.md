# Chapter 2: Function Calling in the SayHelp chat

## Outcome and scope

A customer asks a question on the existing chat page. The same `POST /v1/chat/stream` request lets the configured model select at most one of five business tools, executes it, returns its result to the model, and streams the final answer. The page shows a compact tool badge on that assistant bubble. Conversations, user messages, tool-call requests, tool results, and final answers persist in MySQL.

This chapter does not add an autonomous tool loop, vector search, RAG, or real commerce and logistics integrations. The separate after-sales extraction route remains available.

## Chosen architecture

Keep the existing FastAPI SSE endpoint and page. Split its backend responsibilities into database models/session setup, repositories for conversations/FAQ/tickets, five LangChain `@tool` definitions, a registry/executor, and a single-turn chat orchestrator. `app.main` remains the HTTP composition root. `app.model_service` binds the registered tools for the selection request, then streams the final response without tools bound. This makes the one-tool boundary explicit.

Alternatives considered: a new Agent endpoint would split the current chat workflow; placing tool execution directly in the route would mix validation, retries, persistence, and streaming. Neither serves this chapter's existing-entry acceptance target as cleanly.

## Database and identity

The user-provided MySQL DDL is authoritative. A committed SQL schema file will preserve the supplied column definitions, enum values, indexes, foreign keys, InnoDB engine, and `utf8mb4` charset exactly. Create in dependency order: `conversations`, `messages`, `faq`, `tickets`. SQLAlchemy 2.0 mappings mirror it, including unsigned BIGINT ids. A repeatable seed command inserts FAQ demonstration rows without duplicates; it never seeds commerce or logistics tables.

| Table | Required shape |
| --- | --- |
| `conversations` | Auto-increment unsigned BIGINT `id`; required `user_id`; status `进行中` / `已转人工` / `已结束`; created and updated times; user index. |
| `messages` | Auto-increment unsigned BIGINT `id`; conversation FK; role `user` / `assistant` / `tool`; nullable content, JSON `tool_calls`, nullable `tool_call_id`; created time. |
| `faq` | Auto-increment unsigned BIGINT `id`; question, answer, category; created and updated times; category index. |
| `tickets` | `ticket_no` string business primary key; conversation FK; description; type `售后` / `投诉` / `咨询`; status `待处理` / `已处理`; created time. |

The current app has no authentication. For this demo, the server creates an opaque guest `user_id` for each new conversation. An existing conversation is resumed by its numeric ID, as the current page resumes by ID today. Authorization of conversation ownership is outside this chapter. The HTTP response may encode the numeric ID as a string to avoid JavaScript integer precision loss; the server validates and converts it for database queries.

`DATABASE_URL` supplies the MySQL connection string. Compose starts MySQL with a persistent named volume and a localhost-only published port; local credentials come from ignored `.env` values. The application connects to MySQL and fails clearly when unavailable. It does not silently fall back to in-memory storage. A setup command applies the committed DDL and seed data.

## Tool contracts

All five tools use LangChain `@tool` with typed arguments and descriptions. The registry exposes exactly these names and validates model-supplied arguments against each tool's schema before execution.

| Tool | Input | Result and side effect |
| --- | --- | --- |
| `query_order` | Customer-supplied order ID | Randomly generated demonstration order details, labeled as mock data. No order table. |
| `query_product` | Product name or ID | Randomly generated demonstration product details, labeled as mock data. No product table. |
| `query_logistics` | Customer-supplied order ID or tracking ID | Randomly generated demonstration shipment details, labeled as mock data. No logistics table. |
| `query_faq` | Literal keyword taken from the customer's wording | Parameterized SQL `LIKE` against `faq.question` (and only that field), returning bounded matches or an explicit no-match result. No synonym expansion. |
| `create_ticket` | Description and one of the three DDL ticket types | Inserts a unique ticket number and ticket row for the active conversation; sets its conversation status to `已转人工`; returns the number. |

The FAQ seed includes a question containing `退货` so “退货政策是什么” can match. It includes an `运费` entry but no `邮费` wording. Therefore the literal `邮费` query is a deliberate miss, recorded as the next chapter's retrieval gap. The tool description and system prompt instruct the model to preserve the customer's keyword rather than substitute a synonym; a labeled evaluation checks the behavior with the configured live model.

Tool execution has a bounded timeout and validates name and arguments before invoking. Safe read-only queries may be retried after transient execution errors or timeouts. `create_ticket` is never automatically retried because its first attempt may have committed before a timeout. Errors become safe, explicit tool results for the model; internal exception details and credentials are never sent to the browser. Every model-issued tool call gets a matching `tool` result message, including skipped calls if the provider emits more than one, but only the first permitted call executes. This keeps the conversation protocol valid while enforcing one actual tool execution per user turn.

## Chat and persistence flow

1. Validate the request and existing conversation before beginning SSE. Reserve a conversation against concurrent streams. For a new chat, create its database conversation and guest user ID.
2. Load recent completed turns from `messages`, preserving the existing input token budget and oldest-turn pruning for model context. Database rows are retained even when excluded from the prompt.
3. Save the user message. Ask the tool-bound model once for a selection response. If it requests one or more calls, persist the assistant request with its full `tool_calls` JSON and emit a status event with the selected tool name. Execute only the first permitted call and persist its `tool` response with the matching `tool_call_id`; provide matching skipped/error results for any other requested IDs.
4. Send the selection message and tool result(s) back to the model without tools bound. Stream its final answer as existing `token` SSE events. Persist the final assistant message before `done`. If no tool is selected, stream a direct answer and persist it.
5. Keep the current `session`, `token`, `error`, and `done` event contract. Add a `tool_status` event containing a public tool name and execution state. The page uses it to show a temporary status and a tool badge in the current assistant bubble. Tool output stays in the database and model context; the badge does not expose raw arguments or internal errors.

If selection or final generation fails, send a safe `error` event and do not write an invented assistant answer. Already committed user, request, and tool rows remain as an audit trail. Context reconstruction ignores an incomplete suffix and preserves valid tool-call/result pairing. The existing after-sales extraction route keeps its behavior.

## Verification and delivery

Backend tasks follow TDD for mapping and seed behavior, tool validation/execution, single-tool orchestration, persistence, and SSE event ordering. Pure FAQ seed and prompt behavior use labeled examples rather than unit tests that merely mirror text. The chat page is changed directly using the requested Vibe Coding exception, then inspected in a browser.

Acceptance uses the configured DeepSeek model and local MySQL: ask “订单 1001 的物流到哪了” and inspect tool choice, badge, and answer; ask “退货政策是什么” and inspect FAQ match and answer; ask “邮费是多少” and confirm the SQL keyword miss. Also inspect persisted assistant tool-call JSON, corresponding tool row, final answer, and any ticket row. Run the full deterministic test suite, document runnable demonstration commands and results, update `dev-notes/ch02.md` after every phase, then commit and push verified changes to the configured GitHub remote. Never commit `.env` or secrets.

## Documentation checked before implementation

- FastAPI SSE generator and `ServerSentEvent`: https://fastapi.tiangolo.com/tutorial/server-sent-events/
- SQLAlchemy 2.0 session/select pattern: https://docs.sqlalchemy.org/en/20/orm/session_basics.html
- LangChain `@tool`, `bind_tools`, and tool result handoff: https://docs.langchain.com/oss/python/langchain/models and https://docs.langchain.com/oss/python/langchain/tools
- DeepSeek Function Calling response and tool-message pairing: https://api-docs.deepseek.com/guides/tool_calls
- Docker Compose service and local port behavior: https://github.com/docker/compose

### Authoritative DDL supplied by the user

The implementation copies this DDL into its executable schema file without changing its columns, enum values, indexes, or foreign keys.

```sql
CREATE TABLE conversations (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '会话主键',
  user_id     VARCHAR(64)     NOT NULL                COMMENT '用户标识',
  status      ENUM('进行中','已转人工','已结束') NOT NULL DEFAULT '进行中' COMMENT '处理状态',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '开启时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客服会话';

CREATE TABLE messages (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '消息主键',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '所属会话',
  role            ENUM('user','assistant','tool') NOT NULL COMMENT '角色:用户/助手/工具结果',
  content         TEXT            NULL                     COMMENT '消息正文,assistant 纯工具调用时可为空',
  tool_calls      JSON            NULL                     COMMENT 'assistant 消息带的工具调用申请单',
  tool_call_id    VARCHAR(64)     NULL                     COMMENT 'tool 消息对应的申请单 id,回灌时对号入座',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '产生时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_messages_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话消息流水';

CREATE TABLE faq (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'FAQ 主键',
  question    VARCHAR(512)    NOT NULL                COMMENT '问题',
  answer      TEXT            NOT NULL                COMMENT '答案',
  category    VARCHAR(64)     NOT NULL                COMMENT '分类',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='常见问答';

CREATE TABLE tickets (
  ticket_no       VARCHAR(32)     NOT NULL                COMMENT '工单号,如 T20260701008',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '关联会话,可倒查当时聊了什么',
  description     TEXT            NOT NULL                COMMENT '问题描述',
  ticket_type     ENUM('售后','投诉','咨询') NOT NULL     COMMENT '工单类型',
  status          ENUM('待处理','已处理') NOT NULL DEFAULT '待处理' COMMENT '处理状态',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (ticket_no),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_tickets_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='人工工单';
```
