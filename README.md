# SayHelp 客服系统 · 第 2 章 Function Calling

FastAPI 流式客服现在会先让模型选择一次业务工具，执行并把结果回灌给模型，然后逐块输出最终回答。会话、消息、工具调用和结果保存在 MySQL；聊天气泡显示本轮调用的工具。

## 启动

在仓库根目录使用 PowerShell。已验证环境为 Python 3.14、Docker Desktop 4.93.0（安装在 `E:\docker`，数据目录 `E:\docker\wsl`）、MySQL 8.4。Docker Desktop 启动后运行：

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

创建不会提交的 `.env`，设置 `CHAT_BASE_URL`、`CHAT_MODEL`、`CHAT_API_KEY`、`MYSQL_DATABASE`、`MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_ROOT_PASSWORD`、`DATABASE_URL`。例如 `DATABASE_URL` 的格式为 `mysql+pymysql://用户:密码@127.0.0.1:3307/数据库名?charset=utf8mb4`。默认宿主机端口是 3307，可用 `MYSQL_PORT` 改动。用于数据库集成测试时，另设 `TEST_DATABASE_URL` 为有权创建和删除**临时测试库**的本机 MySQL 管理连接；测试会自动创建并清理独立库。不要把真实生产库或密钥写进仓库。

```powershell
& 'E:\docker\resources\bin\docker.exe' compose up -d --wait
.\.venv\Scripts\python.exe -m app.init_db
.\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

`app.init_db` 按 [建表 DDL](db/schema.sql) 创建 `conversations`、`messages`、`faq`、`tickets` 并灌入示例 FAQ，可重复执行。商品、订单、物流由工具内部随机生成演示数据，不连接公司真实系统，也不建对应表。

打开 [聊天页](http://127.0.0.1:8000/)。例如询问“订单 1001 的物流到哪了”，等待气泡中的 `query_logistics` 徽章和流式回答。“新对话”会开始新的会话；刷新页面后当前页面不会恢复旧会话 ID，但已保存的数据库记录仍在。一次用户请求至多执行一个工具，不运行多轮 Agent Loop。

## 三条验收样例

样例和预期保存在 [ch02_cases.json](tests/fixtures/ch02_cases.json)。在聊天页分别使用“新对话”提交：

| 提问 | 预期工具 | 预期结果 |
| --- | --- | --- |
| 订单 1001 的物流到哪了 | `query_logistics` | 随机模拟物流状态和位置；回答明确说明是演示数据 |
| 退货政策是什么 | `query_faq` | SQL LIKE 命中退货 FAQ，回答包含七天内可申请等内容 |
| 邮费是多少 | `query_faq` | 按原词“邮费”查询无命中，回答承认未查到；这是第 3 章检索升级要解决的漏召回 |

`POST /v1/chat/stream` 接收 `{"message":"...","conversation_id":null}`；后续请求可带第一次 `session` 事件给出的 ID。SSE 依次提供 `session`、可选的 `tool_status`、`token` 和 `done`。工具选择、工具结果、最终回答都写入 `messages`。`POST /v1/aftersales/extract` 保留第 1 章售后字段提取接口。

## 测试

```powershell
& 'E:\docker\resources\bin\docker.exe' compose config --quiet
.\.venv\Scripts\python.exe -c "from dotenv import load_dotenv; import pytest; load_dotenv(); raise SystemExit(pytest.main(['-q']))"
```

第二条从忽略的 `.env` 只向测试进程加载变量，使 `TEST_DATABASE_URL` 生效。没有该变量时，数据库集成案例会跳过。2026-10-04 完整 MySQL 测试在第 4 阶段为 137 项通过；第 5 阶段的最终结果见 [开发记录](dev-notes/ch02.md)。

## 范围和限制

五个 LangChain `@tool` 为 `query_order`、`query_product`、`query_logistics`、`query_faq`、`create_ticket`。前三个随机模拟，FAQ 使用字面 SQL LIKE，工单写入 MySQL。读工具有超时和有限重试；创建工单不会自动重试，以免重复创建。聊天页只显示工具名称与状态，不显示参数或错误细节。并发会话预约在单个 API 进程内协调；本章不含跨进程协调、向量检索或 RAG。

可选上下文参数：`CONTEXT_TOKEN_BUDGET=4096`、`RESPONSE_TOKEN_RESERVE=512`、`MAX_CONVERSATIONS=100`、`MAX_TURNS_PER_CONVERSATION=20`。模型服务使用 `CHAT_BASE_URL` 指向的 OpenAI 兼容接口；实际验收使用已配置的 DeepSeek。

## Chapter 3 offline dense knowledge build

Install `requirements.txt` on Python 3.14. Embedding uses the fixed local
`SentenceTransformer("BAAI/bge-m3").encode()` model (1024 dense dimensions).
The first build downloads public model weights; allow sufficient disk and RAM.
`BGE_CACHE_DIR` optionally selects a local cache outside the repository.

Set `DATABASE_URL`, `MILVUS_MINIO_USER`, and `MILVUS_MINIO_PASSWORD` in your local
`.env` (never commit it). `MILVUS_URI` defaults to `http://127.0.0.1:19530`.
Start MySQL using the existing `compose.yaml`, then start Milvus:

```powershell
docker compose -f compose.milvus.yaml config --quiet
docker compose -f compose.milvus.yaml up -d --wait
python -m app.build_knowledge --policy tests/fixtures/ch03_policy.md --manual tests/fixtures/ch03_manual.md
```

Repeat that build command to resume interrupted writes. FAQ and unchanged Markdown
keep their MySQL IDs. The indexer closes the MySQL read transaction before inference
or network I/O, upserts the same INT64 ID in Milvus, and only then marks MySQL done.
If Milvus succeeds before a crash, retry overwrites that same ID and completes the
MySQL status. `--batch-size 100` bounds each pending snapshot; the command drains
pending rows. Changed documents append new chains; replacing existing sources is
outside this command. Milvus stores only IDs and vectors with COSINE indexing.

`compose.milvus.yaml` follows the official pinned Milvus standalone layout with
persistent named volumes and loopback endpoints. It requires private MinIO values
instead of committing defaults. To run deterministic indexing tests with disposable
MySQL, set `TEST_DATABASE_URL` in the process and run:

```powershell
python -m pytest tests/test_knowledge_index.py -q
```

Set `TEST_MILVUS_URI` only for a disposable Milvus instance with no `knowledge`
collection to enable the real Milvus integration test; it creates and removes that
collection. Real BGE model verification uses `TEST_BGE_M3=1` and may download weights.

Current environment limitation (2026-10-04): the exact official MinIO community
image above is no longer downloadable here (Docker Hub tag API 404; pulls denied,
Quay source 401). MinIO's current official repository documents source-only
community distribution. A compatible image built from official MinIO source or an
existing trusted local copy is required before the startup command can succeed.
No real Milvus acceptance result is claimed until that image is available.
Sources: https://github.com/minio/minio#source-only-distribution and
https://github.com/milvus-io/milvus/blob/v2.6.24/deployments/docker/standalone/docker-compose.yml.

The current deterministic suite verifies MySQL with fake external adapters.
Real Milvus and real BGE tests are opt-in and were not passed in this environment:
Milvus startup is blocked by the MinIO image supply issue, and BGE weights are not
cached (public model configuration is reachable, but real encode has not run).
Skipped opt-in tests do not count as real retrieval/indexing acceptance.

Indexing writers share a database-scoped MySQL advisory lock across processes.
Its dedicated connection stays pinned through each batch, with no active SQL
transaction during model or Milvus I/O; competing indexers fail explicitly and
can be retried. Before declaring completion, the command freshly checks pending
rows. A batch that completes zero rows while pending rows remain fails explicitly.
