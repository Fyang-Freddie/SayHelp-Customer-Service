# SayHelp 客服系统 · 第 3 章 BGE-M3 dense RAG

FastAPI 流式客服现在会先让模型选择一次业务工具，执行并把结果回灌给模型，然后逐块输出最终回答。会话、消息、工具调用和结果保存在 MySQL；聊天气泡显示本轮调用的工具。

## 启动

在仓库根目录使用 PowerShell。已验证环境为 Python 3.14、Docker Desktop 4.93.0（安装在 `E:\docker`，数据目录 `E:\docker\wsl`）、MySQL 8.4。Docker Desktop 启动后运行：

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

创建不会提交的 `.env`，设置 `CHAT_BASE_URL`、`CHAT_MODEL`、`CHAT_API_KEY`、`MYSQL_DATABASE`、`MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_ROOT_PASSWORD`、`DATABASE_URL`、`MILVUS_MINIO_USER`、`MILVUS_MINIO_PASSWORD`。例如 `DATABASE_URL` 的格式为 `mysql+pymysql://用户:密码@127.0.0.1:3307/数据库名?charset=utf8mb4`。默认宿主机端口是 3307，可用 `MYSQL_PORT` 改动。用于数据库集成测试时，另设 `TEST_DATABASE_URL` 为有权创建和删除**临时测试库**的本机 MySQL 管理连接；测试会自动创建并清理独立库。不要把真实生产库或密钥写进仓库。

启动 API 前，先完成下方 [Chapter 3 建库准备](#chapter-3-offline-dense-knowledge-build) 中的固定 MinIO 官方源码镜像构建与公开 BGE-M3 权重下载。下载完成后保持该 PowerShell 中的 `BGE_CACHE_DIR`、`HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE` 设置；然后按顺序启动 MySQL、初始化种子、启动 Milvus、完成知识建库，最后启动单 worker API：

```powershell
docker compose up -d --wait
python -m app.init_db
docker compose -f compose.milvus.yaml up -d --wait
python -m app.build_knowledge --policy tests/fixtures/ch03_policy.md --manual tests/fixtures/ch03_manual.md
python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

`app.init_db` 按 [建表 DDL](db/schema.sql) 创建 `conversations`、`messages`、`faq`、`tickets` 并灌入示例 FAQ，可重复执行。商品、订单、物流由工具内部随机生成演示数据，不连接公司真实系统，也不建对应表。

打开 [聊天页](http://127.0.0.1:8000/)。例如询问“订单 1001 的物流到哪了”，等待气泡中的 `query_logistics` 徽章和流式回答。左侧“历史对话”显示 MySQL 中保存的会话，可点击恢复并继续聊天；手机上点击顶部“历史”。刷新页面恢复当前选中的会话；“新对话”开启空白会话，不删除旧记录。一次用户请求至多执行一个工具，不运行多轮 Agent Loop。

知识库回答下方提供“来源原文”按钮，回答中的合法 `[n]` 也可点击。弹窗直接显示这次检索保存的知识片段；片段与 `knowledge_db/*.md` 的内容唯一精确对应时，同时显示完整 Markdown 文档、章节路径并高亮所在行。旧记录没有可靠文档对应关系时，只展示当时保存的片段并说明未关联文件，不猜原文路径。

历史与原文接口：`GET /v1/conversations?limit=30&before=不透明游标`、`GET /v1/conversations/{id}/messages`、`GET /v1/knowledge/documents/{filename}`。会话/消息 ID 使用十进制字符串；历史不会显示工具调用参数和模型内部草稿。来源通过新增 `sources` SSE 事件传给页面，原有 FAQ 工具契约不变。文档读取仅允许知识目录内的顶层 Markdown 文件。

验证本次界面修复（需 `.env` 中配置专用 `TEST_DATABASE_URL`）：

```powershell
.\.venv\Scripts\python.exe -c "import os,pytest; from dotenv import dotenv_values; os.environ['TEST_DATABASE_URL']=dotenv_values('.env')['TEST_DATABASE_URL']; raise SystemExit(pytest.main(['tests/test_chat_views.py','tests/test_chat_api.py','tests/test_chat_service.py','tests/test_tools.py','-q']))"
```

启动命令仍使用上方 `uvicorn` 命令；已经运行的旧服务需要重启后刷新页面才会提供新历史接口。2026-10-05 最终完整回归 **257 passed, 1 skipped**（真实 MySQL、缓存 BGE-M3；要求空临时 Milvus 的旧恢复测试未对现有知识库运行）。前端另以隔离浏览器验证历史恢复和桌面/手机来源展示；过程见 `dev-notes/ch04.md`。本次是历史/原文界面修复，第 4 章完整混合检索设计仍见设计草案。

## 三条验收样例

本章 dense 检索标注和“邮费”命中预期保存在 [ch03_cases.json](tests/fixtures/ch03_cases.json)。[ch02_cases.json](tests/fixtures/ch02_cases.json) 是第 2 章历史记录，其中“邮费”无命中属于旧 SQL 行为。在聊天页分别使用“新对话”提交：

| 提问 | 预期工具 | 预期结果 |
| --- | --- | --- |
| 订单 1001 的物流到哪了 | `query_logistics` | 随机模拟物流状态和位置；回答明确说明是演示数据 |
| 退货政策是什么 | `query_faq` | dense 检索命中退货 FAQ，回答包含七天内可申请等内容 |
| 邮费是多少 | `query_faq` | 语义命中运费 FAQ；回答依据命中原文并保留结算页限定 |

`POST /v1/chat/stream` 接收 `{"message":"...","conversation_id":null}`；后续请求可带第一次 `session` 事件给出的 ID。SSE 依次提供 `session`、可选的 `tool_status`、`token` 和 `done`。工具选择、工具结果、最终回答都写入 `messages`。`POST /v1/aftersales/extract` 保留第 1 章售后字段提取接口。

## 测试

```powershell
& 'E:\docker\resources\bin\docker.exe' compose config --quiet
.\.venv\Scripts\python.exe -c "from dotenv import load_dotenv; import pytest; load_dotenv(); raise SystemExit(pytest.main(['-q']))"
```

第二条从忽略的 `.env` 只向测试进程加载变量，使 `TEST_DATABASE_URL` 生效。没有该变量时，数据库集成案例会跳过。2026-10-04 完整 MySQL 测试在第 4 阶段为 137 项通过；第 5 阶段的最终结果见 [开发记录](dev-notes/ch02.md)。

## 范围和限制

五个 LangChain `@tool` 为 `query_order`、`query_product`、`query_logistics`、`query_faq`、`create_ticket`。前三个随机模拟，FAQ 使用本地 BGE-M3 与 Milvus dense 召回、MySQL 原文回填；工单写入 MySQL。读工具有超时和有限重试；创建工单不会自动重试，以免重复创建。聊天页只显示工具名称与状态，不显示参数或错误细节。并发会话预约在单个 API 进程内协调；API 使用单 worker；知识摄入和索引通过 MySQL advisory lock 协调跨进程写入。

可选上下文参数：`CONTEXT_TOKEN_BUDGET=4096`、`RESPONSE_TOKEN_RESERVE=512`、`MAX_CONVERSATIONS=100`、`MAX_TURNS_PER_CONVERSATION=20`。模型服务使用 `CHAT_BASE_URL` 指向的 OpenAI 兼容接口；实际验收使用已配置的 DeepSeek。

## Chapter 3 offline dense knowledge build

Activate the installed environment (`.\.venv\Scripts\Activate.ps1`) before the
`python` commands below. Install `requirements.txt` on Python 3.14. Embedding uses the fixed local
`SentenceTransformer("BAAI/bge-m3").encode()` model (1024 dense dimensions).
The first build downloads public model weights; allow sufficient disk and RAM.
`BGE_CACHE_DIR` optionally selects a local cache outside the repository.

Set `DATABASE_URL`, `MILVUS_MINIO_USER`, and `MILVUS_MINIO_PASSWORD` in your local
`.env` (never commit it). `MILVUS_URI` defaults to `http://127.0.0.1:19530`.
First build the MinIO image and download the model using the commands below.
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

The former official prebuilt MinIO image cannot be pulled here. The 2026-10-05
acceptance compiled the exact official release source and packaged a local image
for the committed Compose. This is a locally built development image; Milvus stays
pinned to 2.6.24. Build it in PowerShell with access to GitHub and Go modules:

```powershell
$taskBuild = Join-Path $env:TEMP 'sayhelp-minio-image'
New-Item -ItemType Directory -Force $taskBuild | Out-Null
docker run --rm --mount "type=bind,source=$taskBuild,target=/out" golang:1.24 sh -c 'git clone --depth 1 --branch RELEASE.2024-05-28T17-19-04Z https://github.com/minio/minio.git /src && cd /src && CGO_ENABLED=0 go build -o /out/minio .'
Set-Content (Join-Path $taskBuild 'Dockerfile') -Value @('FROM golang:1.24', 'COPY minio /usr/bin/minio', 'CMD ["minio"]')
docker build -t minio/minio:RELEASE.2024-05-28T17-19-04Z $taskBuild
```

Verified MinIO source commit: `f79a4ef4d0dc3e6562cad0d1d1db674bc8c75531`.
Direct Go compilation reports `DEVELOPMENT.GOGET`; provenance is the exact source
commit. The build/runtime base is official `golang:1.24`, digest
`sha256:d2d2bc1c84f7e60d7d2438a3836ae7d0c847f4888464e7ec9ba3a1339a1ee804`.
It includes curl for the Compose healthcheck and the build toolchain; this image
is for development. Official guidance: [MinIO README](https://github.com/minio/minio#source-only-distribution),
[Milvus pinned Compose](https://github.com/milvus-io/milvus/blob/v2.6.24/deployments/docker/standalone/docker-compose.yml).

Download the public dense model files without a Hugging Face token. After this
finishes, offline mode avoids extra Hub metadata requests during model loading:

```powershell
$env:BGE_CACHE_DIR = Join-Path $env:USERPROFILE '.cache/huggingface/hub'
python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', cache_dir=os.environ['BGE_CACHE_DIR'], token=False, allow_patterns=['*.json','sentencepiece.bpe.model','pytorch_model.bin','1_Pooling/*'])"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
```

Verified public model revision: `5617a9f61b028005a4858fdac845db406aefb181`;
actual local encode produced 1024-dimensional vectors. Cache files stay outside Git.

Indexing writers share a database-scoped MySQL advisory lock across processes.
Its dedicated connection stays pinned through each batch, with no active SQL
transaction during model or Milvus I/O; competing indexers fail explicitly and
can be retried. Before declaring completion, the command freshly checks pending
rows. A batch that completes zero rows while pending rows remain fails explicitly.

## Chapter 3 conversation QA mining

Run mining once, then resume all pending knowledge indexing through the same path:

```powershell
python -m app.qa_scheduler --once --batch-size 20
```

Run exactly one dedicated scheduler process for daily 02:00 Asia/Shanghai mining:

```powershell
python -m app.qa_scheduler --batch-size 20
```

The scheduler uses APScheduler 3.11 with one job, `max_instances=1` and coalescing.
Use the existing `CHAT_BASE_URL`, `CHAT_MODEL`, `CHAT_API_KEY`, `DATABASE_URL`,
`MILVUS_URI` and optional `BGE_CACHE_DIR` settings. Keep secrets in your ignored
local `.env`. The scheduler runs separately from API workers.

Mining reads complete turns in conversation/message ID order, sends only user and
final assistant evidence, and omits tool payloads, unfinished turns, recognized
credentials and oversized turns. Each request contains at most the configured
number of distinct conversations and 12,000 evidence JSON characters; complete
turns above 4,000 characters or above the serialized JSON budget (including
escaped control characters) are skipped instead of truncated. Model output must
be strict JSON with a valid source reference and verbatim supported excerpts.
The extraction prompt excludes personal/account-specific or uncertain answers.

Each extraction batch commits to `qa_extraction_staging` as `extracted` before
global deduplication. Exact normalized question AND answer matches are discarded
first. Local BGE-M3 question/answer similarities identify semantic candidates,
and the configured model must confirm both meanings and all conditions before
discarding a candidate. Conflicting numeric answers always survive. Kept QA is
inserted as pending FAQ with category `历史客服对话`; knowledge insertion and all
final staging statuses commit together. Both mining and knowledge ingestion use
MySQL advisory ownership; dedup holds ingestion ownership through snapshot,
judgment and commit, with no SQL transaction during external inference.

Rerun `--once` after a failure to resume committed extracted rows and pending
vectors. Staging retains source-turn IDs and audit history. Terminal rows are
skipped, and later complete turns remain eligible. The supplied staging schema
has no empty-result marker, so turns yielding no QA may be evaluated again on a
later run without creating duplicate knowledge. BGE/model failures retain
staging; indexing failures retain pending knowledge.

```powershell
python -m pytest tests/test_qa_mining.py -q
```

Labeled cases are in `tests/fixtures/ch03_qa_cases.json`. Configured model
extraction evaluation: 8/8 correct (4 supported QA, 4 abstentions; precision 4/4).
Dedup judgment: 7/8 labels correct, zero false merges across 6 negative cases;
one expected equivalent involving unspecified support versus human support was
conservatively retained.
Real local BGE-M3 candidate evaluation on the eight dedup pairs admitted both
positive labels (2/2) and four negative pairs (4/6). Keep candidate gates 0.85
question / 0.90 answer; numeric-conflict checks and model equivalence confirmation
remain required. The 7/8 judgment discrepancy above is still conservatively
retained. This small set does not establish general production precision.

## Chapter 3 online demo and observed acceptance

After service startup and one-shot build, run one API worker:

```powershell
python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Open [chat](http://127.0.0.1:8000/) or submit the unchanged SSE endpoint:

```powershell
$body = '{"message":"邮费是多少","conversation_id":null}'
Invoke-WebRequest -Uri http://127.0.0.1:8000/v1/chat/stream -Method Post -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

`query_faq(keyword: str)` retains `keyword`, `matches`, `message`; each match has
`question`, `answer`, `category`. The API emits `session`, `tool_status`
(running/success), streamed `token`, then `done`. The other four tools and
售后 extraction retain their contracts. Backend failure or exhausted FAQ timeout
returns empty matches and a temporary-unavailable message. Models are lazy:
initial loading can exceed the tool deadline; allow it to finish and retry a cold
first request. `KNOWLEDGE_MIN_SCORE` defaults to 0.55; no keyword fallback exists.

Observed 2026-10-05 acceptance used disposable MySQL, real Milvus 2.6.24, real
local BGE-M3, and the configured DeepSeek chat model:

- Two CLI builds indexed 10 then 0 rows: pending=0/done=10, stable MySQL and Milvus
  IDs exactly 1–10.
- Separate runs injected `SystemExit` after pending commit before embedding, and
  immediately after a real upsert. Each had pending=3/done=0 before retry; vector
  IDs were empty and `[1]` respectively. Both recovered pending=0/done=3 and exact
  unique IDs `[1,2,3]`; another retry indexed 0. This is an injected interruption
  check, not an OS process-kill check.
- Real retrieval labels: 6/6 passed. Positive top COSINE scores were 0.649842,
  0.664953, 0.747769; unrelated weather/math/sports scores were 0.468744,
  0.398202, 0.266353. Keep 0.55 for this evaluated set; six cases do not establish
  accuracy on all customer questions.
- Real orphan Milvus IDs and MySQL pending rows were omitted during hydration.
- Real SSE POST via FastAPI TestClient chose `query_faq`, included shipping FAQ
  ID 2 (`订单运费如何计算？`) plus fixture policy ID 7, and grounded the final
  answer in both, including region/order amount/checkout qualification. Fixture
  policy intentionally says 3.14 yuan and full-order free shipping; this is demo
  data. Persisted roles were user/assistant/tool/assistant with matching call ID,
  unchanged JSON keys and SSE sequence.

Enable real-service tests only on a disposable Milvus with no `knowledge`
collection; the test creates and removes it:

```powershell
$env:TEST_MILVUS_URI = 'http://127.0.0.1:19530'
$env:TEST_BGE_M3 = '1'
python -c "import torch; torch.set_num_threads(4); from dotenv import load_dotenv; import pytest; load_dotenv(); raise SystemExit(pytest.main(['-q']))"
docker compose config --quiet
docker compose -f compose.milvus.yaml config --quiet
git diff --check
```

Final branch verification: **245 passed, 0 skipped in 125.60s**. MySQL fixtures create
and clean independent databases. Both Compose checks and diff-check passed.
Private connections were loaded into process variables; `.env`, credentials,
model weights, virtual environments and ignored evidence scripts stay out of
commits. Detailed evidence is in [dev-notes/ch03.md](dev-notes/ch03.md).


### 历史置顶与删除

历史列表支持鼠标滚轮和加载更多。每条记录右侧 `...` 悬停或点击可显示「置顶/取消置顶」「删除」，键盘可聚焦并打开菜单；删除需确认。置顶保存在 MySQL，刷新页面仍保留，置顶项按置顶时间及会话 ID 降序，其他项按最近消息 ID 降序。

- `PATCH /v1/conversations/{id}/pin`，严格 JSON `{"is_pinned": true}`，返回置顶状态；取消置顶传 `false`。
- `DELETE /v1/conversations/{id}`，成功及重复删除均返回 204。逻辑删除使列表、读取及续聊不可见，消息、工单与低置信度记录保留作追溯。
- `GET /v1/conversations?limit=30&before=...` 中 before 取上一页 next_cursor，旧数字游标需刷新后重新加载。
- 生成中的会话置顶/删除返回 409，存储失败返回 503。必须保持 **单进程 `--workers 1`**：聊天预留与历史管理共用进程内锁；多 worker 部署需要先实现跨进程协调。
