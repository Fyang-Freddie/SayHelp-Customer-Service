# SayHelp 客服系统

## ch05 Workflow + Agent（当前分支）

在线聊天入口现在使用 LangGraph：原话透传 → 七类意图分类 → 固定路由。商品咨询、退款退货先检索并通过置信闸；订单、物流、售后走有预算的 Agent 循环；闲聊和投诉使用固定回复。选择工具时的草稿不发给用户，结束选择后由未绑定工具的模型真实流式生成。每轮最多6次 Agent 模型调用（含最后生成）、6次实际工具执行，默认12000累计token（含分类）；分类请求单独计数。

已有完整 ch04 数据库和知识索引的环境，先备份 MySQL，再执行增量迁移；该命令只增加4张 workflow 元数据表，遇到不兼容/部分迁移会拒绝，不会重建知识库。不要以迁移为由重新摄入或删除已有 Milvus 集合。

```powershell
python -m pip install -r requirements.txt -c requirements-ch05.lock.txt
python -m app.init_ch05_db
python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8001 --workers 1
```

运行配置沿用本地 `.env`，不得提交凭据。启动时验证 ch05 原话检索校准指纹并预热本地检索模型。`eval/ch05/confidence.json` 与当前语料、模型、原话查询模式绑定；ch04 校准文件不能替代它。保留的原始知识验收失败及独立确认集结果见 [验收报告](eval/ch05/report.md)，不要把旧章节状态理解为本章全部完成。

在仓库根目录运行以下命令。两个演示入口始终使用 `.env` 或进程环境配置的真实模型及现有只读知识检索，需已有 MySQL、Milvus、模型缓存和匹配校准。每次运行会向配置模型发送问题、提示、工具结果及检索证据并产生调用费用；没有模拟模式或配置失败后的假模型回退。无需先启动 API 服务，CLI 使用临时 SQLite checkpointer。

```powershell
python scripts/demo_ch05.py --mode bare --case multi-step
python scripts/demo_ch05.py --mode workflow --case logistics
python scripts/demo_ch05.py --mode workflow --case policy
python scripts/demo_ch05.py --mode workflow --case complaint
python scripts/demo_ch05.py --mode workflow --case chitchat
python scripts/demo_ch05.py --mode workflow --case weak-evidence
python scripts/demo_ch05.py --mode workflow --case missing-info
python -m app.bare_agent --message "查询订单1001的状态"
```

`bare` 与 `workflow` 都接受上述7个 case；`--live` 参数已移除。**订单和物流数据源尚未接入**，工具只返回未接入原因，不生成状态、金额或位置。multi-step 先查订单，仅真实结果为已发货/已完成才允许继续查物流；当前应在订单未接入后停止并说明需要核实。商品规格和政策继续通过真实文档检索与置信闸回答，未绑定知识适配器的商品工具也明确返回未接入。后续数据计划存放在 `knowledge_db`，目前没有约定格式或新增连接器。演示不写业务历史或真实 tickets；投诉只打印建议，实际建单需在网页点击「建工单」并确认。末行 JSON 记录调用、工具、token、节点、真实流块数量；`--output 路径.json` 可保存结果。

网页的「转人工」和「建工单」是独立动作。转人工保留前端模拟，只显示“已转接人工客服”，不写工单；建工单打开确认框，可编辑类型/描述或取消。相同动作同载荷重试复用原工单，修改已提交载荷会冲突。忽略建议可以继续聊天，过期/已删除会话动作不能建单；网络结果未知时由用户明确重试，不自动追加请求。

SQLite 默认路径为 `.runtime/ch05/checkpoints.sqlite`（可用 `WORKFLOW_CHECKPOINT_PATH` 覆盖），MySQL 保存业务和完整历史。只能单实例、单进程 `--workers 1`；进程内预约锁和本地 SQLite 不提供跨进程协调。备份前停止这个 API 实例并等待连接关闭，保留 SQLite 及可能存在的关联日志文件，同时独立备份 MySQL。两库没有分布式事务，运行时复制单个 SQLite 文件或分别在不同时间备份两库不保证一致恢复；故障恢复以完整 MySQL 历史重建上下文，不能自动重放写工具。临时 CLI SQLite 在命令退出后删除。

`.runtime/ch05/events.jsonl` 保存受限结构化事件；弱证据原问题单独写入本地 `low-confidence.jsonl`，不进旧低置信度池或飞轮数据库。文件不提交 Git，含原问题的日志需要本地访问控制、备份和清理；当前没有自动轮转/保留策略。日志写失败时不会谎称“已记录”。

移除假数据前的本章验收、真实调用次数与未完成项保留在 [report.md](eval/ch05/report.md)、[acceptance_results.json](eval/ch05/acceptance_results.json)，其中模拟数据结果属于历史基线，不代表当前运行行为；本次结果见 [移除模拟数据补充验收](eval/ch05/no-simulation-report.md)，过程见 [dev-notes/ch05.md](dev-notes/ch05.md)。以下 ch01–ch04 内容保留为历史流程；旧 ChatService 离线评估仍独立保留，旧生产注入参数已不再适用于 create_app。

`main` 已整合 ch01–ch04 各功能分支当前已有的代码。知识问答默认使用 Milvus 原生 BM25 与 dense 混合检索、RRF 融合、bge-reranker-v2-m3 重排和证据门控；业务工具、会话和消息保存在 MySQL。历史列表支持滚动、置顶/取消置顶和逻辑删除。

**ch04 仍在开发中。** 已合并代码不表示整章验收完成：真实模型演示、完整四策略评估、编造个案台账流程，以及满意度前端收尾仍以[实施计划](docs/superpowers/plans/2026-10-05-sayhelp-ch04-hybrid-rag.md)的未完成项为准。开发和验证记录见 [dev-notes/ch04.md](dev-notes/ch04.md)。

## 启动

在仓库根目录使用 PowerShell。已验证环境为 Python 3.14、Docker Desktop 4.93.0（安装在 `E:\docker`，数据目录 `E:\docker\wsl`）、MySQL 8.4。Docker Desktop 启动后运行：

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

创建不会提交的 `.env`，设置 `CHAT_BASE_URL`、`CHAT_MODEL`、`CHAT_API_KEY`、`MYSQL_DATABASE`、`MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_ROOT_PASSWORD`、`DATABASE_URL`、`MILVUS_MINIO_USER`、`MILVUS_MINIO_PASSWORD`。例如 `DATABASE_URL` 的格式为 `mysql+pymysql://用户:密码@127.0.0.1:3307/数据库名?charset=utf8mb4`。默认宿主机端口是 3307，可用 `MYSQL_PORT` 改动。用于数据库集成测试时，另设 `TEST_DATABASE_URL` 为有权创建和删除**临时测试库**的本机 MySQL 管理连接；测试会自动创建并清理独立库。不要把真实生产库或密钥写进仓库。

当前 API 需要 ch04 数据库迁移、真实知识库索引和匹配的校准产物。基础设施构建可参考下方 [Chapter 3 建库准备](#chapter-3-offline-dense-knowledge-build)；模型需准备 BGE-M3 与 bge-reranker-v2-m3，离线模式需要提前缓存两者。新环境按以下顺序准备；本机已完成这些步骤时可直接运行最后的 API 启动命令：

```powershell
docker compose up -d --wait
python -m app.init_ch04_db
docker compose -f compose.milvus.yaml up -d --wait
python -m app.build_ch04_knowledge --dry-run
python -m app.build_ch04_knowledge
python -m app.calibrate_ch04
python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

`app.init_ch04_db` 验证并补齐已有业务表及 [ch04 迁移](db/ch04_migration.sql)，知识索引读取 `knowledge_db` 中的真实文档，默认使用独立 `knowledge_ch04` 集合。`app.calibrate_ch04` 会调用配置的模型归一化标注问题并生成校准产物；换环境重新建库后需重新校准，不能直接把仓库中的既有运行快照视为通用结果。当前商品工具已使用真实知识文档，订单/物流数据源尚未接入并明确返回不可用。

下面保留旧章节的 dense 检索及数据提取说明作历史参考；当前聊天页已兼容 ch04 的 `citations` 和旧 `sources` 事件。

打开 [聊天页](http://127.0.0.1:8000/)。例如询问“订单 1001 的物流到哪了”，等待气泡中的 `query_logistics` 徽章和流式回答。左侧“历史对话”显示 MySQL 中保存的会话，可点击恢复并继续聊天；手机上点击顶部“历史”。刷新页面恢复当前选中的会话；“新对话”开启空白会话，不删除旧记录。一次用户请求至多执行一个工具，不运行多轮 Agent Loop。

知识库回答下方提供“来源原文”按钮，回答中的合法 `[n]` 也可点击。弹窗直接显示这次检索保存的知识片段；ch04 记录通过来源文件、摘要及行范围显示完整 Markdown 文档、章节路径并高亮引用范围；原文更新后仍保留当次证据快照，并提示版本差异。旧记录没有可靠文档对应关系时，只展示当时保存的片段并说明未关联文件，不猜原文路径。

历史与原文接口：`GET /v1/conversations?limit=30&before=不透明游标`、`GET /v1/conversations/{id}/messages`、`GET /v1/knowledge/documents/{filename}`。会话/消息 ID 使用十进制字符串；历史不会显示工具调用参数和模型内部草稿。来源通过 `citations` SSE 事件传给页面，兼容旧 `sources`；`retrieval_status` 展示处理阶段，`timings` 提供不含原文的阶段耗时。文档读取仅允许知识目录内的顶层 Markdown 文件。

验证本次界面修复（需 `.env` 中配置专用 `TEST_DATABASE_URL`）：

```powershell
.\.venv\Scripts\python.exe -c "import os,pytest; from dotenv import dotenv_values; os.environ['TEST_DATABASE_URL']=dotenv_values('.env')['TEST_DATABASE_URL']; raise SystemExit(pytest.main(['tests/test_chat_views.py','tests/test_chat_api.py','tests/test_chat_service.py','tests/test_tools.py','-q']))"
```

启动命令仍使用上方 `uvicorn` 命令；已经运行的旧服务需要重启后刷新页面才会提供新历史接口。2026-10-05 最终完整回归 **257 passed, 1 skipped**（真实 MySQL、缓存 BGE-M3；要求空临时 Milvus 的旧恢复测试未对现有知识库运行）。前端另以隔离浏览器验证历史恢复和桌面/手机来源展示；过程见 `dev-notes/ch04.md`。本次是历史/原文界面修复，第 4 章完整混合检索设计仍见设计草案。

## ch04 本轮修复演示与耗时复现

本机使用的页面为 [8001聊天页](http://127.0.0.1:8001/)。在根目录启动（端口已有服务时先停止旧进程，不要叠加启动）：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8001 --workers 1
```

等待 `Application startup complete`，再刷新页面。启动阶段会预热两个本地模型，第一次启动可能需要读取/下载模型；不调用远程聊天模型。在线检索保留两路 Top-50 和重排 Top-10，减少诊断请求并采用本机实测较快的重排批次。

- 询问“运费怎么算？”或“MH-LP50是否有App远程监控和健康记录？”，点击回答中的 `[n]` 查看原文及章节路径；打开历史消息也可点击。
- 询问“支持火星配送吗？”，应得到明确拒答；真实聊天的 useful=false 会按原有规则写低置信度池。
- [耗时对比报告](eval/ch04/latency_comparison.md)：12道固定输入热检索中位数4.56→3.69秒；该样本不代表所有问题，远程模型与机器负载仍影响实际等待。

```powershell
# 只读本地模型及Milvus，不调用远程模型，不写聊天记录
.\.venv\Scripts\python.exe scripts/compare_ch04_retrieval.py --output eval/ch04/latency_local_rerun.json
.\.venv\Scripts\python.exe scripts/benchmark_ch04_latency.py --output eval/ch04/latency_replay.json
# Edge + Playwright：自编HTTP/SSE夹具，不操作真实会话
node scripts/validate_ch04_citations.cjs
```

只有给 benchmark 脚本显式添加 `--include-generation` 才会把4道标注测试题及对应证据发送给配置的模型服务。`PLAYWRIGHT_MODULE` 可指定已安装的 Playwright 模块路径。完整过程见 [ch04记录](dev-notes/ch04.md)。

## 三条验收样例

本章 dense 检索标注和“邮费”命中预期保存在 [ch03_cases.json](tests/fixtures/ch03_cases.json)。[ch02_cases.json](tests/fixtures/ch02_cases.json) 是第 2 章历史记录，其中“邮费”无命中属于旧 SQL 行为。在聊天页分别使用“新对话”提交：

| 提问 | 预期工具 | 预期结果 |
| --- | --- | --- |
| 订单 1001 的物流到哪了 | `query_logistics` | 明确说明数据源尚未接入，不能提供真实物流状态或位置 |
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

五个 LangChain `@tool` 为 `query_order`、`query_product`、`query_logistics`、`query_faq`、`create_ticket`。订单/物流尚未接入，商品/FAQ 在当前工作流使用真实文档混合检索及置信闸；商品未绑定知识适配器时返回未接入；工单写入 MySQL。读工具有超时和有限重试；创建工单不会自动重试，以免重复创建。聊天页只显示工具名称与状态，不显示参数或错误细节。并发会话预约在单个 API 进程内协调；API 使用单 worker；知识摄入和索引通过 MySQL advisory lock 协调跨进程写入。

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
