# ch04 · 混合检索、证据控制与评估 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在真实知识库上交付四策略可复现评估、带证据的客服回答与拒答记录，以及历史删除/置顶和前端满意度入口。

**Architecture:** MySQL 保存权威 chunk、来源与回答证据快照；独立 `knowledge_ch04` Milvus 集合提供 dense/BM25 原生混合召回，指定 CrossEncoder 重排。单条 query 理解先区分知识、操作、闲聊和澄清；知识路由必须通过检索与生成自评门控，再保存答案与证据并发送已验证文本。评估复用同一检索、上下文组装和生成器，历史管理复用会话持久化与流式请求预留锁。

**Tech Stack:** Milvus 2.6.24 / PyMilvus 2.6.17，MySQL 8.4 / SQLAlchemy 2.0.54，FastAPI 0.142.2 原生 SSE，langchain-openai 1.6.7，sentence-transformers 6.1.0；BAAI/bge-m3、BAAI/bge-reranker-v2-m3。

**Spec:** [已批准设计](../specs/2026-10-05-sayhelp-ch04-hybrid-rag-design.md)，用户于 2026-10-05 回复“设计文档过关”。用户已回复“计划通过，本会话执行”，按 Native 方式实施。

## Global Constraints

- 固定 Milvus 原生 BM25、内置 `analyzer_params={'type': 'chinese'}`、`hybrid_search` + `RRFRanker(k=60)`、BAAI/bge-reranker-v2-m3；dense 为 BAAI/bge-m3 / 1024 维 COSINE。
- dense、BM25 各 Top-50；混合融合 Top-50；重排 Top-10。分数按阶段分别记录，RRF/BM25 不套用 dense 0.55；重排分数不视为概率。
- 只使用 knowledge_db 的四份真实 Markdown 构建正式新库与标注集；不导入演示 FAQ、历史抽取数据或随机生成知识，不修改用户原文。
- 同义词只在检索侧扩展；不入库拆存多份；不做指代消解和多轮改写。型号、数值、否定条件必须保留。
- 用户提供的 low_confidence_questions / faith_cases DDL 原样保存，包含中文注释、ENUM、索引、外键及所有处置/复发字段；本章只在最终 useful=false 入池，不增加 useful 列。
- citations 是实际喂入的完整证据全集；上下文只减少完整 chunk，先按相关性选取，再从首尾向内排列，最后编号。引用不得在生成后改号或伪造。
- 生成无效或证据不足明确拒答；依赖故障提示服务暂不可用，不能登记为知识缺口。未校验答案不得向客户端流出。
- 保留现有数据库记录与旧 `knowledge` 集合；显式增量迁移，不静默删除或重建。历史删除为逻辑删除，保留消息、工单、低置信度关联。
- 后端 RED → GREEN → 有必要的重构 → 验证/评审；Prompt 与人工标注数据用样例验证替代 TDD。前端直接 Vibe Coding + 浏览器验收，不套 brainstorm/TDD/code review。
- 每个阶段/任务完成即时追记 dev-notes/ch04.md 四项信息；阶段验证后仅提交本任务文件并推送指定 GitHub。保留未跟踪 test.py，不提交 .env、凭据、venv、缓存和临时验收文件。
- 每个具体库/API任务先查 Context7 并核对本机签名。遇到固定技术无法实现，停止相关任务向用户说明矛盾；不能换方案或填造评估数字。

## Review Focus

1. 原文表格分块会重复表头、重叠正文也会重复：来源范围必须来自解析位置，不能用全文首次字符串匹配猜行号（Task 2）。
2. MySQL 与 Milvus 双写中途失败、原文更新或重建集合：不能让旧 done 状态跳过新库，不能把陈旧过滤命中当最新证据（Tasks 2–3、6）。
3. 模型丢型号/否定、返回错误 JSON 或把知识问法误选为操作：必须保留限制、重新澄清或安全拒答，不能用商品模拟工具回答规格（Tasks 4、7–8）。
4. 客户断开或重复取消、入池写入失败：事务完成前不能发 done/释放会话预留；答案/证据/入池必须一致（Task 8）。
5. 置顶时间相同、超过一页、生成开始与删除同时发生：排序有唯一游标、操作与预留共锁，删除不影响审计关联或其他会话（Task 10）。

## 文件与接口地图

| 文件 | 职责 |
| --- | --- |
| db/ch04_schema.sql、db/ch04_migration.sql；app/ch04_db.py、app/init_ch04_db.py | 用户两表原始 DDL、显式附加迁移、映射与兼容性校验 |
| app/knowledge_types.py | 查询、过滤、各阶段结果、证据快照的共享类型 |
| app/knowledge_chunking.py、app/ch04_ingest.py、app/ch04_index.py、app/build_ch04_knowledge.py | 精确原文范围、真实文档导入、独立新索引构建与恢复 |
| app/hybrid_store.py | Milvus BM25/dense schema、过滤、原生 RRF；保留旧 vector_store.py |
| app/query_understanding.py、app/query_prompts.py | 单条 query 归一化、检索同义词、意图与限制保留 |
| app/reranking.py、app/rag_retrieval.py、app/evidence.py | 指定重排、四策略入口、置信门控和 prompt 排列 |
| app/rag_generation.py、app/rag_prompts.py | 结构生成、自评、引用校验和明确拒答 |
| app/chat_service.py、app/model_service.py、app/main.py、app/tools.py、app/schemas.py、app/config.py | 聊天/工具接入、SSE 与配置；保持原操作工具行为 |
| app/repository.py、app/chat_views.py、app/db.py、app/knowledge_db.py | 消息证据持久化、公开历史、置顶/删除及新增元数据 |
| app/evaluation/*、app/evaluate_ch04.py、app/faith_cases.py；eval/ch04/* | 标注校验/绑定、校准、指标、独立裁判、报告及台账 CLI |
| app/web/index.html | 现有聊天页引用、反馈、历史菜单；纯前端实现与浏览器验收 |
| tests/test_ch04_*.py、tests/test_rag_*.py、tests/test_conversation_management.py | 后端行为测试、独立真实 MySQL/Milvus 集成验证 |

`RetrievalStrategy = Literal['dense','bm25','hybrid','hybrid_rerank']`；`KnowledgeFilters` 仅接受 category/product_category/content_type 字符串字段，拒绝额外字段与空值。`PreparedQuery` 包含 raw_question、standard_question、search_text、synonyms、intent（knowledge/operation/chitchat/clarify）、clarification、preserved_models/numbers/negations；search_text 保留标准问题并追加受控同义词。`Citation` 与 spec 第 7 节字段同名；内部 id 为整数，所有对外 chunk_id/conversation_id/message_id 为十进制字符串。

`EvidenceChunk` 包含 id、question、answer、category、product_category、content_type、section_path、source_file、source_start_line、source_end_line、source_digest；`StageHit` 包含 chunk_id、rank、score、stage；`RetrievalResult` 包含 strategy、query、candidates（融合/单路 Top-50）、ranked（最终相关性排序最多 Top-10）、stage_hits。`PromptEvidence` 包含 messages、citations、relevance_order、prompt_tokens；生成/评估只消费其中 citations，不能从未喂入 candidates 伪造快照。

## 执行与验证约定

计划分三个里程碑：Tasks 1–6 完成真实检索与证据准备；Tasks 7–9 完成在线知识回答和离线评估；Tasks 10–12 完成历史/前端与交付。推荐在本会话由主代理逐任务实现，最后独立后端评审；接口依赖较强，前端遵循用户例外。若用户选择 subagent-driven，则按该技能逐任务执行评审，仍不对前端套代码评审。

后端测试使用现有 tests/test_db.py 的临时 MySQL 数据库模式；业务 fixture 明确初始化 ch04 schema，旧 DDL 精确验收另保留未经 ch04 迁移的 fixture。新增映射列不能使旧业务测试查询未迁移表。真实 Milvus fixture 创建 `sayhelp_ch04_test_<uuid>` 独立集合，仅清理本测试创建并登记的集合，不要求共享实例没有 knowledge。旧恢复测试在 Task 12 改为同样独立集合，修正环境前置条件，不删除真实集合。

PowerShell 默认 Python：`$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'`。测试进程从 ignored .env 读取 TEST_DATABASE_URL、BGE_CACHE_DIR，只设置进程环境，不打印连接串。下面 pytest 命令均在该环境下运行；纯离线服务边界测试允许 fake，真实集成验收另外列明。每个后端任务提交前对照 spec/本任务进行评审并记录结论；Native 为主代理任务自查及最后独立整体评审，Subagent-driven 为独立逐任务评审及整体评审。任务完成时更新本计划 checkbox、追记四项开发记录、提交并推送验证过的文件；未通过测试/评审的实现不能记为完成。

### Task 1：显式数据库迁移与共享类型

**Files:** Create db/ch04_schema.sql、db/ch04_migration.sql、app/ch04_db.py、app/init_ch04_db.py、app/knowledge_types.py、tests/test_ch04_db.py、tests/ch04_support.py；Modify app/db.py、app/knowledge_db.py、tests/test_db.py、tests/test_knowledge_db.py、tests/test_chat_views.py，以及直接初始化旧 schema 的业务 fixture。

**Interfaces:** `initialize_ch04_database(database_url: str) -> None`；映射 `LowConfidenceQuestion`、`FaithCase`、`KnowledgeIndexState`。附加迁移为 knowledge_chunks 的五个来源/品类字段、conversations.is_pinned/pinned_at/deleted_at、messages.citations（nullable JSON）、独立 knowledge_index_states。后者以 (collection_name, chunk_id) 为唯一键保存 payload_digest、状态、错误及更新时间；不复用旧 vectorize_status。shared types 使用上文定义。

- [x] **Step 1 — RED:** test_ch04_db 验 db/ch04_schema.sql 与 spec 附录 A 的用户 DDL 原文一致；ORM 及真实 information_schema 的类型、默认、ENUM、注释、索引/外键全部一致；重复初始化数据不变；部分/不兼容 schema 报错；旧行来源/citations 为空、未置顶未删除。运行 `& $py -m pytest tests/test_ch04_db.py -q`，确认新接口缺失失败。
- [x] **Step 2 — 实现:** 先 Context7 查 SQLAlchemy/MySQL schema inspection 和 DDL；新两表 SQL 完整保留用户原文，附加项另存 migration。不能按简单 split(';') 切 SQL：用户 COMMENT 字符串内包含分号，脚本执行需识别字符串/注释边界。初始化使用数据库级命名锁和结构预检，每步执行后再核验；MySQL DDL 自动提交的半途失败给出明确修复说明，不假装全部回滚。不改变前三章 DDL/种子或原数据。
- [x] **Step 3 — GREEN:** 同命令通过；既有 test_db/test_knowledge_db 回归，临时库验证证据 JSON 中文与超 JavaScript 安全整数 ID。旧 DDL 精确验证仍对原始 schema 的列/索引/外键逐项全量检查，不能删断言；ORM 原字段对照不变，新增字段由 ch04 复合迁移测试单独精确验证。业务 fixture 升级至 ch04，避免新映射查询不存在列；原 initializer 幂等/部分表拒绝测试继续用旧 fixture。
- [x] **Step 4 — 留痕/提交:** 记迁移边界与验证数字，提交 `feat: add chapter 4 schema and provenance migrations`，推送工作分支。

### Task 2：真实文件来源、行范围与可恢复全量构建

**Files:** Create app/ch04_ingest.py、app/ch04_index.py、app/build_ch04_knowledge.py、eval/ch04/corpus.json、tests/test_ch04_ingest.py、tests/test_ch04_index.py；Modify app/knowledge_chunking.py。

**Interfaces:** 保持 `chunk_markdown(...) -> list[KnowledgeDraft]` 旧参数与 source_line，新增 source_end_line，解析单元同时携带实际起止位置；`ingest_corpus(session_factory, manifest: Path) -> CorpusSnapshot`；`build_index(session_factory, embedder, store, snapshot: CorpusSnapshot, batch_size=32) -> BuildSummary`；CLI `python -m app.build_ch04_knowledge --manifest eval/ch04/corpus.json`，可重跑恢复。CorpusSnapshot 定义在 app/knowledge_types.py，包含 files（source_file→digest）、chunks（id→EvidenceChunk）、payload_digests（id→digest）、corpus_digest；BuildSummary 包含 indexed/reused/removed/verified 整数计数及 corpus_digest。

- [x] **Step 1 — RED:** 验段落重叠、跨行句子、引用、代码块、表格重复表头的实际行范围；两次导入不复制同义词/正文；已有 done 行仍入新构建；新版本不与旧来源混淆。模拟 upsert 后提交失败、行被并发修改及集合被重建；恢复后有效 ID/payload 指纹与清单相等。跑两文件及 test_knowledge_chunking，确认失败。
- [x] **Step 2 — 实现:** corpus.json 显式列出四个文件及 content_type。用标题/型号映射标准商品品类，政策/售后类型明确；不调用 ingest_faq 或 QA staging。同一来源版本精确复用唯一链，无法确认的旧行不伪造来源；新版本单独完整链，旧链保留但从当前构建清单排除。text 统一为标题/问题/正文，payload_digest 包含索引文本、过滤元数据与 source_digest。
- [x] **Step 3 — 实现恢复边界:** 新 collection 独立遍历清单中所有 ID；每次外部 I/O 之前关闭 MySQL 事务。确认 Milvus 主键/元数据后才标记独立状态；存在已完成清单也必须对照服务端实体，发现漏行/旧指纹重新写入。构建完成删除的仅是本新集合中不在当前清单的旧版本实体，保留 MySQL 和旧 knowledge；未全量验证不能切换在线检索。
- [x] **Step 4 — GREEN/提交:** 聚焦测试通过；四份真实文件 dry-run 输出文件/chunk 数量和定位抽检，不能把测试 fixture 当正式知识。记录结果，提交 `feat: build recoverable chapter 4 corpus with exact provenance`。

### Task 3：Milvus 原生 BM25、前置过滤与 RRF

**Files:** Create app/hybrid_store.py、tests/test_ch04_milvus.py；Modify app/config.py、app/ch04_index.py。

**Interfaces:** `HybridMilvusStore(uri, *, collection_name='knowledge_ch04', client=None)`，`ensure_collection()`、`upsert(chunk_id: int, vector: list[float], text: str, metadata: dict[str,str]) -> int`、`retrieve(query: PreparedQuery, vector: list[float] | None, strategy: RetrievalStrategy, filters: KnowledgeFilters) -> StoreResult`、`close()`。StoreResult 定义在 app/knowledge_types.py，含 hits（服务端融合/单路 StageHit）、stage_hits（stage→StageHit列表）、entities（id→text及过滤字段/source_digest），各路名为 dense/bm25/rrf；hybrid_rerank 在此仅作 hybrid，由 Task 6 重排。

- [x] **Step 1 — RED:** 验 schema 的 id/vector/text/sparse/category/product_category/content_type/source_digest，VARCHAR analyzer 和 BM25 Function；不兼容既有 schema 拒绝；写入不含 sparse。spy 验两条 AnnSearchRequest.limit==50、同一 expr/expr_params、RRFRanker(k=60)、hybrid_search limit50。参数注入/未知字段拒绝，纯 BM25 不需要 vector。运行 `tests/test_ch04_milvus.py`，确认失败。
- [x] **Step 2 — 实现:** Context7 再核对 PyMilvus 2.6 与本机签名；text VARCHAR max_length=65535、enable_analyzer=True、chinese；BM25 sparse SPARSE_INVERTED_INDEX，dense COSINE AUTOINDEX，Strong consistency。文本/字段长度按 UTF-8 字节验证，主键为正 INT64。白名单过滤编译为带参数表达式，并传入每条请求；不是 MySQL 事后筛选。
- [x] **Step 3 — 阶段 trace:** 融合以 Milvus hybrid_search 返回为准，不在 Python 重算 RRF。为记录各路分数/名次另运行同参数 dense/BM25 Top-50 查询；报告标明额外诊断查询及耗时，不将其结果冒充 hybrid_search 内部返回。纯路只执行自己的请求。
- [x] **Step 4 — 真实 GREEN:** 独立集合导入真实手册片段，BM25 查询 `MH-LP50` 命中其正确来源，品类/类型前置过滤排除其他实体；检查 schema/函数/索引，再执行两路/RRF。结果必须来自真实 Milvus。旧集合列表/数量不变，lazy client/close 回归通过；记录、提交 `feat: add native Milvus BM25 hybrid retrieval`。

### Task 4：单条 Query 理解与检索侧扩展

**Files:** Create app/query_understanding.py、app/query_prompts.py、eval/ch04/query_examples.json、tests/test_ch04_query.py；Modify app/model_service.py。

**Interfaces:** `QueryUnderstanding.prepare(raw_question: str) -> PreparedQuery`（async）；`ModelService.understand_query(raw_question: str) -> QueryDecision`。模型只收到本轮原话；输出严格结构，保留限制验证由本地函数完成。已保留 filters 不由模型猜出更窄商品范围。

- [x] **Step 1 — RED（代码）:** schema 拒绝额外字段/无效 intent；型号缺失、数字变化、否定消失不能继续回答；坏 JSON/模型失败报安全服务错误；不向模型传历史；纯 BM25 不触发 embedder。运行 `tests/test_ch04_query.py` 确认失败。
- [x] **Step 2 — 实现:** Context7 核对 LangChain structured_output/json_mode 与解析失败；QueryDecision 固定字段 standard_question/synonyms/intent/clarification。规范化副本可改变型号空格/大小写，检索文本必须额外带原型号；受控同义词限 8 个、每个最多 32 字，去重，不删除原句条件。模糊型号/涉及上一轮代词返回 clarify；库存/当前价格不让商品随机工具补事实。
- [x] **Step 3 — Prompt 样例验证:** 人工对照真实文档编写至少 12 个样例，含“邮费谁出”、MH-LP50 无 App、退货质量/非质量、未知型号、明确订单操作、问候及无法单轮判定的问题；用实际聊天模型跑一遍，保存预期/实际标准问法、意图和限制保留结果。未通过例子修 Prompt 重跑，不能以 mock 通过替代。
- [x] **Step 4 — GREEN/提交:** 代码测试通过且样例验证有实测记录，提交 `feat: normalize single queries for retrieval without duplicating knowledge`。

### Task 5：指定重排模型与真实标注集

**Files:** Create app/reranking.py、tests/test_ch04_reranking.py、eval/ch04/calibration.json、eval/ch04/test.json、app/evaluation/dataset.py、tests/test_ch04_dataset.py。

**Interfaces:** `BgeReranker(cache_dir=None, *, model=None).rerank(query: str, chunks: list[EvidenceChunk], limit=10) -> list[RankedChunk]`；RankedChunk 在 knowledge_types.py 定义为 chunk/rank/score。`load_cases(path: Path) -> list[EvalCase]`；EvalCase 在 evaluation/dataset.py 定义并验证 Step 3 全部字段，`bind_gold(cases, corpus: CorpusSnapshot, session_factory) -> list[BoundCase]`；BoundCase 为 case/gold_ids/corpus_digest。

- [x] **Step 1 — RED（代码）:** 输入 50 个候选，model spy 必须收到 50 对 query/text，输出恰最多10个，按重排分数降序、同分按原召回名次；非有限分、输出长度错误、过长 pair 或模型异常明确失败，不降级。dataset validator 拒重复 ID、交叉校准/测试题、不存在或歧义 gold。跑 reranking/dataset 两文件确认失败。
- [x] **Step 2 — 重排实现:** Context7 核对 CrossEncoder/predict；固定 BAAI/bge-reranker-v2-m3、惰性单例、推理锁、batch_size=8、原始 logits（Identity）。读取实际 tokenizer 支持上限，完整 pair 超限先报出 chunk/来源而非静默截断关键限制；不能用别的 reranker。运行 real model smoke 验真实型号近似干扰的 Top-10 和实际模型名。
- [x] **Step 3 — 数据制作与样例验证（代替 TDD）:** 人工编写独立校准12题、冻结测试60题；测试 A_policy/B_model/C_colloquial/D_unknown/E_multi 各12题，每桶 easy/medium/hard 各4。字段 eval_id(<=16)/bucket/difficulty/query/filters/answerable/ground_truth/required_facts/forbidden_claims/evidence，evidence 为 source_file/section_path/evidence_quote，每个 anchor 要唯一绑定当前清单 chunk，多来源明确所有必要条目。
- [x] **Step 4 — 实标复核:** 对照四份原文逐条复查事实/限定条件，自动验精确 quote 与来源指纹，unknown 不标 gold；至少一题涉及 MH-LP50 无 App，一题退款处理≠到账，一题多来源全覆盖。同一近似问法不跨校准/测试泄漏；冻结后不得根据测试成绩改标注/门槛。记录人工对照结果并跑 validator，提交 `feat: add fixed BGE reranking and source-labeled evaluation cases`。

### Task 6：四策略统一入口、置信校准与首尾证据

**Files:** Create app/rag_retrieval.py、app/evidence.py、app/evaluation/calibration.py、app/calibrate_ch04.py、tests/test_rag_retrieval.py、tests/test_rag_evidence.py；Modify app/config.py。

**Interfaces:** `RagRetrieval.retrieve(query: PreparedQuery, strategy: RetrievalStrategy, filters: KnowledgeFilters) -> RetrievalResult`（同步，在线以 to_thread 调用）；`select_prompt_evidence(query, result, settings) -> PromptEvidence`；`ConfidencePolicy.assess(result) -> ConfidenceDecision(sufficient, reason)`；CLI `python -m app.calibrate_ch04 --dataset eval/ch04/calibration.json --output eval/ch04/confidence.json`。

- [x] **Step 1 — RED:** 四策略相同 PreparedQuery/filters；pure BM25 embedder zero calls；hybrid_rerank 必须将融合50交给 Task 5 返回10；MySQL hydration 保持名次且核验 text/metadata/digest，stale 行不作为证据。验证 10 条的 relevance_order 为 `[1,3,5,7,9,10,8,6,4,2]`，最终位置 n=1..10；budget 先剔除低相关完整 chunk，再重新排列编号；无完整条目能放下时证据不足。
- [x] **Step 2 — 实现:** 水合权威 MySQL 行并对比 Milvus text/过滤字段/source_digest 和当前清单；不匹配须报告索引陈旧，不能当正常低分。retrieval 返回阶段 rank/score，最多50候选、所有策略生成阶段最多10条。prompt 预算计入 system、完整问题/证据 JSON、reserve；不夹带历史知识事实。
- [x] **Step 3 — 校准样例验证:** 真 Milvus + 真 embedder/reranker，仅用独立12题为每策略本身分数尺度选择 top-1 门槛，dense取dense、bm25取bm25、hybrid取rrf、hybrid_rerank取reranker。候选阈值为校准观测分数和相邻中点，加全部接受/全部拒绝边界；正例需answerable且预算后证据完整覆盖gold，其他为负例。优先最少负例接受，再最少正例误拒，同分取更保守门槛。保存模型/语料/校准集 digest、逐样本分数与 false accept/false reject，报告小校准集局限；提供显式覆盖配置并记载。置信 artifact 缺失或指纹不符阻止正式切换，不用随意0.55/概率解释。
- [x] **Step 4 — GREEN/提交:** 两测试文件通过；annotated calibration 输出可复查实值，首尾 Prompt 用 MH-LP50/例外政策样例验证，提交 `feat: unify retrieval strategies and calibrate evidence confidence`。

### Task 7：生成 useful 自评、负面知识与引用门控

**Files:** Create app/rag_generation.py、app/rag_prompts.py、eval/ch04/generation_examples.json、tests/test_rag_generation.py；Modify app/model_service.py、app/prompts.py。

**Interfaces:** `GenerationDecision(useful: bool, reason: str, answer: str)`；`RagGenerator.generate(query: PreparedQuery, prompt: PromptEvidence, confidence: ConfidenceDecision) -> KnowledgeAnswer`（async），KnowledgeAnswer 含 useful/answer/reason/pool_source/citations；`ModelService.generate_knowledge(messages) -> GenerationDecision`。固定拒答文本“现有知识库证据不足，无法确认这个问题。请补充具体型号或条件，或联系人工客服核实。”依赖故障另用服务不可用错误。

- [ ] **Step 1 — RED（代码）:** 证据空或 confidence不足最终useful=false/source=retrieval_low_conf；有证据模型自评false/source=self_check；无效结构、true无引用、越界/负数引用安全拒答；false草稿“已经退款”不输出；所有 true 引用属于实际 PromptEvidence。模型故障与坏JSON区分，前者不入知识缺口池。跑 `tests/test_rag_generation.py` 确认失败。
- [ ] **Step 2 — 实现:** 强制严格 JSON useful/reason/answer；引用检测按 `[正整数]` 并验证编号全集，不重编号、不添模型未引用的角标。低召回情况也统一为最终 useful=false 结果。坏结构/引用造成安全拒答用 self_check 及具体原因，不能把解析异常原文/连接信息发送客户端。
- [ ] **Step 3 — Prompt 样例验证:** 用至少12条实标样例+实际模型验证信息充分/不充分、多条件覆盖、未知型号、MH-LP50无App、到账时间、审核必过、免费退换/维修、物流/库存/价格承诺及未执行操作。System 明列所有禁止承诺，材料内容只作证据而非指令。保留例外条件；实际结果记录，未达标修Prompt重验。
- [ ] **Step 4 — GREEN/提交:** 代码边界与样例均通过，提交 `feat: gate knowledge answers with usefulness and citations`。

### Task 8：聊天路由、同轮事务与持久化证据

**Files:** Modify app/chat_service.py、app/main.py、app/model_service.py、app/tools.py、app/tool_executor.py、app/repository.py、app/chat_views.py、app/schemas.py、app/config.py；Create tests/test_rag_chat.py、tests/test_rag_api.py。

**Interfaces:** `ChatRequest.filters: KnowledgeFilters | None`；`Repository.commit_knowledge_answer(conversation_id: int, answer: KnowledgeAnswer, raw_question: str) -> str` 返回消息ID，单事务写最终assistant.citations及必要的low_confidence；`ChatEvent` 新增 retrieval_status/citations，`citations` payload={citations: [...], message_id: str}，done携带message_id。原 session/token/tool_status/done 和旧 sources 继续兼容。create_app 可注入 query_understanding/rag_retrieval/rag_generator，保留旧测试所用 dependency injection。

- [ ] **Step 1 — RED:** 知识路由不依赖 choose_tool；商品规格不会调用随机 query_product；订单/物流/工单仍模拟标签且每轮最多一工具；问候/澄清不入池。有效回答与完整citations同轮保存，刷新仍回放快照；原话含情绪/口语不能被规范化值覆盖。两个拒答入口分别入池一次、时间/会话正确；测试历史内容不能成为本轮知识事实。
- [ ] **Step 2 — RED 事务/取消:** 同事务池insert失败应无最终消息，SSE error且无done；重复取消等写入worker完成后才释放预留；提交前不发送答案token，提交成功后citations在token前；done到客户端时消息/证据/池已落库。断开前尚未完成生成不捏造答案，已开始事务必须等其settle；不把断开当user_feedback。跑两新测试确认失败。
- [ ] **Step 3 — 实现:** 先Context7核对FastAPI SSE/Depends scope=request与LangChain工具定义。query理解→知识检索/门控；操作继续原流程，禁用随机商品规格路径，并将误选query_faq/query_product转同一证据管线。query_faq接受可选filters并复用RagRetrieval，保留keyword/matches基本shape，额外携带来源；pure工具结果不得绕开useful门控。KnowledgeAnswer先事务提交，再通过token分块显示；保留预留直到producer和同步写入结束。
- [ ] **Step 4 — 历史兼容:** public_messages优先使用messages.citations快照，旧记录fallback现有精确sources_from_tool。完整文档白名单保持；新来源字段与URL从服务端生成，版本变化提示而非替换回答当时证据。最终知识/操作assistant都在done返回稳定message_id，前端反馈据此恢复。
- [ ] **Step 5 — GREEN/提交:** 新测试+既有chat/api/tools/extract/history回归；实机问 `MH-LP50能用App吗` 得合法引用，问文档没有的问题明确拒答并SQL只读核对池原话/source；不通过测试脚本发工单或删除用户数据。记录，提交 `feat: integrate persistent evidence-gated knowledge chat`。

### Task 9：评估指标、独立裁判与跨轮编造台账

**Files:** Create app/evaluation/metrics.py、app/evaluation/judge.py、app/evaluation/runner.py、app/evaluation/report.py、app/evaluate_ch04.py、app/faith_cases.py、tests/test_ch04_metrics.py、tests/test_ch04_faith_cases.py、tests/test_ch04_evaluation.py；Modify app/config.py。

**Interfaces:** `retrieval_metrics(ranked_ids: list[int], gold_ids: set[int]) -> dict`；`FaithfulnessJudge.judge(query, answer, citations) -> FaithJudgment`（async，claims: text/supported/reason，拒答无事实N/A）；FaithJudgment 定义在 evaluation/judge.py，含 claims、score（float或None）、reason、failed。`FaithCaseRepository.record(case, strategy, answer, judgment, citations, judge_model)`；`resolve(eval_id, status, resolution)`；`run_evaluation(cases, strategies, services, output_dir) -> RunSummary`，RunSummary 在 evaluation/runner.py 定义，含 run_id/result_path/report_path/completed/failed。CLI `python -m app.evaluate_ch04 --dataset eval/ch04/test.json --strategies dense bm25 hybrid hybrid_rerank --output eval/ch04/reports`；`python -m app.faith_cases list --status 未解决`；`resolve E01 --status 已解决 --resolution "…"`。

- [ ] **Step 1 — RED 指标/报告:** 两个gold在第2/第8，Recall@1/5/10为0/.5/1，MRR=.5；gold空N/A；多证据all-covered须全中。claim两支持一不支持Faithfulness=2/3，纯拒答N/A，judge异常显式failed不记1。聚合分strategy/bucket/difficulty并报告分母、拒答/误拒、unknown拒答、禁止承诺违规、引用合法率、Top50/Top10/prompt实际覆盖；failures不能从分母悄悄消失。
- [ ] **Step 2 — RED 真MySQL台账:** 同eval_id两次编造seen_count=2且一行，快照含未引用证据；标已解决/无需解决须非空说明；复发status未解决/resolution清空、resolved_at和first_seen_at保持、last_seen_at更新。并发upsert不丢计数，一题四策略多次事件均计数；本轮只有忠实成功不自动“已解决”。跑三文件确认失败。
- [ ] **Step 3 — 实现:** Context7查MySQL on_duplicate_key_update；用原子increment和显式时间字段，DDL不新增judge标记列。JUDGE_MODEL可独立配置，必要base_url/key从环境读取不入报告；温度固定0、生成/裁判模型参数记录。与Task7相同generator/evidence/门槛，无运行时调测试集参数。独立裁判逐claim对实际证据判断；少于全supported才记faith_case。
- [ ] **Step 4 — 跑真评估:** 冻结60题×四策略，query归一化一次共享、同过滤/源库/生成参数；按run ID保存逐题JSON、judge理由与完整证据、Markdown汇总，旧run保留不覆盖。包括原文/标注/门槛digest、实际model名、阶段耗时和同模型裁判偏差说明。任何依赖失败如实列failed并使CLI非零；验收前解决后重跑，不能提供假分数。
- [ ] **Step 5 — GREEN/提交:** 三代码测试通过；正式报告四策略均有实值，B_model具体型号BM25命中可定位，D_unknown拒答结果可核查，台账CLI临时库处置/复发验证通过。报告为source-backed实际产物，记录并提交 `feat: evaluate retrieval and faithfulness with durable failure cases`。

### Task 10：历史置顶/逻辑删除、分页与流式冲突

**Files:** Modify app/repository.py、app/main.py、app/schemas.py；Create tests/test_conversation_management.py；数据库字段复用Task1。

**Interfaces:** `Repository.set_pinned(id: int, is_pinned: bool) -> dict`；`soft_delete_conversation(id: int) -> None`；`list_conversations(limit=30, cursor: str | None=None) -> dict`；HTTP PATCH `/v1/conversations/{id}/pin` body `{is_pinned: bool}`，DELETE `/v1/conversations/{id}` 返回204。未知404，非法ID/body/cursor422，生成中409，存储失败503；重复成功删除204；已删除置顶/读取/续聊404。

- [ ] **Step 1 — RED 排序/持久化:** 三条pin同秒和多页mixed排序，普通按max Message.id降序；pin按pinned_at降序、Conversation.id降序打破同分。重复同pin不更新时间；unpin恢复活动位置。带版本的base64url JSON cursor包含pin_group/pinned_at/activity_id/id，与完整lexicographic order一致，拒绝损坏/非法字段；静态遍历无dup/missing。
- [ ] **Step 2 — RED 删除/竞态:** 删除后list/read/continue不可见、重复DELETE幂等，messages/tickets/lowconfidence均保留，别的会话不变；active pin/delete409。模拟聊天预留与删除争锁：先删除则新聊天404，先预留则删除409；HTTP存储异常无成功假响应。运行新测试确认失败。
- [ ] **Step 3 — 实现:** Context7核对SQLAlchemy keyset条件/MySQL行锁与FastAPI请求校验。复用create_app的reservation_lock，聊天预留的存在/未删除检查移到同一锁内；pin/delete检查active并完成DB操作后才释放锁，事务用行锁。单worker运行契约写进README。pagination不再用消息ID作为唯一码；保留before入口名时其值为新opaque cursor，旧数字cursor拒绝并提示刷新。
- [ ] **Step 4 — GREEN/提交:** 真临时MySQL与ASGI覆盖上述场景；旧history/API测试更新为新排序cursor且其他行为回归。仅测试库操作历史，不删除/置顶用户实际会话。记录、提交 `feat: persist chat history pinning and soft deletion`。

### Task 11：前端引用快照、一次性满意度与历史菜单（Vibe Coding）

**Files:** Modify app/web/index.html。现有来源弹窗/移动端侧栏复用，不套前端brainstorm、TDD或code review。

**Interfaces:** 消费Task8 citations/message_id/done与旧sources事件、Task10 list/pin/delete；本地满意度键 `sayhelp.feedback.<message_id>`，value={message_id,conversation_id,choice:'up'|'down',created_at}；流式完成之前不显示可操作反馈，历史完成消息用稳定id恢复。

- [ ] **Step 1 — 直接实现引用:** 新citations字段显示章节、原文、首尾行与source_url；旧字段fallback仍可读。点击角标只展开本轮证据快照，文档跳转显示原文范围；digest不符显示版本变化，不能把新文档当旧证据。全程文本节点渲染，非法编号不生成链接。
- [ ] **Step 2 — 直接实现反馈:** 每段完成回答左下角👍/👎；一次点亮、显示“已反馈”、锁两按钮。写localStorage失败仍内存锁定；刷新按message_id恢复已有选择；不调用后端、不登记低置信度或faith_case。失败/中断回答不出现可点反馈。
- [ ] **Step 3 — 直接实现历史操作:** 每行独立更多按钮，阻止点击传播，文字pin/unpin/delete、图钉及键盘入口。确认框含标题与“从历史列表移除，相关审计记录仍保留”；当前会话删除成功后保存new并切空白，其他会话删除保持current。操作失败不改UI；操作成功重载列表/cursor、保留选中；生成中目标禁用并处理409。删过的savedID404后切新对话，不恢复已删对话。
- [ ] **Step 4 — 浏览器验收/提交:** 实际服务只读确认引用/历史；隔离浏览器用真实文档片段和受控HTTP响应验桌面+390px手机：反馈双击/刷新/存储异常、菜单键盘/误切换、取消确认/删除当前或其他/请求失败、pin/unpin/多页/409、来源原文和版本提示、XSS字面渲染、零pageerror。截图检查布局与关闭按钮；不在用户业务库执行删除。记录结果，提交 `feat: add chat feedback and history actions to the chat page`。

### Task 12：全量验收、后端评审与最终交付

**Files:** Modify README.md、tests/test_knowledge_index.py；必要修复只修改所属任务文件；更新本计划、dev-notes/ch04.md；正式报告提交到eval/ch04/reports/<run_id>/。

- [ ] **Step 1 — 修复旧环境测试:** 将test_real_milvus_mysql_recovery改为独立登记集合的fixture；先RED证明共享实例存在knowledge时应可运行，再最小修改测试隔离配置。清理只针对创建的测试collection，不能删旧库换绿。保留原恢复/主键/幂等断言。
- [ ] **Step 2 — 全量运行:** 在临时MySQL、独立Milvus测试集合、真BGE/真reranker环境跑 `& $py -m pytest tests -q --tb=short`，记录pass/fail/skip并逐项解释；必需ch04集成不得skip。真四策略报告已生成，核对测试60题和校准12题版本、每策略完整结果，具体型号检索、拒答池与台账复发，引用/反馈/历史桌面手机验收证据齐全。
- [ ] **Step 3 — 后端独立 code review:** 依requesting-code-review执行，仅评后端/数据/评估/迁移和测试，不评前端代码；以spec+本plan核验固定技术、指标分母、事务取消和分页/逻辑删除。记录结论，发现问题先复现再返工及对应回归；检查通过后按finishing-a-development-branch完成集成，不覆盖用户test.py。
- [ ] **Step 4 — 交付文档:** README提供可复制PowerShell命令：启动compose → init_ch04_db → build_ch04_knowledge → calibrate_ch04 → evaluate_ch04 → uvicorn单worker → 台账查询/处置。说明8000已占用时复用已有服务或明确停止自身服务再启动；引用点击/反馈/历史交互演示，以及知识缺失拒答和池只读查询命令。提供报告路径、实际测试结果和dev-notes路径。
- [ ] **Step 5 — Finish 留痕/推送:** 及时记用户原话/验收产物/评审/返工；git diff --check、暂存范围与敏感文件核查通过后提交推送，核对远端SHA。最终说明哪些功能完成、真实数字和运行方式；任何剩余阻塞如实报告，不能因已推送就宣称全部验收完成。

## 计划自查与评审状态

2026-10-05 自查：spec每项映射到Tasks1–12；两张用户表原文与附加迁移分开，统一query/filter与四策略公平性、实际证据快照、useful=false两入口及取消边界、台账复发、真实数据和前端流程例外均已覆盖。Review Focus五类风险分别有明确验证步骤；共享类型和邻接任务签名已核对，旧功能不重复建设。校准/Prompt样例替代纯数据TDD，独立真实集成与浏览器验收分别列出。

Context7本轮查证：PyMilvus的AnnSearchRequest expr/expr_params与hybrid_search ranker，SQLAlchemy2.0 MySQL原子upsert/onupdate，CrossEncoder6.1.0 predict/Identity；已与本机inspect.signature对照，不能使用3.0TEXT字段。FastAPI/LC既有官方查询见spec第10节，各实现任务开始前再查本任务所涉API。

**用户已批准计划，本会话执行。** 任务按完成证据勾选，审批不等于实现或验收完成。
