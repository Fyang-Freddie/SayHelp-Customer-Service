# SayHelp 第 5 章 Workflow + Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先演示手写 Agent 循环，再以 LangGraph 固定编排实现七类分流、前置检索闸、可控 ReAct、SQLite 状态持久化和独立人工/工单按钮。

**Architecture:** 外层 StateGraph 编排确定性路径，Agent 子图负责模型—工具循环，继承父图 checkpointer。模型决策不向用户输出，终止工具循环后由无工具绑定的模型真实流式生成回复；MySQL 保存业务与历史，官方 AsyncSqliteSaver 保存运行状态。

**Tech Stack:** Python 3.14、FastAPI 原生 SSE、SQLAlchemy 2.0/MySQL、LangChain 现有模型和工具、LangGraph/SQLite checkpointer、现有 Milvus RAG、原生 HTML/CSS/JavaScript。

**Spec:** `docs/superpowers/specs/2026-10-07-sayhelp-ch05-workflow-agent-design.md`（2026-10-07 用户回复“同意”批准）。

**Status:** 用户已选择 Subagent-driven；任务 1–8 已完成，任务 9 的实现、独立评审和离线验证已通过（688 tests，38 项动作交互，25 项引用展示）。真实模型整体验收为 0/10，等待明确的数据发送授权；本章暂不标记完结，保留工作树和 SDD 记录。

## Global Constraints

- SQLite checkpointer，单实例 `--workers 1`；不引入 PostgreSQL、不实现跨进程协调。
- 七类意图一次 JSON 分类；闲聊分类后固定回复。指代消解原样透传。
- 商品咨询、退款退货强制检索并在 Agent 首个 token 前过闸；业务数据类不预检索、不走闸。
- 复用第 2 章工具和第 4 章检索器；不新写业务工具，不升级上下文策略、MCP 业务接入或飞轮入库。
- 每轮最多 6 次 Agent 模型调用（包括最终流式生成）、6 次实际工具执行；累计 token 默认 12000，包含分类；相同工具/参数连续失败两次停止。最终生成预留一次模型调用和输出预算。
- 精确问候：`您好,我是SayHelp,请问有什么可以帮您的`；精确转接提示：`已转接人工客服`。
- create_ticket 不能进入 Agent 可执行列表；建议动作不造成后端副作用，用户忽略建议可正常继续聊天。
- 每任务先失败测试再实现；纯 prompt/数据任务使用标注评估替代 TDD，不以字符串快照冒充效果验证。
- 每阶段即时追加 `dev-notes/ch05.md` 四项记录（关键原话、产出、拒绝/纠偏、翻车/返工），不收尾补记。
- 具体库接口先用 Context7 MCP 核对，工具不可用或指定技术走不通时停下说明，不擅自换技术。
- 保护原有未提交文件：ch04 计划、confidence.json、confidence.md、test.py；隔离工作树不能自动包含这些修改，也不能将其提交到本章。
- 完成并验证的工作提交且推送指定 GitHub；仅明确列出的文件进入提交，不提交 .env、密钥、运行数据库、客户数据、虚拟环境。

## Review Focus

1. 工具流先出现正文、随后出现工具调用：不得把暂定内容、参数或隐藏推理泄露成答案（任务 1、6）。
2. MySQL 成功而 SQLite 最后检查点失败，以及相反顺序：不重复历史、不发完成事件、不自动重放写工具（任务 4、6）。
3. 同一工单动作被双击、超时后重试或换描述重试：最多一张工单，相同请求返回原结果，修改载荷返回冲突（任务 4、7）。
4. 切换会话/删除会话后旧 SSE 或旧按钮仍到达：不得污染新会话或对已删除会话建单（任务 6、7、8）。
5. 多步历史中途失败、重复工具 ID、下一轮继承旧证据/建议：只恢复完整上下文，轮次状态清零（任务 1、5、6）。

## 文件与接口边界

- `app/agent_runtime.py`：框架无关循环、预算、工具允许列表与控制建议；`app/bare_agent.py`：裸循环 CLI 演示。
- `app/workflow_types.py`：共享 State、动作与用量数据；`app/intent.py`、`app/intent_prompts.py`：七类分类及固定路由。
- `app/workflow_knowledge.py`：复用检索与闸；`app/workflow_log.py`：结构化事件/弱证据本地记录。
- `app/ch05_db.py`、`app/init_ch05_db.py`、`db/ch05_schema.sql`：新增本章元数据表；`app/workflow_repository.py`：稳定消息身份与动作存储。
- `app/workflow_graph.py`：外层图与 Agent 子图；`app/workflow_service.py`：SQLite 生命周期、会话恢复和 SSE 事件适配。
- `app/ticket_actions.py`：确认建单服务与 API 接口；现有 `tools.py`、`repository.py` 保留业务实现，解除人工绑定。
- 现有 `model_service.py`、`history.py`、`main.py`、`chat_views.py`、`config.py`、`web/index.html` 仅做本章所需扩展；保留原章节离线评估代码。

共享类型在任务 1 定义：`ActionKind = Literal['handoff','create_ticket']`；`ActionSuggestion(kind, description, ticket_type)`（handoff 不带工单内容）；`Usage(input_tokens, output_tokens, estimated)`；`AgentResult(messages, answer, suggestions, usage, model_calls, tool_calls, stop_reason)`；`Emit = Callable[[str, dict], Awaitable[None]]`。

`WorkflowState` 使用 TypedDict，字段为 conversation_id、turn_id、messages、raw_question、resolved_question、intent、route、filters、evidence、citations、confidence、model_calls、tool_calls、usage、suggestions、answer、status、stop_reason、message_id；LangGraph 消息 reducer 只管理 messages，轮次字段显式覆盖。messages 为 LangChain 消息，其他字段为可序列化标量/list/dict。对象资源放在依赖注入闭包，不进 State。

## 已完成的文档与环境核查

本地版本：Python 3.14.5、FastAPI 0.142.2、SQLAlchemy 2.0.54、langchain-core 1.6.6、langchain-openai 1.6.7、pytest 9.1.1；LangGraph/SQLite checkpointer 尚未安装。保持已有兼容库版本，不顺带升级全栈。

已通过认证 Context7 MCP 定位并查询：

- `/websites/langchain_oss_python_langgraph`：子图默认继承父 checkpointer；官方 SQLite 定位与流式事件方式。
- `/websites/reference_langchain_python_langgraph`：`langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`；返回摘要未完整提供生命周期签名，任务 5 必须再次定向核查并检查安装包签名，不能照抄检索结果里无关的 Postgres 示例。
- `/websites/langchain_oss_python_langchain`：模型工具调用、AIMessageChunk 聚合与流式用量。具体 ChatOpenAI usage 参数仍需在任务 1 针对集成文档核查。
- `/websites/fastapi_tiangolo`：原生 `fastapi.sse.EventSourceResponse`、`ServerSentEvent` 和流结束后的 yield 依赖清理。
- `/websites/sqlalchemy_en_20`：事务、唯一冲突、`Session.begin_nested()`，注意其先 flush 语义。

官方来源：`https://docs.langchain.com/oss/python/langgraph/use-subgraphs`、`https://reference.langchain.com/python/langgraph.checkpoint.sqlite`、`https://docs.langchain.com/oss/python/langchain/models`、`https://fastapi.tiangolo.com/tutorial/server-sent-events/`、`https://docs.sqlalchemy.org/en/20/orm/session_transaction.html`。

以下命令中 `python` 指隔离工作树的 Python 3.14 虚拟环境；执行时先按 using-git-worktrees 建立隔离分支，读取 AGENTS、核对基线并运行相关基线测试。新依赖安装先 dry-run，锁定经过 API 探针和 pip check 验证的实际版本到 `requirements-ch05.lock.txt`，不猜版本号。所有联网探针仅查询公开资料，不发送项目代码或凭据。

## Task 1: 裸 Agent 循环、模型适配与预算

**Files:** Create `app/agent_runtime.py`, `app/bare_agent.py`, `app/workflow_types.py`, `tests/test_agent_runtime.py`; Modify `app/model_service.py`, `app/tool_executor.py`, `app/config.py`, `app/prompts.py`。

**Interfaces:**
- `AgentModel.select(messages: list, tools: list, *, max_tokens: int) -> AIMessage`（async）；`AgentModel.stream_reply(messages: list, *, max_tokens: int) -> AsyncIterator[AIMessageChunk]`；由 ModelService 实现，不删除原方法。
- `run_bare_agent(messages: list, *, model: AgentModel, tools: dict, limits: AgentLimits, usage: Usage, emit: Emit) -> AgentResult`（async）。`AgentLimits` 默认 model_calls=6、tool_calls=6、turn_tokens=12000，工具超时继续复用现有 ToolExecutor 默认 8 秒/只读最多两次尝试。
- `agent_step(state: WorkflowState, *, model, tools, limits, emit) -> dict`、`execute_calls(state: WorkflowState, *, tools, limits, emit) -> dict`、`stream_answer(state: WorkflowState, *, model, limits, emit) -> dict`（async）：裸循环和后续子图复用这些步骤。

- [x] 使用 Context7 核对 ChatOpenAI 的 bind_tools/usage 参数；新增离线适配测试模拟分片和使用量，真实模型能力探针留到任务 9 受控演示。
- [x] 写 `test_order_then_logistics_feeds_each_result`：脚本模型依次查订单、查物流、无工具终止；断言业务工具次数 2、第二次选择能看到订单结果、第三次能看到物流结果、最后调用 stream_reply 并逐片 emit token。最终流不绑定工具，选择阶段的 draft 不进入正文；另测先正文后工具调用只作为内部选择结果。
- [x] 写 `test_budget_reserves_final_generation`、`test_repeated_failures_stop`、`test_invalid_or_duplicate_ids_stop_before_execution`、`test_denies_ticket_even_if_model_requests_it`；断言 model_calls<=6、tool_calls<=6、估算/实测分开，累计含传入分类 usage，预算不足不再请求模型。ToolExecutor 内部重试也占实际执行计数；扩展可观察执行次数回调，不隐瞒重试成本。
- [x] Run `python -m pytest tests/test_agent_runtime.py -q`，先看到目标行为失败，记录 RED 原因。
- [x] 实现最小循环。Agent 无业务工具调用时结束决策循环，使用现有上下文进行一次无工具绑定的真实流式答复（计入 6 次上限）。裸演示清楚展示这个终止生成步骤，不将其称为免费调用。控制信号 `suggest_actions` 只更新建议，补匹配 ToolMessage，不做数据库写入、不计业务工具；单次控制建议后继续循环，重复控制建议受模型步数限制。
- [x] 缺信息时允许直接终止并追问，输入消息/工具结果保持配对；最终输出使用已有模拟数据与证据限制。演示 CLI：`python -m app.bare_agent --message "先查订单 1001 的订单状态，再查物流"`，通过已有配置构建工具，不启动业务写操作。
- [x] Run `python -m pytest tests/test_agent_runtime.py tests/test_tools.py tests/test_config_prompts.py -q`；通过后记录任务 1 结果并提交 `feat: add bounded bare agent loop`。工单旧语义尚未拆除时不启用线上 Agent 入口。

## Task 2: 七类 JSON 意图识别与固定路由

**Files:** Create `app/intent.py`, `app/intent_prompts.py`, `tests/test_intent.py`, `eval/ch05/intent_cases.json`, `scripts/evaluate_ch05.py`; Modify `app/model_service.py`。

**Interfaces:** `IntentDecision(intent: Literal['物流','订单','商品咨询','退款退货','售后','投诉','闲聊'])` 严格结构；`ModelService.classify_intent(text: str) -> AIMessage`（async，单次模型请求，包含原始 JSON 与 usage）；`classify_intent(text: str, *, model: ModelService) -> tuple[IntentDecision, Usage]`（async，校验上一接口输出）；`route_intent(intent: str) -> Literal['knowledge','business','complaint','chitchat']`；`resolve_reference(text: str) -> str` 原样返回。

- [x] 写固定路由与 JSON 解析失败测试：七类逐项映射、额外字段/非枚举/重复键/无效 JSON 被拒绝、透传不改数字/否定。Run `python -m pytest tests/test_intent.py -q` 观察 RED。
- [x] 实现解析及固定映射，分类只请求一次，不调用旧 QueryUnderstanding；失败作为显式分类错误，不默认为业务类。
- [x] 编写至少 35 条标注样例（每类至少 5 条），覆盖退款与泛售后、订单号与物流、明确投诉、问候；另设 7 条边界样例单列报告。标签先写，之后才能运行模型。
- [x] 纯 prompt 部分按用户要求运行评估代替 TDD：`python scripts/evaluate_ch05.py --suite intent --output eval/ch05/intent_results.json`。验收核心 35 条至少 33 条正确，物流验收句/投诉/纯问候/退款强制路径样例必须全部正确；每条保存预期、预测、调用次数、是否成功，不写原始推理。
- [x] 失败时按误判原因修改 prompt 后复测，保留首轮报告；达到标准后 Run `python -m pytest tests/test_intent.py -q`，即时记录并提交 `feat: add seven-intent deterministic routing`。真实评估依赖不可用则明确阻塞，不用 mock 结果顶替。

## Task 3: 复用 RAG 的前置闸与弱证据记录

**Files:** Create `app/workflow_knowledge.py`, `app/workflow_log.py`, `tests/test_workflow_knowledge.py`, `eval/ch05/knowledge_cases.json`; Modify `scripts/evaluate_ch05.py`, `.gitignore`。

**Interfaces:** `KnowledgeGate.prepare(question: str, filters: KnowledgeFilters | None) -> dict`（async，返回 evidence/citations/confidence/stop_reason）；构造依赖为现有 RagRetrieval、ConfidencePolicy、Settings。`WorkflowLog.write(event: str, payload: dict) -> None` 写 `.runtime/ch05/events.jsonl`；弱证据原问题写独立 `low-confidence.jsonl`，不写现有低置信度数据库表。

- [x] 写测试断言实际调用 retrieve 的 strategy='hybrid_rerank'，PreparedQuery 原样问题，闸分数边界/空证据/索引过期/异常都不放行；通过时证据与引用完整，日志区分 weak_evidence 和 retrieval_error；原问题记录失败不得宣称成功记录。
- [x] Run `python -m pytest tests/test_workflow_knowledge.py -q`，观察 RED，再实现薄适配器与安全字段日志，不修改检索器算法。
- [x] 编写至少 12 条知识标注题（6 可回答、6 无足够证据），执行 `python scripts/evaluate_ch05.py --suite knowledge --output eval/ch05/knowledge_results.json`。固定校准集和独立验收集，不用验收集调阈值；任何弱证据误放行先解决，拒答率单列，不以全部拒答充当通过。
- [x] 若原查询预处理的校准不适用，将本章阈值及查询方式指纹保存 `eval/ch05/confidence.json`，用独立校准样例确定后再跑验收；不覆盖 ch04 未提交产物。Run `python -m pytest tests/test_workflow_knowledge.py tests/test_rag_retrieval.py tests/test_rag_evidence.py -q`，记录并提交 `feat: gate workflow knowledge before agent output`。

## Task 4: 幂等历史、动作与工单持久化

**Files:** Create `app/ch05_db.py`, `app/init_ch05_db.py`, `app/workflow_repository.py`, `db/ch05_schema.sql`, `tests/test_ch05_db.py`, `tests/test_workflow_repository.py`; Modify `app/repository.py`, `app/tools.py`, `tests/test_tools.py`。

**Interfaces:**
- 新增四张旁表，不改旧章节 DDL：`workflow_turns`（turn_id PK、conversation_id FK、status、final_message_id）、`workflow_message_keys`（turn_id+position 唯一，message_id FK）、`workflow_actions`（action_id PK、conversation_id/turn_id/message_id、kind、description、ticket_type）、`workflow_ticket_requests`（request_key PK、conversation_id、payload_digest、ticket_no）。
- `WorkflowRepository.append_message_once(conversation_id: int, turn_id: str, position: int, role: str, content: str | None, *, tool_calls=None, tool_call_id=None) -> str`；`commit_reply(conversation_id: int, turn_id: str, position: int, answer: str, citations: list, suggestions: list[ActionSuggestion]) -> tuple[str, list[dict]]`；`load_actions(conversation_id: int) -> dict[str, list[dict]]`；`set_turn_status(turn_id: str, status: str) -> None`。
- 原 `Repository.create_ticket` 增加仅服务端可传的 keyword-only `request_key: str | None = None`；`build_tools` 增加 `ticket_request_key: str | None = None` 闭包参数，保持模型可见 TicketInput 不变。工具仍是原 create_ticket。

- [x] 写独立 MySQL 测试：历史位置重复写返回同一个消息 ID；同一位置不同载荷报冲突；reply 与建议同事务；同一 request_key 重复/并发确认返回同一 ticket_no，改载荷冲突。断言创建工单后 conversation.status 仍为原状态，不出现“已转人工”成功文案。
- [x] Run `python -m pytest tests/test_ch05_db.py tests/test_workflow_repository.py tests/test_tools.py -q` 观察 RED。
- [x] 实现明确可重跑的新增表初始化 `python -m app.init_ch05_db`，先检查现有 ch04 schema；部分/错误 schema 报错，不丢表重建。工单去重记录与 tickets 插入在同一 MySQL 事务完成，唯一冲突后按数据库提交结果读取，不把失败事务继续当可用会话。description、ticket_type 同 request_key 的摘要必须一致。
- [x] 原 Repository 方法移除 conversation.status='已转人工'；原工具成功文案改为“已创建工单，等待处理”。测试夹具的函数签名按可选参数兼容，不篡改工具业务。
- [x] 写故障注入：消息已写后再次调用不重复，数据库失败无半条建议，已删除会话拒绝写入。Run 上述测试加 `tests/test_db.py tests/test_ch04_db.py`；核对新初始化与旧 DDL 测试均可共存，记录并提交 `feat: persist idempotent workflow messages and tickets`。

## Task 5: StateGraph 固定骨架、Agent 子图与 SQLite

**Files:** Create `app/workflow_graph.py`, `tests/test_workflow_graph.py`, `tests/test_workflow_checkpoint.py`, `requirements-ch05.lock.txt`; Modify `app/config.py`, `requirements.txt`, `.gitignore`。

**Interfaces:** `build_workflow(*, model: AgentModel, classifier, knowledge_gate: KnowledgeGate, tools_factory, limits: AgentLimits, log: WorkflowLog, checkpointer) -> CompiledStateGraph`；classifier 为 `Callable[[str], Awaitable[tuple[IntentDecision, Usage]]]`，tools_factory 为 `Callable[[int], dict]`，只能产出任务 1 的允许工具。Agent 子图复用任务 1 三个步骤。父图中显式 `resolve_reference -> classify_intent -> route`，知识 `retrieve -> confidence_gate -> agent/fallback`，其余按固定表，所有正常结束路径汇合 `log_turn`。

- [x] Context7 定向确认 SQLite `from_conn_string`、生命周期及父图/子图 API；安装前 dry-run 固定依赖解析结果，记录版本。安装到隔离虚拟环境，用 `inspect.signature` 核对文档并做最小临时文件检查点探针；不使用 v3/beta 接口。本地签名不符先纠正规划中的调用形式，语义变更须回报用户。
- [x] 写路径测试：七类分别断言 node trace、模型/检索/工具调用次数，弱证据 agent_calls=0、闲聊仅分类一次、投诉返回两个建议。Run `python -m pytest tests/test_workflow_graph.py -q` 观察 RED。
- [x] 实现 State 与子图，子图 `.compile()` 继承父 checkpointer，不自建 MemorySaver；工具执行和模型节点独立。轮次开始显式清空 evidence/citations/suggestions/budgets；步数上限与递归保险分别生效，工具错误不能导致无穷循环。
- [x] 写真实 SQLite 临时文件测试：关闭 saver 后重新创建图，用同 thread_id 恢复完成消息；另一 thread_id 无历史；父/子图 checkpoint 实际存在；新的普通聊天不继承上一轮投诉建议。Run `python -m pytest tests/test_workflow_checkpoint.py -q`，观察失败再实现生命周期适配。
- [x] 配置 `WORKFLOW_CHECKPOINT_PATH=.runtime/ch05/checkpoints.sqlite`、`AGENT_MAX_MODEL_CALLS=6`、`AGENT_MAX_TOOL_CALLS=6`、`AGENT_TURN_TOKEN_BUDGET=12000`；忽略整个 `.runtime/`。对数字正数与预算关系验证，不加入其他数据库。
- [x] Run `python -m pytest tests/test_workflow_graph.py tests/test_workflow_checkpoint.py tests/test_agent_runtime.py -q` 和 `python -m pip check`，记录并提交 `feat: orchestrate customer service with persistent LangGraph`。

## Task 6: 接入聊天 SSE、历史与取消恢复

**Files:** Create `app/workflow_service.py`, `tests/test_workflow_service.py`, `tests/test_workflow_api.py`; Modify `app/main.py`, `app/history.py`, `app/chat_views.py`, `tests/test_history.py`, `tests/test_chat_api.py`。

**Interfaces:** `WorkflowService.stream_turn(conversation_id: int, message: str, filters=None) -> AsyncIterator[ChatEvent]`；`open_workflow_service(...)` 为应用生命周期使用的 async context manager；继承既有 SSE 名称 session/token/tool_status/citations/done，新增 `actions` 载荷 `{message_id, actions:[{id,kind,label,description,ticket_type}]}`。保留现有 ChatService 供旧章节离线测试，生产 create_app 默认只接本章服务。

- [x] 写测试断言闸完成前无 token，工具选择正文不流出，模型 final stream 未结束时已收到首 token（用阻塞事件控制后续分片，不能仅用全量 ASGI 响应断言）。不得将预生成答案切片冒充流式。
- [x] 写 MySQL/SQLite 双向故障注入、用户取消、删除/并发会话测试；任何持久化失败不能发 completed/done 成功语义，释放会话预留且不重试写工具。Run `python -m pytest tests/test_workflow_service.py tests/test_workflow_api.py -q` 观察 RED。
- [x] 实现 service 以稳定 turn_id/position 调用任务 4 仓储；最终历史先持久化，再等待图调用正常完成后发送 actions/completed。SQLite 检查点失败时历史可能已经存在，标记恢复状态，重入按 ID 去重；不谎称跨库事务。SSE 取消则结算已经启动的数据库事务，记录 incomplete，下一轮从最后完整历史新开轮次。
- [x] 扩展 completed_turns 解析多组 assistant(tool_calls)/tool，重复 ID、缺结果、未完成 final 不进入模型历史；旧会话首次接入图导入完成历史，之后 State 与持久化身份协调，不能每次双重追加。旧 prompt 的“一轮最多一个工具”不进入新 Agent。
- [x] 生命周期集中管理 SQLite、现有模型预热与资源释放；测试必须进入真实 lifespan，不新增运行时兼容分支只为旧 fake 通过。历史接口把动作与对应 message_id 一起恢复。
- [x] Run `python -m pytest tests/test_workflow_service.py tests/test_workflow_api.py tests/test_history.py tests/test_chat_api.py tests/test_conversation_management.py tests/test_chat_views.py -q`，更新旧测试中已被本章有意替换的行为断言，并保留旧服务独立契约。记录并提交 `feat: stream workflow replies and restore complete sessions`。

## Task 7: 独立确认建工单 API

**Files:** Create `app/ticket_actions.py`, `tests/test_ticket_actions.py`; Modify `app/main.py`, `app/schemas.py`。

**Interfaces:** `TicketActions.confirm(conversation_id: int, action_id: str, description: str, ticket_type: str) -> dict`（async）；`POST /v1/conversations/{conversation_id}/tickets`，body 为 `{action_id,description,ticket_type}`，服务端以 action_id 作为 request_key，并从存储确认归属及动作类型。成功返回 `{ticket_no,status,message}`；不接受客户端覆盖会话身份。

- [x] 写测试：建议输出期间 tickets 行数零；取消时没有 HTTP 写请求；第一次确认写一行；相同动作相同参数重试同号；不同参数 409；不存在/跨会话/已删除的动作 404；handoff 动作不能建单；无效类型/空描述 422。
- [x] Run `python -m pytest tests/test_ticket_actions.py -q` 观察 RED，然后实现只调用原 build_tools(...ticket_request_key=action_id)['create_ticket']，不复制其业务逻辑。
- [x] 保持与聊天/删除共享的单会话互斥；生成期间建单返回 409；超时结果未知不假成功、不在后台自动再次调用写工具。已确认旧动作可幂等读取结果，换到其他会话的请求不可用旧 action_id。
- [x] Run `python -m pytest tests/test_ticket_actions.py tests/test_tools.py tests/test_workflow_repository.py -q`，记录并提交 `feat: require explicit ticket confirmation`。

## Task 8: 前端两按钮与确认交互

**Files:** Modify `app/web/index.html`; Create `scripts/validate_ch05_actions.cjs`; Modify existing `scripts/validate_ch04_citations.cjs` 仅在协议兼容所需时。

**Interfaces:** `renderActions(view, actions, conversationId)` 将建议绑定到对应回复；handoff 点击仅本地展示两条固定文字；create_ticket 先展示描述/类型确认再调用任务 7 API。按钮/弹窗逻辑分别管理，不用共享“已处理”状态禁用另一按钮。

- [x] 先写浏览器验收脚本，复用现有 Node/Playwright + 本地 HTTP 夹具模式；运行 `node scripts/validate_ch05_actions.cjs` 观察当前页面缺按钮而失败。未找到 Playwright 时先查依赖位置并读对应技能/官方文档，不依赖猜测路径。
- [x] 实现 actions SSE 与历史恢复；确认框取消不发请求，确认后禁用提交直到返回；重复确认使用同 action_id。所有文案安全文本渲染。转人工只显示 `已转接人工客服` 与 `您好,我是SayHelp,请问有什么可以帮您的`，不发网络请求。
- [x] 验证两种点击顺序、只点一个、均不点继续聊天、取消建单、超时、再次打开已成功动作、含 HTML 的描述、窄屏；切会话时旧流与旧弹窗不污染当前会话，删除后旧按钮不建单。
- [x] Run `node scripts/validate_ch05_actions.cjs`、`node scripts/validate_ch04_citations.cjs`；保存结构化结果 `eval/ch05/actions_ui_results.json`，不记录客户内容。记录并提交 `feat: add independent handoff and ticket actions`。

## Task 9: 整体验收、代码评审与交付

**Files:** Create `scripts/demo_ch05.py`, `eval/ch05/acceptance_cases.json`, `eval/ch05/acceptance_results.json`, `eval/ch05/report.md`; Modify `README.md`, `dev-notes/ch05.md`。

**Interfaces:** `python scripts/demo_ch05.py --mode bare|workflow --case logistics|multi-step|policy|complaint|chitchat`，默认使用固定脚本模型与临时 SQLite/独立业务夹具，演示标注 simulation；`--live` 才调用配置模型和现有只读检索。演示 create_ticket 必须显式交互确认，普通验收不写真实 tickets。

- [ ] 写五条用户验收题、弱证据和缺信息题，以及 Task 1/6/7 的故障测试汇总。固定夹具验证所有路径，再运行真实模型小集（最多 10 个完整聊天请求；分类/检索评估按前述小集），报告模型与工具步数及用量，禁止把 mock 通过冒充真实能力。
- [ ] 真实模型不支持所需调用格式时停下报告证据，不换模型/框架或悄悄取消真流式；真实 Milvus/模型不可用的项目保持未完成，不能用整套 skip 报验收通过。临时 MySQL 复用 TEST_DATABASE_URL 随机库 fixture；没有凭据时只报告缺项，不打印现有 .env。
- [x] 执行 `python -m pytest -q` 全量回归、`python -m pip check`、两份浏览器验收；实际命令、通过/失败/跳过数及原因写入报告。按既有章节要求启用本地真实模型缓存测试；不清空生产知识集合。
- [x] 使用 requesting-code-review 技能进行独立整体评审，覆盖 Review Focus 五项、spec 覆盖和完整 diff。记录结论；Critical/Important 全部修复并只复跑受影响测试，修复影响面广才再次全量回归；不得把评审意见未经核实直接照改。
- [x] README 写清迁移、启动 `python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8001 --workers 1`、裸循环与图演示命令、SQLite 路径和备份限制、弱证据日志、按钮行为及真实/模拟数据界限。
- [ ] 按 verification-before-completion、finishing-a-development-branch 技能核对最终状态；用户要求完成后推送，不能只留本地工作树。验证分支与目标代码一致，保护原有四个修改；需要评审分支则推送并提供审阅入口，不能未经既有授权覆盖运行中的服务。
- [ ] 提交/推送后核对远端 SHA；阶段完成即时追加 finish 记录并同步推送。最终答复给演示命令、实测结果、dev-notes 路径及未解决限制，不宣称本章以外的生产能力。

## 计划自检与执行交接

覆盖关系：裸循环/停止/预算→1；指代/意图/分流→2、5；检索/闸/弱证据记录→3、5；State/checkpointer→5、6；独立按钮/工单复用→4、7、8；日志/流式/历史→3、6；验收/评审/推送→9。Review Focus 五项均有对应失败测试。

执行依赖：1→2→3→4→5→6→7→8→9。为避免共享接口冲突不并行改相同文件；子代理方式也按此顺序逐任务实施与评审。

建议 Native（本会话逐任务实施，最后独立整体评审）：本计划涉及 State、消息身份与 SSE 的连续接口，顺序实施便于保持一致。若用户更重视每个任务的独立审查，可选 Subagent-driven（每任务独立实现者与评审者，最后整体评审）。必须等用户审阅计划并选择执行方式后开始实现。
