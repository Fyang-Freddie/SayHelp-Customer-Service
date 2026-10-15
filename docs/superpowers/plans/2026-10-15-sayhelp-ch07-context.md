# SayHelp 第 7 章会话上下文管理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留会话完整历史的同时，按实际窗口预算实现三层上下文、异步追加摘要、可观测模型输入和多会话侧栏。

**Architecture:** MySQL 保存业务原文、分层锚点和追加摘要，既有 SQLite checkpoint 保存完整 LangGraph 消息。统一预算器和 ContextView 构建模型输入；后台摘要通过短事务租约与幂等提交独立运行，不修改图状态。

**Tech Stack:** LangGraph 1.2.2、langgraph-checkpoint-sqlite 3.1.1、LangChain Core、SQLAlchemy 2.0/MySQL、FastAPI、原生 HTML/JS。

**Spec:** `docs/superpowers/specs/2026-10-15-sayhelp-ch07-context-design.md`，2026-10-15 用户“设计通过”。

**Status:** 计划自检完成，待用户审阅和选择执行方式；未开始产品实现。

## Global Constraints

- 只管当前会话，不跨会话、不存用户画像；不实现历史语义检索和主题重要度选择。
- 基线 `codex/ch06-intent-router` / `7131515`；执行时按 using-git-worktrees 建立隔离 `codex/ch07-context`，导入本章已批准 spec、plan、dev-notes。不重置 main 的用户改动，不覆盖其他聊天工作树。
- MySQL conversations/messages 保存业务原文；新增 conversation_summaries。两个锚点只前移，不搬数据；新工具结果不写 messages，完整轨迹保留 checkpoint。旧数据保留。
- State.messages 使用 add_messages，各节点只发增量；模型 ContextView 不回写 reducer。保留现有 interrupt、恢复幂等、退款安全兜底与引用证据。
- 固定 system 与 tools 前缀；层2 → 层1 → 当前用户 → 一条非system摘要/证据背景 → 当前ReAct轨迹。层1不裁字，层2用户原文不动、客服前60字符、工具单行标识且配对。
- 计数统一 chars_per_token=1.0、extra_tokens_per_message=3.0；H=min(N×T_steady,max(0,W−F−R−S−O−U−G−P))；L1=7H//10，L2=H−L1。
- 初始 N=32、T_steady=1024、S0=512、A=512、B≥32、G=ceil(W×5%)；P=K×(A+X+B)。默认候选 W=65536、O=2000、U_max=2000、K=3、X=1200、RERANK_TOP_K=5，R_max=5×512。窗口需核对实际模型支持。
- 摘要目标50–200中文字符，只提炼事实/诉求/未解问题，旧摘要只作背景且不重写。未摘要数据不因后台延迟而偷偷丢弃；超限给明确受控提示。
- 涉及具体库接口先 Context7，再核对安装版签名；不能擅换技术。代码 RED/GREEN，纯 Prompt/数据用冻结标注评估。
- 每个完成阶段即时追加 dev-notes/ch07.md 四项：用户关键原话、关键产出、拒绝/纠偏、翻车与返工。每个任务只暂存自身文件，验证后提交推送；不提交 .env、日志原文、凭据、虚拟环境或真实订单截图。

## Review Focus

1. 租约到期后旧任务比新任务晚返回，且层1边界已再次前移：旧任务不能重复摘要或跳过未覆盖消息（Task 2/5）。
2. 同会话 interrupt 恢复、断线重试、checkpoint 已完成但 MySQL 回执失败：完整历史和摘要边界不得重复或丢失（Task 3/7）。
3. 多工具 batch、长编号、中文/emoji 与旧工具行混合：统一计数不漏协议字段，任何裁剪不能产生孤立 ToolMessage（Task 1/4/7）。
4. 访客A猜到访客B会话ID，或经旧/v1写入口访问：列表、原文、恢复及动作均拒绝串会话（Task 8）。
5. UI先请求会话A后切到B，A响应更晚；同时摘要/历史接口失败：B不被覆盖，聊天仍可用（Task 9）。

## 文件职责与公共数据契约

- 新建 `app/context_types.py`：不可变 dataclass `BudgetLedger`、`TurnRecord`、`ContextSnapshot`、`ContextView`、`SummaryClaim`。消息类型沿用 BaseMessage。
- `BudgetLedger` 字段：window/prefix/evidence/summary/output/user/safety/react_peak/history/layer1/layer2（int）、estimated（bool）。
- `TurnRecord` 字段：turn_id（str）、first_msg_id/last_msg_id（int）、messages（tuple[BaseMessage,...]）、completed（bool）。只包含本会话完成轮次；工具归属由 turn_id 确定。
- `ContextSnapshot` 字段：conversation_id/version（int）、summary_upto_msg_id/layer1_from_msg_id（int|None）、summaries（tuple[dict,...]）；每段含 ordinal/from_msg_id/upto_msg_id/content。
- `ContextView` 字段：snapshot、layer1/layer2（tuple[BaseMessage,...]）、summary_text（str）、ledger、layer1_from_msg_id（int|None）、pending_turns（tuple[TurnRecord,...]）。新边界需持久化成功后才使用，不以本地值冒充已提交边界。
- `SummaryClaim` 字段：conversation_id/ordinal/from_msg_id/upto_msg_id（int）、previous_summary_upto（int|None）、lease_token（str）、source_digest（str）、expires_at（datetime）。租约和摘要边界作为提交条件，不能要求全局version不变而阻止并发正常降级。
- 新建 `context_budget.py`、`context_history.py`、`context_repository.py`、`context_summary.py`、`context_log.py` 分别负责算术/视图、存储、任务与完整输入日志。`summary_prompts.py` 只负责摘要提示与返回结构。
- 修改 `workflow_service.py`、`workflow_graph.py`、`agent_runtime.py`、`query_resolver.py` 只在现有接缝接入，避免把全部功能塞进大文件。

## Task 1: 统一预算与可执行预算检查

**Files:** Create `app/context_types.py`, `app/context_budget.py`, `app/ch07_evaluation.py`, `scripts/check_ch07_budget.py`, `tests/test_ch07_budget.py`; Modify `app/config.py`, `requirements.txt`, `.env.example`（若已有则更新）。

**Interfaces:** Produces `estimate_tokens(messages: Sequence[BaseMessage], tools: Sequence[dict]=()) -> int`；`calculate_budget(*, window:int, prefix:int, evidence:int, summary:int, output:int, user:int, steps:int, selection:int, tool_results:int, protocol:int, target_turns:int, steady:int) -> BudgetLedger`；`validate_request(messages, tools, output:int, window:int) -> int`，超限抛 InputBudgetExceeded。预算 CLI 使用真实 prompt/tools，不执行工具业务。`EvaluationBudget.reserve(provider:str,bucket:str)->str` 在 `.runtime/ch07/eval-usage.sqlite` 原子预占额度并返回调用ID；`finish(call_id:str,outcome:str,usage:dict)->None` 只补结果、不退回已发出请求额度。所有live入口共享账本，不随output路径变化重置。

- [ ] 读取隔离基线与依赖签名；通过 Context7 核对计数、工具schema、trim_messages。冻结实际库版本到依赖要求及预算报告，不能只升级到当前最新。
- [ ] 写 `test_budget_ledger`：固定输入 F=3666/R=2560/S=512/O=U=2000/K=3/A=512/X=1200/B=32，W=18000断言 `(history,layer1,layer2)==(1130,791,339)`；W=65536断言 `(32768,22937,9831)`。另测参数变更会重算、中文/emoji/调用参数被计入、启动装不下一轮报警、真实载荷超限拒绝、EvaluationBudget并发预占和重开后不重置额度。
- [ ] RED：`python -m pytest tests/test_ch07_budget.py -q`，确认因缺少实现或行为不符失败。
- [ ] 实现公共接口和配置：用户新变量优先；旧 CONTEXT_TOKEN_BUDGET/RESPONSE_TOKEN_RESERVE 作为有日志的兼容别名，冲突时新变量优先。实现 `--profile default|demo --json`，输出模型能力是否核实与所有分项。累计费用上限与窗口独立，派生上限覆盖各调用真实输入/输出预留，显式旧累计配置仍执行且给出不足告警。
- [ ] GREEN：相同测试通过，CLI两配置输出可重算；未知模型能力不假称已验证。提交 `feat(ch07): derive and expose context budgets`，即时记录并推送。

## Task 2: 分层元数据、追加摘要与租约事务

**Files:** Create `app/ch07_db.py`, `app/context_repository.py`, `app/init_ch07_db.py`, `db/ch07_migration.sql`, `tests/test_ch07_repository.py`; Modify `app/db.py`, `app/init_db.py`。

**Interfaces:** `ContextRepository.load(conversation_id:int)->ContextSnapshot`；`advance_layer1(conversation_id:int, expected_version:int, from_msg_id:int)->bool`；`claim(conversation_id:int, from_msg_id:int, upto_msg_id:int, digest:str)->SummaryClaim|None`；`complete(claim:SummaryClaim, content:str, estimated_tokens:int)->bool`；`fail(claim:SummaryClaim, reason:str)->None`。租约默认120秒、模型超时45秒，任务期间续租；网络调用期间不持有SQL事务。

- [ ] RED测试 `test_complete_is_atomic_and_idempotent`、`test_late_expired_claim_cannot_commit`、`test_downgrade_during_summary_does_not_skip_range`：双Session模拟争用；插入摘要失败则锚点不变；旧租约结果无法写入；降级不阻止合法旧批提交但不能扩大覆盖上界。
- [ ] `python -m pytest tests/test_ch07_repository.py -q` 确认失败；通过Context7核对 Session.begin/with_for_update 后实现短事务与唯一约束。
- [ ] 迁移使用可重入schema版本检查；记录本会话合法业务ID和完成轮次校验。新增表/列不删除旧工具消息；批次指纹使用规范化来源内容。
- [ ] GREEN：上述测试和真实 MySQL 专用测试库通过；SQLite测试不能冒充行锁验证。`python -m app.init_ch07_db` 重跑无损。提交 `feat(ch07): persist context anchors and summary leases`，记录并推送。

## Task 3: 完整图历史与业务原文对账

**Files:** Create `app/context_history.py`, `tests/test_ch07_full_history.py`; Modify `app/workflow_service.py`, `app/workflow_repository.py`, `app/workflow_types.py`。

**Interfaces:** `reconcile_history(rows:Sequence, checkpoint_messages:Sequence[BaseMessage], turn_rows:Sequence) -> tuple[TurnRecord,...]`；`new_turn_messages(turn_id:str,text:str)->list[BaseMessage]` 返回仅新增用户消息。首次导入用 `db:<id>` 稳定ID；新消息用 `turn_id:position`，业务映射写入已有 WorkflowMessageKey。

- [ ] RED `test_full_history_survives_24_turns_and_reopen`、`test_reply_reconciliation_after_mysql_failure`、`test_legacy_partial_checkpoint_imports_only_missing_messages`：重开SQLite后完整消息顺序/正文一致，重复提交不翻倍，缺失旧工具只记缺口。
- [ ] `python -m pytest tests/test_ch07_full_history.py -q` 确认失败，再移除正常新轮次的 REMOVE_ALL_MESSAGES，稳定ID增量导入。只在迁移时识别旧system/动态证据，不把其误当用户历史；不删除用户原文。
- [ ] 修改 `_persist_reply`：仅落用户/最终助手业务原文，保留原有引用、动作和回执；工具调用/返回通过checkpoint关联，不再插入新工具行。未完成轮次保持审计但不进入可摘要 TurnRecord。
- [ ] GREEN：本任务测试及 `tests/test_workflow_checkpoint.py tests/test_workflow_repository.py tests/test_ch06_recovery.py` 通过。提交 `feat(ch07): preserve complete checkpoint history`，记录并推送。

## Task 4: 分层视图与固定输入装配

**Files:** Extend `app/context_history.py`; Create `tests/test_ch07_context_view.py`; Modify `app/history.py`, `app/prompts.py`。

**Interfaces:** `build_context_view(snapshot:ContextSnapshot, turns:Sequence[TurnRecord], ledger:BudgetLedger)->ContextView`；`compose_model_messages(system:SystemMessage, view:ContextView, current:HumanMessage, evidence:Sequence[dict], active:Sequence[BaseMessage])->list[BaseMessage]`。compose不修改输入对象。

- [ ] RED `test_layer1_verbatim_and_layer2_rules`、`test_no_downgrade_when_history_fits`、`test_fixed_prefix_and_background_order`：用户逐字符相等、客服前60字符、工具成对且结果单行；旧摘要非system；本轮消息只出现一次。
- [ ] `python -m pytest tests/test_ch07_context_view.py -q` 确认失败后实现 trim_messages 统一计数与完整轮次边界；候选截断不能把有tool_calls的助手消息当最终答复。
- [ ] 添加 `test_pending_summary_uses_remaining_capacity_or_fails_closed`：L2可以借剩余空间，但不漏尚未摘要轮次；全量载荷超限则抛预算异常，不隐式丢事实。摘要持续增加时重算S，不能截旧摘要。
- [ ] GREEN：本任务测试通过；固定system补充摘要资料不可信及来源区分指令，修正现有“订单数据源尚未接入”与ch06实际接入矛盾，再跑Task1预算CLI更新实际F，不保留过期3666作为产品硬编码。提交 `feat(ch07): build layered model context views`，记录并推送。

## Task 5: 非阻塞摘要工作器与来源评估

**Files:** Create `app/context_summary.py`, `app/summary_prompts.py`, `tests/test_ch07_summary.py`, `eval/ch07/summary-cases.json`, `scripts/evaluate_ch07_summary.py`; Modify `app/model_service.py`。

**Interfaces:** `SummaryModel.summarize(*, background:str, turns:Sequence[TurnRecord])->str` 异步；`SummaryWorker.submit(claim:SummaryClaim, turns:Sequence[TurnRecord], background:str)->None` 立即返回；`await SummaryWorker.close(timeout:float=5.0)` 有界关闭；`await SummaryWorker.recover()->None` 扫描过期任务后重新领取。所有调用走计量器。

- [ ] RED `test_slow_summary_does_not_block_reply` 用Event阻塞模型但断言回复先完成；另测重复触发、超时、取消、过期结果、空闲聊、异常退出后的恢复。
- [ ] `python -m pytest tests/test_ch07_summary.py -q` 确认失败，实现租约、冻结来源、摘要成功追加事务。每批失败最多2次重试且计入总请求上限；续租失败停止提交。长批按完成轮次分成能放入摘要模型窗口的连续子批，逐段提交，不跳号、不合并旧摘要。
- [ ] 先冻结24个标注样例：商品切换、多个订单、手机号、否定/纠正、未解决诉求、纯闲聊、工具事实与用户说法冲突、旧摘要诱导重复、注入文本。期望字段和禁止新增事实在运行前确定；Prompt步骤以真实评估代替TDD。
- [ ] 实现输出长度/编号来源检查；不通过则失败不前移，不靠截字修复。语义验收核对每条有依据、关键订单/诉求保留、无新增事实；冻结结果不回填标签。`python scripts/evaluate_ch07_summary.py --live --max-calls 24` 在计划批准和实现后运行，未通过则记录原始结果，不宣称成功。
- [ ] GREEN确定性测试及独立语义结果通过后提交 `feat(ch07): summarize old turns asynchronously`，记录并推送；若语义未通过则保持任务未完成并报告，不伪造通过。

## Task 6: 完整输入日志与共同历史

**Files:** Create `app/context_log.py`, `tests/test_ch07_context_log.py`; Modify `app/query_resolver.py`, `app/ch06_prompts.py`, `app/workflow_graph.py`, `app/model_service.py`, `.gitignore`。

**Interfaces:** `ContextLog.write(event:str,payload:dict)->None` 写JSON行到 `log/app.log`；`log_history(conversation_id:int,turn_id:str,view:ContextView)->None`；`log_model_request(conversation_id:int,turn_id:str,phase:str,messages:Sequence[BaseMessage],tools:Sequence[dict],ledger:BudgetLedger)->None`。QueryResolver.resolve 新增 keyword-only `context:ContextView|None=None` 保留旧调用兼容。

- [ ] RED `test_logged_request_equals_model_spy_payload`、`test_chitchat_and_resume_always_log_history`：逐条内容/角色/调用ID/tools与实际传参一致；摘要全文可读，不被现有WorkflowLog白名单丢掉。
- [ ] `python -m pytest tests/test_ch07_context_log.py -q` 确认失败，实现专用上下文日志；只记录模型请求业务载荷，不包含客户端headers、API key、隐藏推理或provider凭据。将log目录加入忽略，保留原workflow事件日志兼容。
- [ ] 替换 resolver 独立8条/6000字裁剪；编号守卫允许来自已验证摘要但不能从任意助手幻觉添加。Jev仍只接收消解后的Query。新增 `test_resolver_uses_summary_order_id_and_rejects_new_id`，摘要生命周期事件含边界/段号/耗时，日志失败明确暴露诊断而不伪报已记录。
- [ ] GREEN：本任务及 `tests/test_query_resolver.py tests/test_jev_classifier.py` 通过。提交 `feat(ch07): log exact model and shared history context`，记录并推送。

## Task 7: 图运行时、峰值限制与应用生命周期集成

**Files:** Create `tests/test_ch07_workflow.py`; Modify `app/workflow_service.py`, `app/workflow_graph.py`, `app/agent_runtime.py`, `app/aftersales_workflow.py`, `app/workflow_knowledge.py`, `app/evidence.py`, `app/main.py`。

**Interfaces:** 工作流装配注入 ContextRepository/SummaryWorker/ContextLog；每轮加载snapshot/TurnRecord并计算视图，CAS推进layer1后重读；L2实际超限才claim/submit。agent_step/execute_calls/stream_answer 接收完整State但通过Task4 compose得到独立模型输入，返回增量。

- [ ] RED `test_tool_batch_peak_is_enforced`：多个工具结果合计不能超过X，调用参数/协议不能绕过预算；`test_interrupt_resume_preserves_history_and_budget`：恢复不重置费用/步数；`test_background_completion_does_not_mutate_running_state`：活跃轮用冻结视图。
- [ ] `python -m pytest tests/test_ch07_workflow.py -q` 确认失败；接入预算硬闸、最多K次选择+一次最终生成、已有低置信分流、证据完整条目准入。resolution/Jev/售后调用单独计费但共享本轮累计上限，各自请求也有窗口检查。
- [ ] FastAPI lifespan启动工作器、过期租约恢复及预算自检，关闭时有界清理；不得每请求创建不受管理的后台任务。Task6日志覆盖每次主力调用和不进Agent的轮次。
- [ ] GREEN：本任务及 `tests/test_agent_runtime.py tests/test_workflow_graph.py tests/test_workflow_service.py tests/test_ch06_interrupts.py tests/test_ch06_recovery.py tests/test_aftersales_workflow.py` 通过。提交 `feat(ch07): integrate context lifecycle into workflow`，记录并推送。

## Task 8: 访客归属与两个只读接口

**Files:** Create `app/conversation_identity.py`, `tests/test_ch07_conversation_api.py`; Modify `app/repository.py`, `app/main.py`, `app/schemas.py`。

**Interfaces:** `get_or_create_visitor(request:Request)->tuple[str,bool]` 产生不透明高熵token，数据库user_id保存其哈希；`assert_owner(conversation_id:int,visitor_id:str)->None`；`list_user_conversations(visitor_id:str,limit:int=30,cursor:str|None=None)->dict`；`user_messages(visitor_id:str,conversation_id:int)->dict`。

- [ ] RED `test_two_visitors_cannot_read_or_resume_each_other`，覆盖/api、/v1、chat、pin/delete、工单、退款及interaction恢复；`test_new_conversations_share_visitor_not_customer_id`；过期/伪造cookie不能认领旧历史。
- [ ] `python -m pytest tests/test_ch07_conversation_api.py -q` 确认失败。设置HttpOnly/SameSite=Lax cookie，HTTPS启用Secure；在实际返回的JSON/SSE Response上设置，不能依赖临时Response自动合入直接返回的SSE。
- [ ] 实现GET `/api/conversations` 返回id/首问预览/has_summary/updated_at及游标，按最近活动降序；GET详情回载原文与现有交互元数据，拒绝归属不匹配。旧/v1保持响应结构兼容但同样校验归属；无旧凭证的会话不自动认领，开发迁移需显式指定会话ID和目标访客，不提供公共认领接口。
- [ ] GREEN：本任务及 `tests/test_conversation_management.py tests/test_workflow_api.py tests/test_ch06_interaction_api.py tests/test_refund_api.py` 通过。提交 `feat(ch07): scope conversation history to visitors`，记录并推送。

## Task 9: 多会话侧栏与浏览器验证

**Files:** Modify `app/web/index.html`; Create `scripts/validate_ch07_browser.cjs`, `eval/ch07/browser-acceptance.md`。

**Interfaces:** 消费Task8列表/详情；侧栏摘要标记用has_summary，完整聊天用messages；保留当前SSE/interrupt/action契约。

- [ ] 先建立可复现浏览器断言：A/B会话切换和晚返回、首问预览、已摘要标记、新会话后旧会话仍在、历史加载错误不打断发消息。记录旧行为失败；原生静态样式不写镜像单测，交互行为用浏览器验证。
- [ ] 修改现有侧栏，改用/api读接口并维护请求版本/AbortController；所有HTML显示用安全文本节点，预览不当HTML执行。只在切换成功且请求仍最新时替换聊天区。
- [ ] `node --check scripts/validate_ch07_browser.cjs`；从HTML提取脚本进行语法检查；真实浏览器逐项通过并记录。加载失败静默，原文回载不显示摘要替代文本；刷新旧售后卡片仍能恢复。
- [ ] 提交 `feat(ch07): show scoped sessions and summary status`，即时记录浏览器实际结果并推送，不将尚未跑的操作写为通过。

## Task 10: 两配置24轮演示、全量回归、评审与交付

**Files:** Create `scripts/demo_ch07.py`, `tests/test_demo_ch07.py`, `eval/ch07/dialogue-cases.json`, `eval/ch07/acceptance-report.md`; Modify `README.md`, `dev-notes/ch07.md`。

**Interfaces:** `python scripts/demo_ch07.py --profile default|demo --live --init-db --output <path>`；脚本先检查MySQL/checkpoint/模型能力/请求余额，再运行固定24轮，在HTTP层保留同一cookie。`--dry-run` 不调用模型。消费Task1的EvaluationBudget，所有live入口共享同一账本。

- [ ] RED CLI测试：参数错误、依赖不可用、预算超限、请求额度到达均退出非0并保留报告；`python -m pytest tests/test_demo_ch07.py -q` 失败后实现脚本。
- [ ] 在任何真实运行前冻结24轮：首轮订单/诉求、后续换话题与大工具结果、末轮“最开始那个订单后来怎么说”。默认样例完整历史须先离线证明能装入层1；不通过则调整已标注样例的冗余长度或公开调整稳态/目标预算，不偷偷扩大模型窗口。两配置必须使用同一组内容。
- [ ] 运行默认和演示各24轮，存储完整日志在忽略目录、脱敏报告在eval。默认零降级/零摘要；演示必须有层1降级、L2超限trigger、后台done、早期订单号及诉求回答正确。记录实际prompt/schema指纹和预算，不断言原始5650。
- [ ] 分离16次以内的中文token校准调用，报告估算/provider usage偏差；参数修订必须在冻结最终验收前完成。复用前面24条摘要评估结果，慢任务用故障注入证明回复不await摘要，不以一次低延迟截图代替。
- [ ] 执行 `python -m pytest -q`、专用MySQL事务测试、checkpoint重开和浏览器验收；逐项说明基线失败/环境跳过，不把SQLite替身当MySQL通过。核心新增失败必须修复。
- [ ] 使用 requesting-code-review，独立review覆盖本章整体diff和Review Focus，修复后针对性复验；即时记录评审结论。按 finishing-a-development-branch 推送已验证分支，能创建PR则附到本聊天；外部权限拒绝如实报告，不将其算代码完成证据。
- [ ] README给出启动迁移、两种演示配置、预算检查、`rg 'model_ctx|history_ctx|summary|层1' log/app.log`、测试结果与开发记录路径；记录finish，仅当全部已要求验收确实通过才宣称完成。提交 `docs(ch07): record verified context-management delivery` 并推送。

## 真实调用上限与评估边界

计划批准后，本章上限：聊天上游600次、Jev64次，所有失败/SDK重试计入；禁止SDK隐式重试绕过计量。不是要求跑满，未用额度不补跑。

- 两配置48轮主链路：最多384次聊天（每轮最多8次，含消解/工具选择/最终生成/必要知识处理）。
- 摘要Prompt标注评估24次、token校准16次、后台摘要48次、浏览器最多8轮×8=64次、显式失败重试共64次；合计384+24+16+48+64+64=600。
- Jev：48轮 + 8轮浏览器 + 最多8次失败重试 =64。摘要评估和token校准不调用Jev。
- 统一持久化计量器在调用前原子预占额度，重启不重置；按功能分别记账。子项和总项均不得超限，真实语义失败不在已看过的保留集上回调；需要新增评估先报告并等待用户授权。
- 不执行ch06 remediation-plan中的新增校准，不改其置信阈值。若ch06保守兜底阻断本章端到端追问，保留失败结果并区分上下文是否正确注入，不能用mock成功替代真实验收。

## 计划自检和执行交接

覆盖映射：spec §2→Task2/3；§3→Task3/7；§4→Task4/6；§5→Task1/7/10；§6→Task2/5/7；§7→Task6/8/9；§8→Task10。公共类型和接口在上述定义，不依赖隐式跨任务约定。

自检纠正：摘要提交不能被无关layer1版本变化误判过期；SSE cookie必须设置到实际响应；旧/v1路由不能绕过访客归属；48轮费用包含消解与知识处理；当前旧system“数据源未接入”需同步真实ch06能力且重新测F。

推荐Native执行：十个任务在State、预算和对账上高度依赖，当前会话连续实现较省重复上下文，最后进行独立整体评审。也可选择Subagent-driven，每任务由新实现代理和新评审代理检查。用户审阅本计划并选择方式后才开始产品实现。
