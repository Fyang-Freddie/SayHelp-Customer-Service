# SayHelp 第 6 章正式分流器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 Jev 八类意图分类、历史查询理解和可恢复的订单选择，将占位分流器升级为有真实评估证据的退款售后 Workflow。

**Architecture:** 在已实现的第5章外层图中增加独立查询理解、Jev适配、集中策略、订单选择和政策检索节点。订单Markdown精确读取，政策沿用MySQL/Milvus检索；官方SQLite checkpointer负责暂停恢复，退款申请事务保存在MySQL。前端保留原生聊天页，按用户要求直接迭代。

**Tech Stack:** Python、FastAPI 0.142.2、SQLAlchemy 2.0.54/MySQL、LangGraph 1.2.2、langgraph-checkpoint-sqlite 3.1.1、langchain-core/langchain-openai 1.6.6/1.6.7、PyMilvus 2.6.x、已有embedding/reranker、httpx、原生HTML/JS。

**Spec:** `docs/superpowers/specs/2026-10-10-sayhelp-ch06-intent-router-design.md`；用户已于2026-10-10明确批准。此计划仍待用户审阅并选择执行方式。

## Global Constraints

- 仅 Jev System One 的 choice 接口进行八类单标签意图识别；不使用通用 LLM 分类、few-shot 分类或另一个分类模型兜底。
- Jev state 仅为处理后的完整 Query；业务输出只含 intent、confidence。分类失败及低置信由集中代码策略处理。
- 退款退货与售后先拿订单，再在检索侧扩写并强制查政策，最后交主力 Agent 判断。知识入库不为扩写复制多份。
- 缺客户ID由主力 Agent 询问。订单号可询问或由聊天卡片点选回填；缺失时不猜。退款原因只在提交表单时从固定类目选择。
- 订单卡片使用 LangGraph interrupt，选择后恢复；前端直接按效果迭代，不套 brainstorm/TDD/code review。
- 后端走 Superpowers；纯 Prompt、criteria、数据部分用标注评估替代 TDD。无跨会话记忆、微调、额外本地分类模型。
- 用户提供的数据源位于 knowledge_db；本章不增加登录体系。客户ID是用户声明的查询范围，不等同于经过认证的身份。
- 实现基线为 codex/ch05-workflow 的 fe16a9c；沿用单 worker、MySQL业务存储、官方SQLite checkpointer。若分支已前进，先检查新提交并记录实际基线。
- 不覆盖main已有ch04计划/校准文件/test.py，不提交.env、凭据、虚拟环境或本地数据库备份；完成并验证后commit并push指定GitHub仓库。
- 每完成一个任务、评审、finish立即追加dev-notes/ch06.md四项记录，不在收尾一次性补记。新库/API先查Context7或Jev官方文档，版本矛盾先问用户。

## Review Focus

1. 等待订单时“先不退了，查物流”应结束旧等待并重新分类，旧卡片不可再触发退款；任务6/7测试覆盖。
2. 一个订单多商品项，或换客户后沿用旧item_id，不能误选/越出当前声明客户范围；任务1/6/8覆盖。
3. SQLite暂停成功而MySQL展示状态写入失败，以及点选后断线/重复点击，不应重复调用Jev或重复建单；任务5/7/8覆盖。
4. 售后手册content_type=manual，退款检索仍必须召回；不允许商品说明书挤掉政策或用不同Query重排分数比较；任务4覆盖。
5. 完整句改写、否定/数字丢失、原因下拉改变资格判断前提、模型别名漂移，不得静默保持旧资格；任务2/3/8覆盖。

## 文件与接口总览

- `app/ch06_types.py`：严格内部契约。`IntentDecision(intent, confidence)`八类且禁止额外字段；`IntentOutcome(decision, reason, model, probabilities, usage)`仅服务端；`Resolution(query, changed, unresolved_references)`；`Expansion(queries)`唯一字段且3条非空；`Eligibility(status, explanation, evidence_ids, missing_facts)`中status为eligible/ineligible/needs_information。
- `app/business_data.py`：`OrderItem(customer_id, order_id, item_id, product_name, variant, quantity, paid_amount, ordered_at, signed_at, order_status, aftersales_status)`；`BusinessSnapshot(items, logistics, reasons, digest)`；`MarkdownBusinessData`只读解析与查找。
- `app/jev_classifier.py`、`app/intent_policy.py`、`app/intent_criteria.py`：HTTP、验证、criteria、校准加载与统一失败策略；`app/intent.py`只保留业务兼容出口和路由。
- `app/query_resolver.py`、`app/query_expansion.py`、`app/ch06_prompts.py`：复用聊天上游的查询处理；不返回八类意图。
- `app/policy_retrieval.py`：多查询合并与政策证据；复用`RagRetrieval`、`KnowledgeGate`及现有完整条款选择。
- `app/ch06_db.py`、`app/ch06_repository.py`、`app/init_ch06_db.py`、`db/ch06_schema.sql`：等待交互、恢复请求、退款动作与申请的存储/幂等。
- `app/aftersales_workflow.py`、`app/workflow_interactions.py`：确定性子流程、槽位澄清与暂停；`workflow_graph/service/types/repository/log.py`做必要接入。
- `app/refund_service.py`：依据持久化动作复核并创建申请；`app/main.py`、`app/schemas.py`接API；`app/web/index.html`接聊天卡片和表单。
- `eval/ch06/`与`scripts/evaluate_ch06.py`、`scripts/demo_ch06.py`：数据、指纹、真实报告和演示；默认不付费，真实调用必须显式`--live`，不提供模拟业务输出模式。

状态必须可JSON序列化；Decimal/时间在图中用字符串，模型资源、HTTP客户端与数据库会话放依赖闭包。意图置信与检索置信分别存，不覆盖已有confidence字典。

## 文档核查和模型预检

已完成GET https://api.typesafe.ai/v1/models，HTTP200，实际返回jev-latest、jev-preview；证据`eval/ch06/model-discovery.json`。已核对Jev请求/响应、LangGraph interrupt、FastAPI SSE、LangChain json_mode、SQLAlchemy事务及PyMilvus AnnSearchRequest。

执行时检查安装版签名，不照搬Context7混入的最新版`stream_events(version='v3')`或ORM Collection.hybrid_search示例。保留现有`astream`及`MilvusClient.hybrid_search`调用方式。

Jev协议验证先从配置读取稳定别名jev-latest，只跑1条有预期标签的非敏感样例；从返回model取得固定版本，再通过JEV_MODEL配置后续评估。不得把列表中的别名当作已验证固定版本；新版本/criteria变更使旧阈值失效。

## 任务1：可验证的Markdown业务数据接口

**Files:** Create `app/business_data.py`, `tests/test_ch06_business_data.py`, `tests/fixtures/ch06/`；Modify `app/tools.py`, `tests/test_tools.py`, `dev-notes/ch06.md`；将用户提供的6个knowledge_db文档作为当前来源快照纳入本章分支。

**Interfaces:** `MarkdownBusinessData(root: Path).load() -> BusinessSnapshot`；`list_items(customer_id: str) -> list[OrderItem]`；`get_item(customer_id: str, order_id: str, item_id: str) -> OrderItem`；`get_logistics(customer_id: str, order_id: str) -> dict`；`refund_reasons() -> list[dict[str,str]]`。失败抛`BusinessDataError(code, source, line)`，不返回伪订单。

- [ ] 执行using-git-worktrees：先list_artifacts，再建/复用本聊天隔离分支codex/ch06-intent-router，从上述ch05基线开始；按文件带入已提交spec/plan/记录，以及当前用户数据，不拷贝其他未提交变更。服务凭据用明确本地路径注入，不混入提交。
- [ ] 写失败测试`test_multiline_order_selects_exact_item`、`test_wrong_customer_rejected`、`test_invalid_row_reports_source_line`、`test_paid_amount_is_item_total`；断言重复ID/同单跨客户/NaN金额/负数量/畸形日期/孤立物流拒绝，未知签收保持未知。
- [ ] Run `python -m pytest tests/test_ch06_business_data.py -q`，确认缺实现导致失败。
- [ ] 实现严格表头解析、规范化及文件digest；订单时间状态矛盾产生可定位的诊断并禁止相关资格判断，不自动改文件。列表按下单时间倒序、相同时按订单/item ID稳定排序。
- [ ] 把query_order/query_logistics接入数据接口，客户身份从当前会话绑定而非模型任填；保留未接入/无记录显式错误。模型仍不能直接创建退款申请。
- [ ] Run上述测试及`tests/test_tools.py`；对真实目录只运行解析/一致性检查，保存`eval/ch06/data-validation.json`，失败报告真实原因并修复读取逻辑或请用户纠正来源。
- [ ] 更新四项记录，提交并推送`feat(ch06): read scoped orders and logistics from Markdown`。

## 任务2：Jev客户端、严格结果契约和集中置信策略

**Files:** Create `app/ch06_types.py`, `app/jev_classifier.py`, `app/intent_policy.py`, `app/intent_criteria.py`, `tests/test_jev_classifier.py`, `tests/test_intent_policy.py`, `scripts/check_jev.py`；Modify `app/intent.py`, `app/config.py`, `app/model_service.py`, `tests/test_intent.py`。

**Interfaces:** `JevIntentClassifier(settings, client).classify(query: str) -> IntentOutcome`（async）；`IntentPolicy.from_artifact(path: Path, expected: dict) -> IntentPolicy`；`IntentPolicy.apply(outcome: IntentOutcome) -> IntentOutcome`；`route_intent(intent: str) -> Literal['knowledge','business','aftersales','complaint','chitchat','fallback']`。`decision.model_dump()`只含intent/confidence。

- [ ] 按已批准spec落实八类criteria与instructions，写真实样例预期供协议验证；criteria不是few-shot。纯文本不做镜像单测，用任务3的标注集评估。
- [ ] 写失败测试：完整state字符串直传且没有历史/system prompt；候选恰为八类；失败/额外字段/缺概率/非枚举/布尔数值/NaN/重复JSON键拒绝；低置信输出其他并保留真实confidence，服务端reason明确。
- [ ] Run `python -m pytest tests/test_jev_classifier.py tests/test_intent_policy.py -q`，确认RED。
- [ ] 实现httpx异步封装、有限超时、取消传播、默认不自动重试，禁止重定向携带认证头；密钥repr隐藏。概率和容差为1e-6，choice须在最大概率并列集合中。JEV_MODEL从配置读取，程序不散落版本名。
- [ ] 实现统一策略：valid_other/low_confidence/timeout/provider_error/invalid_response/calibration_missing/calibration_mismatch；API错误decision为其他/0.0。threshold只能由有效评估产物提供，缺失不启用自动业务路线。
- [ ] Run测试加旧intent契约回归；删除生产路径的通用LLM意图分类，旧ch05评估报告保留但不再作为当前验收。
- [ ] Run `python scripts/check_jev.py --live --output eval/ch06/protocol-check.json`，仅1次choice请求，确认actual model后配置JEV_MODEL；此请求不宣称已校准。验证密钥不在结果/异常/日志中。
- [ ] 记录API核查、真实调用次数、结果，提交并推送`feat(ch06): classify intent through Jev choice`。

## 任务3：历史查询理解与真实意图校准

**Files:** Create `app/query_resolver.py`, `app/ch06_prompts.py`, `tests/test_query_resolver.py`, `app/evaluation/ch06.py`, `tests/test_ch06_evaluation.py`, `eval/ch06/intent-cases.json`, `eval/ch06/dialogue-cases.json`, `scripts/evaluate_ch06.py`；Modify `app/model_service.py`。

**Interfaces:** `QueryResolver(model).resolve(raw: str, history: list, confirmed: dict) -> tuple[Resolution, Usage]`（async）；`ModelService.resolve_query(messages: list) -> AIMessage`（async）；`evaluate_intents(cases, resolver, classifier, policy, output: Path) -> dict`（async）；`calibrate_intents(results: list[dict], fingerprints: dict) -> dict`。

- [ ] 写控制流失败测试：changed=false逐字返回raw，已完成历史有界且本会话隔离，数字/否定丢失拒绝，未证实ID拒绝，未知指代保留标记；Run `python -m pytest tests/test_query_resolver.py -q`确认RED。
- [ ] 实现上游`with_structured_output(..., method='json_mode', include_raw=True)`并显式JSON指令；不传不支持的strict=True。原始JSON再做重复键/有限长度/extra forbid校验，不信任parsed单独保证。完整句透传是否判对由真实样例验证。
- [ ] 建立冻结数据：八类各20条=160条，另24条未知/边界；12组恰好3轮的物流→退款→物流=36轮，合计220。按整组对话和每类均衡划分calibration/acceptance各110轮，记录原话、完整问题约束、标签和对象。
- [ ] 写评估代码失败测试：验收集不参与阈值选择；其他不算正确自动业务覆盖；无合格阈值禁用该类；指纹变化失效。Run `python -m pytest tests/test_ch06_evaluation.py -q`确认RED。
- [ ] 实现切点选择，候选为真实confidence观测值，条件为核心场景校准误路由为零且有正确自动路由，最大化覆盖，平局选更保守切点；保存统一`eval/ch06/intent-policy.json`及混淆矩阵/覆盖/弃权报告。confidence>=threshold才放行。
- [ ] Run聚焦测试；Run `python scripts/evaluate_ch06.py --suite intent --split calibration --dry-run`核对110条与预算，再`--live --output eval/ch06/intent-calibration-r1.json`。Prompt/criteria最多进行一轮有记录的修订与校准重跑，不触碰验收集。
- [ ] 阈值和配置冻结；独立验收留任务10执行。中文失败如需改变技术选型必须停下来问用户，不能换模型类别或补关键词分类器。
- [ ] 即时记录原始评估、误例和返工，提交并推送`feat(ch06): resolve conversational queries and calibrate intent routing`。

## 任务4：新知识清单、核心场景扩写与政策检索

**Files:** Create `app/query_expansion.py`, `app/policy_retrieval.py`, `tests/test_query_expansion.py`, `tests/test_policy_retrieval.py`, `eval/ch06/corpus.json`, `eval/ch06/policy-cases.json`, `tests/fixtures/ch04_corpus/`；Modify `app/ch04_ingest.py`, `app/build_ch04_knowledge.py`, `app/rag_retrieval.py`, `app/hybrid_store.py`, `app/workflow_knowledge.py`, `app/model_service.py`, `app/main.py`, `scripts/evaluate_ch06.py`, `tests/ch04_support.py`, `tests/test_ch04_ingest.py`, `tests/test_ch04_index.py`, `tests/test_ch04_dataset.py`, `tests/test_rag_index_health.py`, `tests/test_workflow_knowledge.py`。

**Interfaces:** `QueryExpander(model).expand(query: str, item: OrderItem) -> tuple[Expansion, Usage]`（async）；`RagRetrieval.retrieve(query, strategy, filters=None, *, allowed_ids: frozenset[int] | None=None) -> RetrievalResult`；`PolicyRetrieval.prepare(query: str, queries: list[str], item: OrderItem) -> dict`（async，返回evidence/citations/knowledge_confidence/stop_reason）。

- [ ] 写失败测试：扩写唯一queries字段、恰3条、拒绝约束丢失/新增事实，FAQ无扩写；退款/售后两来源均召回、商品文档排除、同chunk只出现一次、合并后统一rerank只调用一次。Run `python -m pytest tests/test_query_expansion.py tests/test_policy_retrieval.py -q`确认RED。
- [ ] 更新当前三份知识文档白名单、manifest、引用文件校验；orders/logistics/reasons绝不入库。历史ch04测试需要的旧文档放测试fixtures并显式指定，不能恢复为当前业务语料或改旧校准产物。
- [ ] 实现strict Expansion+json_mode与语义约束；输出重复扩写作为格式失败，原问题与扩写合并去重后最多4条。错误显式停在检索前，不偷偷用单Query冒充扩写成功。
- [ ] 从权威CorpusSnapshot按source_file导出政策allowed_ids，内部透传到dense/BM25所有AnnSearchRequest的参数化`id in {allowed_ids}`，而非假定Milvus已有source_file字段。allowed_ids空集返回空命中，不能表示不筛选；原有公共KnowledgeFilters保持不变。
- [ ] 多查询每条取hybrid candidates，按chunk_id做RRF(k=60)，稳定同分顺序按ID，最多50个候选，以原完整问题统一rerank取10条；复用完整条款预算与来源digest校验。最多2个检索并发，参数仅集中配置。
- [ ] 写“手册manual仍命中”和“过期语料/异常检索不给资格结论”测试；Run上述测试加`tests/test_rag_retrieval.py tests/test_rag_evidence.py tests/test_ch04_milvus.py`。
- [ ] 以`knowledge_ch06`独立Milvus collection建立当前语料，沿用MySQL版本化chunk，核验不会删除历史被引用chunk；新增CLI manifest参数，Run `python -m app.build_ch04_knowledge --manifest eval/ch06/corpus.json --collection knowledge_ch06`。无重建旧collection。
- [ ] 建48条政策样例（24校准/24验收，含退款、售后、无答案、限制条款与FAQ），先`evaluate_ch06 --suite policy --split calibration --dry-run`再`--live`；最多修订一轮，产物`eval/ch06/knowledge-policy.json`绑定语料/扩写/查询方式/重排指纹。单Query商品咨询也需当前语料校准，不能沿用旧raw-query门限。
- [ ] 记录并提交推送`feat(ch06): retrieve policy evidence with query expansion`。

## 任务5：等待交互与退款动作的持久化边界

**Files:** Create `app/ch06_db.py`, `app/ch06_repository.py`, `app/init_ch06_db.py`, `db/ch06_schema.sql`, `tests/test_ch06_db.py`, `tests/test_ch06_repository.py`；Modify `app/workflow_repository.py`, `app/chat_views.py`。

**Interfaces:** `InteractionRepository.publish_wait(conversation_id, turn_id, interrupt_id, kind, payload, checkpoint_id) -> dict`；`claim_resume(conversation_id, turn_id, interrupt_id, request_id, payload_digest) -> dict`；`finish_resume(request_id, result: dict) -> None`；`cancel_wait(conversation_id, interrupt_id) -> None`；`pending(conversation_id) -> dict | None`；`RefundRepository.issue_action(conversation_id, turn_id, item, eligibility, fingerprints) -> dict`；`RefundRepository.create_request(conversation_id, action_id, reason_code, request_id, item, expected_fingerprints) -> dict`（任务8实现提交事务）；`initialize_ch06_database(database_url: str) -> None`。

- [ ] 写真实临时MySQL失败测试：四张新增表`workflow_interactions`、`workflow_resume_requests`、`refund_actions`、`refund_requests`，完整键/外键/状态/十进制金额，初始化重复无变化，部分/不兼容表显式报错；Run `python -m pytest tests/test_ch06_db.py tests/test_ch06_repository.py -q`确认RED。
- [ ] 定义交互以interrupt_id主键并关联conversation/turn；恢复请求request_id唯一且绑定payload_digest；退款动作action_id唯一、来源版本和资格上下文持久化。退款请求request_id及action_id唯一，退款单号独立。
- [ ] 实现短事务与同行锁；重复键异常后退出失败事务再查结果，不在已回滚session继续flush。未结束恢复保持processing，不因网络断线重新申请同一动作。
- [ ] 增加暂停展示消息与稳定幂等位置，历史展示等待但不把失败半轮当作已完成回答；保留原消息身份兼容。取消标记使旧卡片和未提交表单动作失效。
- [ ] 故障注入断言无半条动作、无越会话访问、同request_id不同payload返回冲突；Run上述测试加`tests/test_workflow_repository.py tests/test_chat_views.py`。
- [ ] 记录并提交推送`feat(ch06): persist recoverable interactions and refund actions`。

## 任务6：确定性退款售后子流程和真实interrupt

**Files:** Create `app/aftersales_workflow.py`, `app/workflow_interactions.py`, `tests/test_aftersales_workflow.py`, `tests/test_ch06_interrupts.py`；Modify `app/workflow_graph.py`, `app/workflow_types.py`, `app/workflow_log.py`, `app/agent_runtime.py`, `app/prompts.py`, `app/model_service.py`, `tests/test_workflow_graph.py`, `tests/test_workflow_checkpoint.py`。

**Interfaces:** `build_aftersales_subgraph(*, business, expander, policy_retrieval, model, limits, emit)`返回CompiledStateGraph，继承父checkpointer；`resolve_execution_input(text: str, pending: dict) -> dict`（async，受限slot_reply/new_request/unclear判定及仅显式ID提取）；`judge_eligibility(query: str, item: OrderItem, evidence: list, facts: dict) -> tuple[Eligibility, Usage]`（async，主力模型，不做意图分类）。

- [ ] 写图路径失败测试：resolve→Jev→policy→route顺序；核心场景先客户/选单→order→expand→retrieve→gate→agent；低置信/错误不读订单；FAQ无扩写；未解指代不猜对象。Run `python -m pytest tests/test_aftersales_workflow.py tests/test_ch06_interrupts.py -q`确认RED。
- [ ] reset清理轮次动作/证据/意图，同时只保留同会话已确认相关对象；插入独立resolve/classify/policy节点，所有失败走统一reason。用新state.intent_decision，保留内部reason但不污染双字段JSON。
- [ ] 缺客户ID由主力Agent受限问句节点生成询问，随后独立wait_customer节点interrupt；补齐后缺订单/商品项进入wait_order节点interrupt。等待节点只做可重放读取，不含付费模型、写表或检索调用。
- [ ] 校验回填归属及来源digest；读取订单后扩写、检索、证据闸，最后Agent判断。信息不足留在Agent澄清，不伪造条件，不要求退款原因枚举；明确售后需求仍按维修/换货等目标答复。
- [ ] 统一计算实际付费usage；原模型调用上限6及总token上限继续约束聊天模型，包括改写、扩写和澄清，Jev请求独立计数但token计入轮次总量。恢复不重置计数，无预算时显式停止，不能无限interrupt循环花费。
- [ ] 用真实SQLite文件测试关闭重开后同thread_id恢复、另一thread隔离；断言resolver/classifier调用次数仍为1，GraphInterrupt不计node_error。新诉求结束旧等待并重新走Jev；错误ID保留有效等待。
- [ ] Run上述测试加既有graph/checkpoint/agent_runtime测试；记录并提交推送`feat(ch06): add deterministic aftersales workflow with interrupts`。

## 任务7：恢复接口、SSE与双存储对账

**Files:** Create `tests/test_ch06_interaction_api.py`, `tests/test_ch06_recovery.py`；Modify `app/workflow_service.py`, `app/main.py`, `app/schemas.py`, `app/chat_views.py`, `app/workflow_repository.py`, `tests/test_workflow_service.py`, `tests/test_workflow_api.py`, `tests/test_conversation_management.py`。

**Interfaces:** `WorkflowService.resume_turn(conversation_id, turn_id, interrupt_id, request_id, value) -> AsyncIterator[ChatEvent]`；`cancel_interaction(conversation_id, interrupt_id)`（async）；`get_pending_interaction(conversation_id) -> dict | None`（async）。API路径、请求字段遵循spec第10节，ID字符串不作前端数值转换。

- [ ] 写API失败测试：interaction_required→paused结束且无done/error；合法resume继续，错误/过期/跨会话返回409/404，重复提交不重放；Run `python -m pytest tests/test_ch06_interaction_api.py tests/test_ch06_recovery.py -q`确认RED。
- [ ] 在现有astream updates分支显式处理__interrupt__，不把tuple当节点dict；传入Command恢复，核查安装版interrupt_id属性与StateSnapshot格式。请求和取消都沿用会话单写锁。
- [ ] SQLite状态写成功后发布waiting展示信息，再发interaction_required/paused；只有最终消息与checkpoint均成功才发done。已暂停断线不标失败；执行中取消仍按既有安全取消路径处理。
- [ ] GET历史合并SQLite权威pending与MySQL展示，不一致按turn/interrupt/checkpoint标识修复；不把MySQL pending单独当可恢复证据。SQLite恢复推进但MySQL结果落后时从checkpoint补齐，不重跑付费节点。
- [ ] 新普通聊天在等待时走执行输入判别；纯ID补槽，改变诉求则取消旧等待再建立新轮。旧卡片禁止恢复，客户变化清理旧绑定。删除会话禁止恢复和动作提交。
- [ ] 故障注入两个存储先后失败、服务重启、重复点选、resume处理中断线；断言模型/退款次数与预期一致。Run上述测试及service/api/conversation_management模块；记录并提交推送`feat(ch06): stream and restore human-in-the-loop interactions`。

## 任务8：退款申请表单后端与幂等提交

**Files:** Create `app/refund_service.py`, `tests/test_refund_service.py`, `tests/test_refund_api.py`；Modify `app/ch06_repository.py`, `app/main.py`, `app/schemas.py`, `app/chat_views.py`, `app/workflow_service.py`。

**Interfaces:** `RefundService.prepare_action(conversation_id, turn_id, item, eligibility, evidence) -> dict`；`RefundService.submit(conversation_id: int, action_id: str, reason_code: str, request_id: str) -> dict`（async）；结果包含refund_no/status，status为待处理。客户端不传可决定金额/资格的值。

- [ ] 写失败测试：eligible才给表单、原因不在枚举拒绝、已退款/活跃申请拒绝、重复请求同号、异payload同键冲突、篡改金额额外字段拒绝；Run `python -m pytest tests/test_refund_service.py tests/test_refund_api.py -q`确认RED。
- [ ] 表单动作绑定当前商品项全量数量和实付Decimal金额、源文件digest与证据版本；提交重新读订单及动作。源版本变化要求重新判断，不执行旧动作。
- [ ] 原因改变资格前提时用主力Agent已加载证据复核，必要时重新检索；资格不明返回澄清/待重新评估，不建单。主力Agent只判断资格，最终创建由代码控制。
- [ ] MySQL事务中锁定同一客户/商品项的退款操作：按规范化元组导出的有界MySQL advisory lock串行化跨会话申请，再检查活跃申请并插入；advisory lock的SQL接口在实现前通过Context7核对。request_id/action_id唯一约束保证重放返回同号，锁释放必须在finally。
- [ ] 无资金划转调用，不改orders.md；成功文案为“退款申请已提交，等待处理”。重复、取消和数据错误不返回“退款成功”。
- [ ] Run上述测试加ch06_repository；记录并提交推送`feat(ch06): submit idempotent refund requests after confirmation`。

## 任务9：聊天订单卡片与退款表单（前端例外）

**Files:** Modify `app/web/index.html`；浏览器证据保存`eval/ch06/browser-acceptance.md`与允许提交的无敏感截图。

**Interfaces:** 消费任务7的interaction_required/paused和pending_interaction；使用resume接口回填ID；退款表单消费服务端动作/原因列表并调用任务8接口。

- [ ] 直接实现聊天流内订单商品卡片：商品名/规格、订单号、状态、金额、时间，选中后disable防连点；客户ID提示与卡片属于当前轮，不覆盖历史消息。
- [ ] 实现paused状态、页面刷新恢复、取消本次操作、切换话题、失败可重选提示；金额/ID按文本展示，DOM安全插入，绝不把Jev密钥或API调用放前端。
- [ ] 直接实现简单退款表单：只读商品项信息、固定原因下拉、提交按钮、申请号/待处理回执；不追问退款原因，不硬编码另一套原因枚举。
- [ ] 用真实浏览器走未带订单号→客户ID→卡片→点击恢复→政策答复→表单；再检查刷新/重复点击/取消/切话题。此任务不做brainstorm、TDD或独立前端code review；所依赖的后端安全/契约测试仍必须通过。
- [ ] 按用户反馈直接调整效果，逐次记录；完成后提交推送`feat(ch06): add chat order cards and refund form`。

## 任务10：独立验收、全分支后端评审与交付

**Files:** Create `scripts/demo_ch06.py`, `tests/test_demo_ch06.py`, `eval/ch06/acceptance-report.md`；Modify `scripts/evaluate_ch06.py`, `README.md`, `dev-notes/ch06.md`。

**Interfaces:** `demo_ch06.py --case context-switch|order-picker|refund-form --live`；`evaluate_ch06.py --suite intent|policy|workflow --split calibration|acceptance --dry-run|--live --output PATH`。CLI缺配置返回非零并明确原因；禁止模拟运行冒充真实验收。

- [ ] 按verification-before-completion核对数据库初始化与启动指引：明确`python -m app.init_ch05_db`、`python -m app.init_ch06_db`前置条件，MySQL变更前创建ignored备份；单worker启动示例`python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1`。不遗漏初始化再让用户排查503。
- [ ] 为CLI显式live、缺配置无假输出写失败测试并实现；Run `python -m pytest tests/test_demo_ch06.py -q`。
- [ ] 冻结criteria、模型版本、两个校准产物后Run意图110轮独立验收和政策24条独立验收，另跑12条有序端到端场景（每条最多3轮）；核心用户验收必须逐轮正确。只读演示不自动提交退款，表单写入仅在明确演示操作中提交。
- [ ] 验收失败记录真实结果，停止宣称完成；修复确定性bug后只重跑受影响检查，若暴露语义策略问题则重新冻结新验收样例并告知新增真实请求需求，不反复调同一验收集。
- [ ] Run `python -m pytest -q`一次全量回归（使用既有可清理临时数据库/Milvus资源），`python -m pip check`、`git diff --check`；长测保留原进程并报告进度，不重复启动。记录通过/失败/跳过/警告与实际耗时。
- [ ] 调用requesting-code-review，对全分支后端与验收覆盖做独立审查（前端按用户要求豁免）；先修Critical/Important再复核。即使Native执行，最终也保留一次独立后端评审。
- [ ] 按finishing-a-development-branch完成集成：提交并推送`codex/ch06-intent-router`，创建可审阅PR并attach；未获合并授权不宣称已合入main。交付演示命令、测试/真实评估报告、dev-notes路径和提交/PR。

## 真实请求预算与执行顺序

本计划批准后才运行以下付费评估；已完成模型列表查询不属于decision调用。

| 阶段 | Jev请求上限 | 聊天上游请求上限 | 说明 |
|---|---:|---:|---|
| 协议验证 | 1 | 0 | 只确认契约和实际模型版本 |
| 查询理解/意图校准与独立验收 | 330 | 330 | 110首次校准+最多110修订校准+110独立验收 |
| 政策扩写校准与验收 | 0 | 72 | 24首次校准+最多24修订+24验收；含不扩FAQ时实际更少 |
| 端到端/浏览器 | 36 | 216 | 12场景×最多3轮；每轮聊天模型最多6次，含提交原因复核 |
| **总上限** | **367** | **618** | 所有失败/重试也计数，不是预算阈值或保证会跑满 |

无网络自动重试的评估预算以实际provider尝试为准。每个请求前持久化占额记录，返回后补结果/usage，断线也不退额度；中断后只续跑未尝试样例，避免重复付费。端到端因早期步骤失败而不能继续时不得拿虚构上下文补齐。

`--dry-run`输出用例数、最大请求数、当前模型名、配置缺口和token预算；不打印密钥/完整.env。报告按实际usage统计token；未核实聊天上游价格时不虚报精确费用。超出以上预算、需要更换固定模型或政策数据存在矛盾时先说明并询问用户。

任务按1→2→3→4→5→6→7→8→9→10执行，因接口和持久化耦合推荐Native：当前会话逐项实现，最后一次独立后端评审。子代理逐任务执行也可，但必须先由用户选择；本计划阶段不派发实现子代理。

## 计划自检结论

覆盖映射：spec 1–3→任务1/4；4→任务3；5→任务2/3；6→任务6；7→任务4；8→任务5/6/7；9→任务8/9；10→任务2/5/7/8；11→任务3/4/10；12→每项文档核查与任务10。

接口检查：公开IntentDecision只有两字段；内部IntentOutcome容纳reason/usage；Selection以客户+订单+商品项三元组校验；图confidence字段不覆盖；API与UI共享interrupt/action身份。Review Focus五项均已映射到失败测试。

自检修正：原Milvus schema没有source_file，改用权威chunk ID集合过滤；json_mode不搭配strict=True；重放节点不含付费调用；单商品项跨会话重复申请在数据库边界串行化；当前知识文件改名与旧fixture兼容被纳入任务4；模型列表与真实分类评估分开记。
