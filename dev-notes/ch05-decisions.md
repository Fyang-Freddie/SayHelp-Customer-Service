# 第 5 章控制端裁决归档

按 SDD ledger 出现顺序完整归档 26 条 `Ruling:`，附判断有误时的代价。原始裁决未改写；完成日期 2026-10-10。

1. **工作树与推送沿用已批准授权**。代价：若范围判断错误，会产生非预期分支或推送；本次以用户计划和 AGENTS 为依据。

   > Ruling: Worktree creation and branch pushes are authorized by the approved plan and AGENTS; no repeated permission prompts.

2. **复用既有依赖基线，新增依赖不顺带升级**。代价：环境继承可能影响新机器复现；后续用隔离环境和锁文件补足。

   > Ruling: Reuse main virtualenv for existing baseline; when adding LangGraph do not upgrade existing packages; record pip dry-run and install constraints.

3. **任务内聚焦测试，最终做全量回归**。代价：跨模块问题可能到整体验证才暴露。

   > Ruling: Full regression is required at final integration; per-task focused checks per approved plan, with broad suite run before each first implementation commit when feasible; no repeated unchanged broad runs after fixes.

4. **通过私有配置读取认证 Context7 MCP**。代价：辅助脚本或配置不兼容会中断文档核查；不输出认证头。

   > Ruling: All technical queries remain authenticated Context7 MCP. Helper reads user config without printing headers.

5. **所有代理只修改隔离工作树**。代价：原工作区未提交上下文不会自动进入本章分支。

   > Ruling: Shared worker agents must use this absolute worktree; never edit the original dirty checkout.

6. **超大工具结果以配对错误替代，不截断证据**。代价：该结果无法直接用于回答，可能需要缩小问题。

   > - Ruling: preserve call pairing with bounded error for oversized result, never silently clip evidence; add failing regression and focused validation. Same implementer resumed; no new full run unless broader concern emerges.

7. **执行工具前验证整批最小配对与最终输出预留**。代价：保守预算可能提前停止本来可回答的轮次。

   > - Ruling: validate complete minimal pairing allowance before admitting a batch or performing reads; if impossible retain prior paired context and terminate safely. Same implementer resumed, boundary RED regression required. This is a reserve proof correction, not raising budgets or truncating evidence.

8. **分类模块用聚焦测试和标注评估，不重复无变化全测**。代价：集成回归可能延后发现。

   > Task 2 implementer /root/implement_2; base15362ea. Ruling: independent classifier/parser changes use plan focused checks and real evaluation, broad task1 already passed and task9 final integration broad remains; avoid unchanged costly full repetitions without impact justification.

9. **可分离校准集采用间隔中点，另用预标注新确认集**。代价：小样本门槛仍可能不能泛化；旧失败和受影响的留出集地位如实保留。

   > - Ruling: new raw-query artifact may use calibration-only midpoint(highestnegative,lowestpositive) for a separated calibration set; ch04 algorithm/files unchanged. Retain original outputs and mark oldheldout as development feedback once method changes after observing it. Freeze method, prelabel a fresh12 confirmatory set before scoring, report all runs. Overlap must fail rather than invent a separating threshold. This guards against adapting the acceptance set while claiming untouched holdout evidence.

10. **工单保持原会话状态，修正旧测试和动作字段映射**。代价：依赖旧人工转接副作用的调用者需要适配。

   > Task4 Ruling: update tests/test_db.py original create_ticket status expectation to unchanged进行中 as required by removal of handoff coupling; task brief file list omission corrected, old business tools remain. Cost if wrong: legacy callers must accept intentional user-requested independent handoff behavior. Repo actions use action_id, SSE service maps to id; conflicts ValueError, missing/deleted KeyError.

11. **隔离虚拟环境只读继承原包，新依赖本地安装**。代价：包查找顺序与全新安装可能不同；锁文件及依赖检查用于约束。

   > Task5 Ruling: isolated worktree .venv may inherit original environment packages read-only via .pth/include-system-site-packages; install only constrained LangGraph/SQLite additions locally. Preserves original env and avoids stack reinstallation; cost if wrong: dependency lookup differs from fresh full install, mitigated exact lock/signature/pipcheck and documented clean-install reproduction. Original env remains unchanged.

12. **补充计划遗漏的 State reducer 与安全日志接口**。代价：共享接口改动面扩大，需要覆盖相关调用者。

   > Task5 Ruling: narrow extensions to workflow_types.py add_messages reducer and workflow_log.py safe turn_finished event required for graph interfaces though brief file list omitted them. Existing knowledge logging semantics preserved; no raw customer or reasoning in generic events. Cost if wrong: broader logging interface affected, covered focused log tests. Resolved compatible LangGraph1.2.2/sqlite3.1.1, originalhttpx0.28.1 preserved; realSQLiteclose/reopenprobe passed.

13. **用完整且有界的 MySQL 轮次恢复模型上下文**。代价：图状态只保留提示窗口，完整审计仍须查 MySQL。

   > Task6 Ruling: reconcile checkpoints by replacing messages with bounded COMPLETE MySQL history via existing prepare_context and stable DB message IDs/currentID; preserve existing wholeturn dropping/maxturn/contextbudget/reserve, no contextstrategyupgrade. Full audit remainsMySQL; do not rely unboundedhistory plus Agent cumulativebudget. Cost if wrong: graphstate has promptwindow rather than wholeaudit, intentional priorbehavior retained and history API remainscomplete. Finalreply persists on top-level update, actions/completed only after astreamnormalexhaustion.

14. **现有知识工具接入结构化检索闸适配器**。代价：旧的生成答案格式变为证据格式，消费者需要正确适配。

   > Task6 Ruling: reuse existing build_tools knowledge_answer wrappers with thin async KnowledgeGate.prepare structured evidence adapter, no newbusiness tool or legacy generation/selfcheck. Business outerroute remains no pre-retrieval/gate; optionalAgenttool is actual read. Cost if wrong: wrapper payload differs from legacy generatedanswer, mitigated structuredpayloadtest and Agentfinalgeneration.

15. **排除未完成轮次并在锁后刷新状态**。代价：已取消的审计轮次不会进入后续模型上下文。

   > Task6 Ruling: add WorkflowRepository.incomplete_message_ids to exclude running/failed/incomplete cancelledturns before existing contextwindow; preserve auditrows and recoverable checkpoint_failed fullypersisted finalhistory. Address deferredTask4Minor lockthenrefreshturn because statuses now loadbearing. Cost if wrong: recovered modelcontext may omit intentionallycancelled auditturn, required to avoid using unfinishedconversation; faulttests distinguish completedMySQL recovery.

16. **验证实际子检查点先完成的故障顺序**。代价：未来若重排父检查点与 MySQL 顺序，需要新增测试。

   > Task6 Ruling: reverse-order fault proves actual childSQLite completedcheckpoint before MySQLfinalfailure; parentterminalSQLite necessarily follows MySQL in ordered top-level-update design. Do not introduce concurrentgraphdrain just to force impossible rootterminal-first ordering. Must show childcheckpoint real, noactions/completed afterSQLfail, clean freshnextturn/noresume. Cost if wrong: test coverage of hypothetical alternative rootterminal-first orchestration absent; current architecture cannot generate it, actual partial-state risk covered and limitation explicitly reported.

17. **Milvus 恢复测试使用严格归属的随机独立库**。代价：误指目标可能伤及知识数据；用唯一名称、非默认库和清理归属校验防护。

   > Task9 Ruling: test-only legacyMilvus recovery may require TEST_MILVUS_DATABASE freshunique nondefault test-prefix and MilvusClient(db_name=...), officialContext7verified. Ignoredrunner createsownedrandomdatabase and drops only verifiedownedexactname afterfullsuite; preserveemptyknowledgeguard, inspectallTEST_MILVUS_URIuses, productionstoreunchanged. Cost if wrong: mistakenDBtarget could damageknowledge; mitigatednondefaultprefix/uniqueownedname and guardedcleanup, no productiondefaultcollectiondrop.

18. **CLI 投诉只展示选项，真实确认建单留给网页/API**。代价：CLI 本身不演示数据库工单写入。

   > Task9 Ruling: CLI complaintdemo only prints suggestions, no ticketcreation path; conditional explicitconfirmationrule applies ifoffered. Webbutton remains actual confirmedcreation demo. Originalreadonlymocktools+temporaryofficialSQLitecheckpoint suffice CLI independentfixture; no newSQLitebusinessdatabase. Cost if wrong: CLI doesnotdemonstrate DBticketwrite, coveredrealMySQL APItests+webflow instead and documented.

19. **恢复已安装 Docker Desktop 与既有测试依赖**。代价：其他设置为自动启动的既有容器可能同时上线。

   > Task9 Ruling: start existing local Docker Desktop hidden after Context7 official docs because process/pipe absent and disposableDB tests require it; reversible dependency recovery, no config/container recreation/deployment. Cost if wrong: existing auto-restart containers may come online; inspect only known stack and do not mutate data. Controller owns startup/availability check; worker avoids duplicate actions.

20. **保留单进程 SQLite 边界，不扩展高可用**。代价：超过单实例部署范围需要后续架构工作。

   > Final Ruling: retain single-worker SQLite scope and exclude distributed coordination/high availability — explicitly approved chapter boundary — cost if wrong deployment beyond one worker is unsupported and requires later architecture work.

21. **再次确认现有跨库写入顺序及其测试边界**。代价：未来改变编排顺序时不能照搬现有故障覆盖结论。

   > Final Ruling: retain actual child-checkpoint-first failure coverage and ordered parent-final-after-MySQL design, per prior Task6 ruling — do not invent impossible parent-terminal-first ordering — cost if wrong a future reordered implementation needs new cross-store tests.

22. **如实保留一次 Context7 查询晚于首改的历史**。代价：历史流程偏差无法靠后续代码追溯消除。

   > Final Ruling: preserve truthful Task6 late Context7 lookup history rather than claim retroactive compliance — API subsequently verified, historical deviation cannot be repaired by code — cost if wrong process trust requires revisiting the recorded evidence.

23. **模拟不能替代有授权的真实模型验收**。代价：授权前能力未知且交付延迟；现已取得授权并完成 9 轮验证。

   > Final Ruling: real-provider quality and streaming remain unverified until explicit external destination/payload authorization and actual capped evaluation — simulation cannot replace live evidence — cost if wrong release remains incomplete and provider incompatibility can surface later.

24. **代理警告只修正模拟测试客户端配置**。代价：若另有真实代理网络问题，仍需单独诊断。

   > Final Ruling: treat observed proxy warning as mocked-test output hygiene and explicitly configure its test transport, without changing production proxy configuration — no production failure evidence — cost if wrong a separate real proxy problem remains undiagnosed.

25. **人工转接问候按前端本地模拟处理**。代价：刷新后该视觉状态不会作为服务端历史恢复。

   > Final Ruling: local handoff greeting need not persist as durable server history — approved frontend simulation only — cost if wrong handoff visual state disappears on reload and would need a later persistence requirement.

26. **预算内增加一个预标注商品正例**。代价：消耗一轮真实调用，且只提供有限的商品措辞验证。

   > Task9 Ruling: add one prelabelled positive MH-W60 product case after the original eight live turns pass, staying within the human-authorized ten-turn cap; source product-specs.md49-50 independently confirms6L/304stainless. This validates the changed grounded-product prompt, not calibration retuning; preserve original seven labels and failures. Cost if wrong: consumes one remaining live attempt and gives only narrow product-wording evidence, not broad generalization.

