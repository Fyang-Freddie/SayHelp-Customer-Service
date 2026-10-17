# ch05 移除模拟数据补充验收（2026-10-10）

## 变更范围

用户批准：删除假数据，未接入的来源明确说明；后续业务数据存放在 knowledge_db，当前不猜测格式或实现连接器。

- 删除演示的 SimulationModel、SimulationRetrieval、固定证据和随机种子，移除 --live 模式切换；两个演示模式均使用配置模型和现有真实知识检索。
- 原 query_order/query_logistics 和未绑定知识适配器的 query_product 返回 available=false / data_source_not_connected，不生成状态、金额、位置、价格或库存。已绑定商品/FAQ 知识适配器保留真实检索。
- 裸循环接入现有真实知识闸。固定路由、分类后闲聊/投诉固定回复、SQLite、独立工单按钮和前端转人工模拟保持原要求。
- 测试隔离替身仅用于测试；历史 acceptance_results.json/report.md 是旧基线，没有伪装成此次运行结果。

## 验证

- TDD 同一聚焦命令：RED 10 failed / 42 passed / 3 skipped；GREEN 52 passed / 3 skipped。跳过原因是该进程未加载测试数据库配置。
- 扩展离线回归：141 passed / 3 skipped；无付费调用。
- 独立最终代码评审：approved，无 Critical/Important 问题。
- pip check：No broken requirements found。
- 警告复核：同环境加载TEST_DATABASE_URL，运行 `pytest tests/test_chat_api.py -q --tb=short -W error::pytest.PytestUnraisableExceptionWarning`，**30 passed / 0 warnings / exit0，64.64s**。见no-simulation-cancellation-recheck.txt。本次未复现，未修改取消逻辑，也不宣称根因已修复。
- 全量回归：**681 passed / 0 skipped / 3 warnings，exit 0**；原始记录见 no-simulation-full-suite.txt。真实 MySQL 临时库、独立 Milvus 测试库、缓存 BGE 模型离线验证，临时 Milvus 库清理已确认。计时22258.66s（原进程跨越会话额度中断，未重启）。三条 PytestUnraisableExceptionWarning 来自取消测试中的 LangGraph 协程回收和 ContextVar 清理；这些关联代码本次无修改，不能将本次结果写成零警告。

真实验证预先标注于 no-simulation-live-case.json，结果见 no-simulation-live-result.json。命令：

```powershell
.venv/Scripts/python.exe scripts/demo_ch05.py --mode workflow --case multi-step --output eval/ch05/no-simulation-live-result.json
```

此命令已运行一次，不需为查看结果重复运行。真实模型 deepseek-v4-flash：1 分类、2 选择、1 最终流式；仅 query_order 一次，说明“订单数据源尚未接入”后停止，没有物流查询、虚构业务状态或工单写入。输入3621/输出598token，估算0，65流块。此前9轮加本轮已达原授权10/10上限，0新增重试。

限制：回复仍建议“提供运单号让我尝试查询物流”，而物流来源也未接入。独立复核判为 Minor 措辞限制；不影响该轮停止和无假数据，但不能据此宣称答复质量完美。本次记录该限制，不额外调优或重复付费验证；可能的代价是用户多发一次不可执行的查询。

## 功能演示

在工作树根目录、配置真实模型和已有知识库环境后运行（会调用真实模型）：

```powershell
.venv/Scripts/python.exe scripts/demo_ch05.py --mode workflow --case logistics
.venv/Scripts/python.exe scripts/demo_ch05.py --mode workflow --case policy
.venv/Scripts/python.exe -m app.bare_agent --message "查询订单1001的状态"
```

开发过程按阶段追记于 ../../dev-notes/ch05.md。
