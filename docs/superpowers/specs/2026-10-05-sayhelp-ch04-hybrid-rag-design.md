# ch04 · 混合检索、证据控制与评估设计

日期：2026-10-05。状态：设计草案，待用户审阅；尚未进入实现计划或产品实现。

## 1. 用户目标与固定边界

把现有 SayHelp 客服的检索质量和可追溯性提升到可测量、可复盘的水平。验收包括四策略数字报告、具体型号 BM25 命中、点击引用查看原文，以及知识缺失时明确拒答并记录原话。

固定使用 Milvus 原生 BM25、内置 chinese analyzer、hybrid_search + RRF、BAAI/bge-reranker-v2-m3。保留 BAAI/bge-m3 dense 嵌入。只对当前单条问题做归一化及检索侧同义词扩展；不做指代消解、多轮改写。不会为了改善数字复制同义问法入库。

知识来源为 knowledge_db 的 after-sales-manual.md、product-faq.md、product-specs.md、returns-policy.md。新建库和正式评估均不使用随机生成知识或第 2/3 章测试夹具。旧测试数据只用于自动测试。现有订单/物流演示工具继续明确标注模拟结果；商品规格问题必须走真实知识检索，不能被旧随机商品工具回答。

用户指定：后端代码走 Superpowers/TDD/评审；纯 Prompt、标注数据用真实标注样例验证代替 TDD；前端按 Vibe Coding 直接实现、浏览器验收，不套 brainstorm/TDD/code review。阶段完成时即时追记 dev-notes/ch04.md；最终验证、提交并推送指定 GitHub 仓库。

## 2. 现有代码与环境

- 主分支 main，起点 4544b0f；knowledge_db/ 和 test.py 是用户已有未跟踪文件。
- Milvus 2.6.24、MySQL 8.4 容器健康，PyMilvus 2.6.17。符合用户要求的 Milvus 2.5 起原生全文检索。
- FastAPI 0.142.2、SQLAlchemy 2.0.54、langchain-core 1.6.6、langchain-openai 1.6.7、sentence-transformers 6.1.0。
- app/vector_store.py 当前 knowledge 集合只有 id/vector；app/knowledge_search.py 返回 question/answer/category，按 dense cosine 0.55 判匹配。
- MySQL knowledge_chunks 已保存章节路径和相邻指针，但未保存来源文件及行号。chunk_markdown 已计算 source_line，可扩展为精确起止范围。
- app/chat_service.py 由模型选一次工具，然后直接流式回答；只改 query_faq 内部不能保证模型选错工具或未调用工具时仍执行知识拒答。
- app/main.py 使用 FastAPI 原生 SSE。app/web/index.html 当前纯文本显示，无引用/反馈交互。

## 3. 接入方案比较与推荐

推荐：新增独立知识回答管线，由单条 query 理解结果确定知识/操作/闲聊路由；知识路由强制检索、自评与引用校验，操作路由复用现有订单/物流/工单流程。query_faq 复用同一检索服务，商品规格不再调用随机 query_product。这样知识拒答与引用不依赖模型自觉选择工具。

备选一：仅改 query_faq 和 System Prompt。改动小，但工具未调用时可绕过证据控制，不满足强制拒答验收。

备选二：所有消息都强制 RAG。证据门控简单，但订单操作、问候会被无关知识拦截，破坏现有功能。

三者均不改变指定检索技术；推荐方案只增加必要边界，不引入 LangGraph/Langfuse 或新的外部服务。

## 4. 数据与索引迁移

原样保存用户给出的 low_confidence_questions / faith_cases DDL 到 db/ch04_schema.sql，包括 SET NAMES utf8mb4、外键、ENUM、唯一键、注释、全部处置与复发字段。ORM 与真实 MySQL information_schema 对照验证。初始化必须幂等，发现半成品/不兼容 schema 明确报错。

另外增加显式 ch04 知识溯源迁移：knowledge_chunks 的 source_file（仓库相对路径）、source_start_line、source_end_line、product_category（标准品类）、source_digest（原文版本指纹）。旧行字段可空，不伪造来源。source_file 不接受任意客户端磁盘路径。正文、章节、chunk ID 的权威源仍是 MySQL。

构建命令默认从 knowledge_db 的显式文件清单导入；product-faq.md 也按 Markdown 原文分块，不追加数据库演示 FAQ。原文与元数据一份存储，同义词只在查询侧扩展。由标题确定品类：猫砂盆、饮水机、猫爬架、加热垫、喂食器、摄像头；政策/售后另设明确类型。不直接把已有 category 的整条标题路径当作商品品类。

新增 knowledge_ch04 集合，与旧 dense knowledge 集合并存；不删除旧集合、不通过不兼容 schema 静默重建。字段：id、vector（1024 dense）、text（VARCHAR，enable_analyzer=True，analyzer_params={type: chinese}）、sparse（SPARSE_FLOAT_VECTOR）、category、product_category、content_type、source_digest。text 使用权威标题/问题/正文；BM25 函数由 text 自动生成 sparse，客户端不写 sparse 向量。dense COSINE 索引 + sparse SPARSE_INVERTED_INDEX/BM25 索引。

切换前全量构建新集合，包括旧库已 done 的真实文档行；不能复用旧 done 标志就跳过新索引。独立保存构建进度与完成清单，失败可恢复；切换前校验主键、文本/元数据指纹、索引参数与真实文件集合。测试/演示来源不得进入新集合正式评估。正文更新必须更新索引并校验指纹，避免陈旧 Milvus 过滤结果被直接使用。

## 5. Query 理解、过滤与检索

单次归一化输出：原话、标准问题、检索侧同义词、意图。型号、数值、否定条件不能被改写丢掉，型号保留原字符串，大小写/空格规范化只作为额外查询表示。模糊到无法确定商品型号时请求澄清，不能靠历史做多轮改写。

metadata filter 通过 HTTP/工具的可选结构化参数提供，采用字段白名单、类型验证及参数化表达式。每条 AnnSearchRequest 都使用同一过滤表达式，在 Milvus 内过滤后召回。不先召回再用 MySQL 过滤冒充前置过滤。

四个策略共用相同 query 归一化、过滤、源库和生成器：dense Top-50、bm25 Top-50、hybrid（两路各 Top-50，Milvus hybrid_search + RRFRanker(k=60)，融合取 Top-50）、hybrid_rerank（融合 Top-50 交给指定 reranker，取 Top-10）。纯 BM25 不做 dense 模型推理；重排失败明确报错或安全拒答，不能静默降级称已 rerank。

检索返回 chunk_id、权威正文、章节、来源、各阶段 rank/score。BM25、RRF、dense、reranker 分数各自记录，不能拿 dense 的 0.55 阈值判 RRF。重排分数不是已校准概率。可配置置信门槛由独立标注校准集验证，再在冻结测试集报告误答/拒答效果。

检索排序与 prompt 排序分离。最终 Top-10 按相关性名次从外向内布置：名次 1 放首位、2 放末位、3 放第二位、4 放倒数第二位，以此类推。引用编号在最终喂入证据确定后才分配，之后禁止重排改号。上下文预算不足时按相关性减少完整 chunk、再做首尾排列；不截断关键条件、不在生成后伪造 Top-10 全部已喂入的快照。

## 6. 生成、自评与拒答

知识路由只把本轮问题与本轮证据作为事实依据。历史仍可服务会话展示与操作上下文，不能充当知识事实或用于本章多轮改写。

生成返回严格结构 useful、reason、answer。检索为空/低置信时最终 useful=false；召回有证据也必须由生成模型判断能否覆盖问题的全部必要条件。useful=false 时忽略草稿答案，输出固定明确拒答和可行下一步；useful=true 的答案必须有合法引用，每个编号对应本轮已喂入的 chunk。无效结构/越界引用安全拒答；依赖故障另报服务暂不可用，不伪装成真实知识缺口。

依照用户 DDL 注释，本章只在最终 useful=false 时写 low_confidence_questions，不新增 useful 列，不因👍/👎点击入池。检索不足使用 retrieval_low_conf；生成自评证据不足使用 self_check；保存用户原话（非改写文本）、conversation_id、具体原因、时间。一次正常请求只入池一次。取消/落库失败不得先发送 done 宣称记录成功；与现有消息写入一样保证事务完成与取消边界。

知识回答先完成结构生成及引用校验，再通过现有 SSE token 事件发送已确认文本（保留前端逐字显示），此前可发检索/生成状态；避免先泄露未经检查答案后再拒答。增加 citations 事件携带本轮完整证据快照及 source URL。模型自评不能保证绝无幻觉，离线 Faithfulness 裁判独立检验。

System Prompt 明列：不承诺退款到账时间；处理时限不等于到账；不承诺审核必过、免费退换/维修、物流必达、库存/价格事实及任何未执行操作；保留例外条件（定制/生鲜/拆封消耗品、质量原因、人为损坏、偏远地区等）。MH-LP50 无 App 等型号差异不能套用通用智能设备联网说明。

## 7. 引用与前端交互契约

citations 为最终喂入证据全集，格式至少 n、chunk_id（字符串）、section_path、question、answer、source_file、source_start_line、source_end_line、source_digest、source_url。答案 [n] 对应这一份列表。只要页面显示当前回答，点击编号就查看这一轮快照，避免后续入库覆盖当时证据。

点击 [n] 打开来源面板，展示章节路径、chunk 原文和原文跳转链接；源文档接口只能访问 knowledge_db 白名单 Markdown，展示完整原文并定位到行范围，文档更新时提示版本不一致。纯文本安全渲染，不把模型/知识内容当 HTML。旧无来源行仅显示可验证的 chunk，注明缺失原文路径。

回答左下方👍/👎，只在回答正常完成后可点击。点击一次点亮所选、显示“已反馈”、禁用两按钮；前端保存带本地回答 ID、会话 ID、选择和时间的信号，并保留未来接入的数据接口边界，不调用后端、不进 user_feedback 池。存储不可用时页面仍锁定一次选择。该部分按用户要求直接实现并做浏览器点击验收。

## 8. 评估体系与 faith_cases

从四份真实文档人工编写并对照标注：独立校准集 12 题、冻结测试集 60 题。正式桶 A_policy 12、B_model 12、C_colloquial 12、D_unknown 12、E_multi 12；每桶覆盖 easy/medium/hard。增加型号近似干扰、口语、条件例外、首尾布局、多来源回答、不存在型号、到账承诺、政策冲突误读和未知事实样例。

每题包含稳定 eval_id（<=16）、bucket、difficulty、query、filters、answerable、ground_truth、required_facts、forbidden_claims、source_file/section/evidence_quote 标注。不用运行时数据库自增 ID 作为跨环境 gold；执行前以原文/章节/证据片段绑定真实 chunk，缺失或歧义就失败。多来源题明确所有必要证据，不能命中任一条就算完整召回。

计算 Recall@1/5/10/50（检出的相关 chunk 占标注相关 chunk 的比例）及 MRR；另报多证据全覆盖率。D_unknown 没有 gold chunk，召回指标标 N/A，不混入分母。报告区分 Top-50 召回、Top-10 重排排名和预算裁剪后实际 prompt 证据覆盖。

Faithfulness 独立裁判拆解答案事实主张，逐条对照实际喂入的证据全集，分数为有证据支持主张数/全部可核实主张数。无事实主张的纯拒答标 N/A，不能算 1.0 抬高平均分；另报拒答准确率、answerable 问题覆盖率/误拒率、unknown 问题拒答率、禁止承诺违规数与引用合法率。裁判失败显式计数、不记成忠实；模型自评 useful 不能代替 Faithfulness。

四策略逐题生成并记录同样的生成/裁判模型、参数、原文指纹、标注集版本、查询改写、过滤、实际证据、答案、评分理由、耗时。输出带 run ID 的 JSON 逐题结果和 Markdown 报告，按策略、query 桶、难度聚合，保留原轮次，不只覆盖单个 latest 文件。同模型裁判时明确报告偏差局限；支持独立 JUDGE_MODEL 配置。

裁判发现不受支持主张则写 faith_cases，使用用户定义的 eval_id 唯一键原子 upsert：最新策略/答案/理由/完整证据/judge_model 覆盖，seen_count+1，last_seen_at 更新，first_seen_at 保留，status 回未解决、resolution 清空，resolved_at 保留以显示复发。一个题在四策略产生多次编造时每次事件计数，最终快照是该轮最后一次失败，报告仍保存每个策略完整结果。

提供最小台账查询与处置命令；标“已解决/无需解决”强制填写非空处置说明并设置 resolved_at。本章不扩展人工运营后台。

## 9. 验证与交付

后端先 RED 再实现：原生 schema/两路 Top-50/RRF 及过滤、重排 Top-10、源文档安全定位、引用编号、低置信度入池、取消持久化、faith_cases 跨轮复发与处置必填。真实 MySQL/Milvus 集成测试不能用 mocks 冒充；测试 mocks 只验证服务边界。Prompt/数据用标注样例验证。前端浏览器验收引用展开/原文跳转/反馈点亮与锁定。

实际验收命令最终写入 README：环境启动、新表/迁移、真实知识建库、四策略评估、聊天启动、型号命中演示、缺失知识拒答及池记录查询、台账处置、完整测试。正式评估必须运行指定本地模型与真实 Milvus；无法跑真模型/服务时报告阻塞，不生成伪数字或替换技术。

按任务即时更新 dev-notes/ch04.md；阶段记录含用户原话、产物/评审、拒绝/纠偏、翻车/返工。后端完成任务评审和整体评审；前端按用户例外不套代码评审。完成并验证后提交、推送 GitHub，保留用户 test.py 与所有本地凭据。

## 10. Context7 已查证的官方接口

查询时间 2026-10-05；执行时仍与实际安装包签名核对，不把最新 master 的新增接口当 2.6 可用接口。

- Milvus/PyMilvus：Function/FunctionType.BM25、VARCHAR analyzer、SPARSE_FLOAT_VECTOR、SPARSE_INVERTED_INDEX/BM25、AnnSearchRequest、MilvusClient.hybrid_search、RRFRanker。https://github.com/milvus-io/pymilvus/blob/master/examples/full_text_search/bm25.py 与 https://milvus.io/docs/chinese-analyzer.md 。使用现有版本可用的 VARCHAR；Context7 一次返回 3.0.x 的 TEXT 示例，未选用。
- FastAPI：保留当前 fastapi.sse.EventSourceResponse/ServerSentEvent 的 POST SSE。https://fastapi.tiangolo.com/tutorial/server-sent-events/ 。
- SQLAlchemy 2.0：MySQL insert.on_duplicate_key_update、inserted、ENUM/JSON；upsert 显式更新时间，Python onupdate 不会自动执行。https://docs.sqlalchemy.org/en/20/dialects/mysql.html 。
- LangChain OpenAI：with_structured_output(..., method='json_mode', include_raw=True)，JSON mode 必须在 Prompt 明确字段并验证解析失败。https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI/with_structured_output 。
- Sentence Transformers：CrossEncoder 关键字参数、predict activation_fn；显式固定模型 BAAI/bge-reranker-v2-m3。https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/cross_encoder/model.py 。

## 11. 草案自查与下一阶段

已检查固定技术、四种分数尺度、两个拒答入口与 useful=false、原文定位和快照、Top-10 与上下文预算、未知题/拒答的指标分母、台账复发、真实数据和前端流程例外；未发现需更换固定技术的矛盾。

尚需用户审阅本草案。通过后按 writing-plans 产出实施计划，再按技能要求评审计划、选择执行方式。草案状态不能记录成“brainstorm 定稿”或“计划评审通过”。
