# ch04 · 混合检索、证据控制与评估设计

日期：2026-10-05。状态：用户已确认“设计文档过关”，设计定稿，进入实施计划编写与评审；历史恢复和来源原文展示已提前修复。历史删除、置顶随 ch04 一起实现。

## 1. 用户目标与固定边界

把现有 SayHelp 客服的检索质量和可追溯性提升到可测量、可复盘的水平。验收包括四策略数字报告、具体型号 BM25 命中、点击引用查看原文，以及知识缺失时明确拒答并记录原话。

新增历史管理需求：历史对话支持删除、置顶和取消置顶，刷新后保持操作结果。用户此前要求“先不要实现，等后面一起实现”，现已批准整份设计；该功能纳入 ch04 实施计划，与其余功能一起实现。

固定使用 Milvus 原生 BM25、内置 chinese analyzer、hybrid_search + RRF、BAAI/bge-reranker-v2-m3。保留 BAAI/bge-m3 dense 嵌入。只对当前单条问题做归一化及检索侧同义词扩展；不做指代消解、多轮改写。不会为了改善数字复制同义问法入库。

知识来源为 knowledge_db 的 after-sales-manual.md、product-faq.md、product-specs.md、returns-policy.md。新建库和正式评估均不使用随机生成知识或第 2/3 章测试夹具。旧测试数据只用于自动测试。现有订单/物流演示工具继续明确标注模拟结果；商品规格问题必须走真实知识检索，不能被旧随机商品工具回答。

用户指定：后端代码走 Superpowers/TDD/评审；纯 Prompt、标注数据用真实标注样例验证代替 TDD；前端按 Vibe Coding 直接实现、浏览器验收，不套 brainstorm/TDD/code review。阶段完成时即时追记 dev-notes/ch04.md；最终验证、提交并推送指定 GitHub 仓库。

## 2. 现有代码与环境

- 初始设计基线为 main/4544b0f；当前基线为 main/f508777，已包含历史恢复和来源原文展示修复，四份真实知识 Markdown 已入版本管理。用户 test.py 保留未跟踪。
- Milvus 2.6.24、MySQL 8.4 容器健康，PyMilvus 2.6.17。符合用户要求的 Milvus 2.5 起原生全文检索。
- FastAPI 0.142.2、SQLAlchemy 2.0.54、langchain-core 1.6.6、langchain-openai 1.6.7、sentence-transformers 6.1.0。
- app/vector_store.py 当前 knowledge 集合只有 id/vector；app/knowledge_search.py 返回 question/answer/category，按 dense cosine 0.55 判匹配。
- MySQL knowledge_chunks 已保存章节路径和相邻指针，但未保存来源文件及行号。chunk_markdown 已计算 source_line，可扩展为精确起止范围。
- app/chat_service.py 由模型选一次工具，然后直接流式回答；只改 query_faq 内部不能保证模型选错工具或未调用工具时仍执行知识拒答。
- app/main.py 使用 FastAPI 原生 SSE，已增加历史列表、历史消息、知识文档原文读取接口及 sources 事件。app/web/index.html 已支持历史切换/刷新恢复、来源按钮/引用点击和原文弹窗；满意度反馈、历史删除/置顶尚未实现。

## 3. 接入方案比较与推荐

推荐：新增独立知识回答管线，由单条 query 理解结果确定知识/操作/闲聊路由；知识路由强制检索、自评与引用校验，操作路由复用现有订单/物流/工单流程。query_faq 复用同一检索服务，商品规格不再调用随机 query_product。这样知识拒答与引用不依赖模型自觉选择工具。

备选一：仅改 query_faq 和 System Prompt。改动小，但工具未调用时可绕过证据控制，不满足强制拒答验收。

备选二：所有消息都强制 RAG。证据门控简单，但订单操作、问候会被无关知识拦截，破坏现有功能。

三者均不改变指定检索技术；推荐方案只增加必要边界，不引入 LangGraph/Langfuse 或新的外部服务。

## 4. 数据与索引迁移

原样保存用户给出的 low_confidence_questions / faith_cases DDL 到 db/ch04_schema.sql，包括 SET NAMES utf8mb4、外键、ENUM、唯一键、注释、全部处置与复发字段。ORM 与真实 MySQL information_schema 对照验证。初始化必须幂等，发现半成品/不兼容 schema 明确报错。

另外增加显式 ch04 知识溯源迁移：knowledge_chunks 的 source_file（仓库相对路径）、source_start_line、source_end_line、product_category（标准品类）、source_digest（原文版本指纹）。旧行字段可空，不伪造来源。source_file 不接受任意客户端磁盘路径。正文、章节、chunk ID 的权威源仍是 MySQL。

历史管理另外通过显式迁移扩展 conversations：is_pinned（默认否）、pinned_at（可空）、deleted_at（可空），并增加适配可见历史/置顶排序的索引。旧会话迁移后默认未置顶、未删除。删除采用逻辑删除，隐藏会话及其消息，但保留底层消息、工单、低置信度问题等关联记录供复盘；不级联物理清除。用户提供的两张新表 DDL 不因此改写。此项本轮不执行迁移，随 ch04 后续实现。

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

### 历史对话删除与置顶（新增，待 ch04 一起实现）

在现有历史列表每条会话旁提供“更多”操作菜单，包含“置顶/取消置顶”和“删除对话”。菜单在桌面与手机均可直接操作，有明确文字和键盘可访问入口；点击菜单操作不会误触会话切换。

置顶状态持久化到后端，不只存浏览器。置顶会话显示图钉标识并排在列表最上方；多条置顶按置顶时间倒序，普通会话继续按最近消息活动倒序。取消置顶后回到普通会话的活动排序位置。重复设置相同状态幂等，不刷新既有置顶时间。列表分页游标需包含置顶排序所需信息，与最终排序一致，不能沿用当前仅按消息 ID 的游标造成漏项或重复。置顶操作成功后刷新列表并保持当前会话选中状态。

删除前显示确认框，标明会话标题并说明“从历史列表移除，相关审计记录仍保留”。确认后逻辑删除会话；历史列表、历史消息读取和继续聊天入口都不再提供该会话，刷新后也不能恢复它。删除当前会话时清除浏览器保存的当前会话 ID，切换为空白新对话；删除其他会话不打断当前聊天。取消确认不产生变化；请求失败保留原列表与选中状态，显示失败说明，不假装操作成功。

沿用现有会话 API 增加删除接口与设置置顶状态接口，使用十进制字符串会话 ID。删除、置顶均由服务端校验目标会话和状态；已删除会话不能被再次置顶或续聊。生成中的会话禁用这两项入口，服务端对活跃会话拒绝操作并返回可理解的冲突提示，避免界面操作与流式写入竞态。已完成的重复删除保持幂等。删除结果不新增低置信度问题、工单或满意度反馈记录。

本次仅记录以上契约，不新增按钮、接口、字段或迁移。后续实现遵循已有分工：必要后端逻辑走 TDD 与评审；前端按用户指定的 Vibe Coding 方式直接实现并做浏览器验收。

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

新增历史管理验收：置顶/取消置顶与排序在刷新后保持；多条置顶与普通会话分页无重复、无遗漏；删除确认可取消；删除当前会话后进入空白新对话且刷新不恢复已删除会话；删除其他会话不影响当前聊天；服务端拒绝已删除会话读取/续聊、活跃会话操作和无效 ID；接口失败不误改界面。数据库验证逻辑删除保留关联消息、工单及低置信度问题，重复操作幂等；桌面/手机浏览器验证菜单、图钉、确认框和错误提示。以上均在后续实现时执行，本轮文档更新不运行产品功能验收。

实际验收命令最终写入 README：环境启动、新表/迁移、真实知识建库、四策略评估、聊天启动、型号命中演示、缺失知识拒答及池记录查询、台账处置、完整测试。正式评估必须运行指定本地模型与真实 Milvus；无法跑真模型/服务时报告阻塞，不生成伪数字或替换技术。

按任务即时更新 dev-notes/ch04.md；阶段记录含用户原话、产物/评审、拒绝/纠偏、翻车/返工。后端完成任务评审和整体评审；前端按用户例外不套代码评审。完成并验证后提交、推送 GitHub，保留用户 test.py 与所有本地凭据。

## 10. Context7 已查证的官方接口

查询时间 2026-10-05；执行时仍与实际安装包签名核对，不把最新 master 的新增接口当 2.6 可用接口。

- Milvus/PyMilvus：Function/FunctionType.BM25、VARCHAR analyzer、SPARSE_FLOAT_VECTOR、SPARSE_INVERTED_INDEX/BM25、AnnSearchRequest、MilvusClient.hybrid_search、RRFRanker。https://github.com/milvus-io/pymilvus/blob/master/examples/full_text_search/bm25.py 与 https://milvus.io/docs/chinese-analyzer.md 。使用现有版本可用的 VARCHAR；Context7 一次返回 3.0.x 的 TEXT 示例，未选用。
- FastAPI：保留当前 fastapi.sse.EventSourceResponse/ServerSentEvent 的 POST SSE。https://fastapi.tiangolo.com/tutorial/server-sent-events/ 。
- SQLAlchemy 2.0：MySQL insert.on_duplicate_key_update、inserted、ENUM/JSON；upsert 显式更新时间，Python onupdate 不会自动执行。https://docs.sqlalchemy.org/en/20/dialects/mysql.html 。
- LangChain OpenAI：with_structured_output(..., method='json_mode', include_raw=True)，JSON mode 必须在 Prompt 明确字段并验证解析失败。https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI/with_structured_output 。
- Sentence Transformers：CrossEncoder 关键字参数、predict activation_fn；显式固定模型 BAAI/bge-reranker-v2-m3。https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/cross_encoder/model.py 。

## 11. 设计自查与下一阶段

已检查固定技术、四种分数尺度、两个拒答入口与 useful=false、原文定位和快照、Top-10 与上下文预算、未知题/拒答的指标分母、台账复发、真实数据和前端流程例外；本次补查历史置顶持久化/分页排序、逻辑删除与关联记录保留、当前会话恢复、活跃生成冲突及失败交互。未发现需更换固定技术的矛盾。

用户已回复“设计文档过关”，本设计定稿。下一步按 writing-plans 产出实施计划并自查，再按技能要求由用户评审计划、选择执行方式。设计批准不等于实施计划已获批，也不等于功能已经实现。

## 附录 A. 用户提供的 ch04 DDL 原文

计划阶段将原始输入随设计保存，供后续实现与精确 schema 验证使用；本附录没有执行建表。

```SQL
-- =============================================================
-- ch04 · RAG 进阶 · 建表 DDL
-- 本章新建:low_confidence_questions(低置信度问题池)、faith_cases(编造个案台账)
-- 生成阶段模型自评知识不够答就拒答,把原话落进这张池子,是 ch09 数据飞轮的入口
-- 本章只靠 useful 自评判入池(useful=false 才入,不单独存该字段);ch09 会给这张表 ALTER 加归并字段
-- =============================================================

-- 确保中文 ENUM 定义值/DEFAULT/COMMENT 按 utf8mb4 解析
-- (否则 latin1 默认的 mysql client 会把中文 double-encode,ENUM 值存成乱码)
SET NAMES utf8mb4;

CREATE TABLE low_confidence_questions (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT '来源会话',
  raw_question    TEXT            NOT NULL                COMMENT '用户原话,带情绪口语',
  source          ENUM('retrieval_low_conf','self_check','user_feedback') NOT NULL COMMENT '入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决',
  reason          TEXT            NULL                    COMMENT '判不能的原因,留作复盘',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '入池时间',
  PRIMARY KEY (id),
  KEY idx_source (source),
  KEY idx_created_at (created_at),
  CONSTRAINT fk_lcq_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信度问题池';

-- ch04 编造个案台账:忠实度裁判判出的每条编造,不只留在这一轮的报告里,进表长期管理。
--
-- 为什么要一张表:报告是产物,重跑一次就被覆盖,上一轮判出的个案连带它的处置状态一起没了。
-- 而这些个案的价值恰恰在跨轮追溯——同一道题反复被判编造,说明库里那一格一直没补对。
--
-- 一题一行(uk_eval_id):同一道题再次被判编造不新增行,只把答案/理由/角标快照更新成最近一次、
-- seen_count 加一。已经标「已解决」的题又被判出来,状态自动退回「未解决」——那是复发,
-- 不是新问题,得让它重新出现在待处理列表里。
--
-- citations 存的是这一轮喂给模型的 Top-K 证据**全集**(答案里的角标 [n] 就是这份列表的序号)。
-- 裁判说「证据里没有」,追溯时必须能当场看到当时喂进去的到底是什么,不能只留一句结论;
-- 而且要看得出「手里有哪几条、实际只引了哪几条」——没被引用的那些同样是判断依据。
CREATE TABLE faith_cases (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  eval_id       VARCHAR(16)     NOT NULL                COMMENT '评估集题号,如 A43;一题一行',
  bucket        VARCHAR(24)     NOT NULL                COMMENT '题目所属桶:A_policy / B_model / C_colloquial / E_multi',
  query         VARCHAR(512)    NOT NULL                COMMENT '用户问题原文',
  strategy      VARCHAR(24)     NOT NULL DEFAULT 'hybrid_rerank' COMMENT '产出这条答案的检索策略',
  answer        TEXT            NOT NULL                COMMENT '被判编造的那版生成答案原文',
  reason        TEXT            NOT NULL                COMMENT '裁判给的理由:编在哪一句',
  citations     JSON            NULL                    COMMENT '这一轮喂给模型的 Top-K 证据全集快照:[{n,chunk_id,section_path,question,answer}];答案里的角标 [n] 就是这份列表的序号,答案通常只引用其中两三条;老数据没记为 NULL',
  judge_model   VARCHAR(64)     NULL                    COMMENT '判这条的裁判模型',
  status        ENUM('未解决','已解决','无需解决') NOT NULL DEFAULT '未解决' COMMENT '处置状态,人工点按钮改',
  seen_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '被判编造的累计次数(跨轮)',
  first_seen_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '第一次被判编造的时间',
  last_seen_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '最近一次被判编造的时间',
  resolution    VARCHAR(300)    NULL                    COMMENT '处置说明:标已解决要写清怎么解决的,标无需解决要写清为什么不用改;退回未解决时清空。空着的处置在台账上等于没有交代',
  resolved_at   DATETIME        NULL                    COMMENT '最近一次被标为已解决/无需解决的时间;复发后仍保留,用来标「复发」',
  PRIMARY KEY (id),
  UNIQUE KEY uk_eval_id (eval_id),
  KEY idx_status (status),
  KEY idx_last_seen_at (last_seen_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ch04 忠实度编造个案台账';
```
