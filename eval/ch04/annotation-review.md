# ch04 标注复核记录

2026-10-06，作者逐条对照四份仓库原文完成标注。题目由作者编写；业务事实仅从原文摘取，未生成新商品/政策事实，不含真实会话、订单号或个人信息。

- calibration.json：12题，A/B/C各2题、D/E各3题，easy/medium/hard各4题。
- test.json：冻结60题，A_policy/B_model/C_colloquial/D_unknown/E_multi各12题，每桶easy/medium/hard各4题。
- 每条可答题明确required_facts与禁止承诺；所有必要来源都标anchor。未知题无gold，ground_truth要求明确拒答。
- anchor由source_file/section_path/原文精确quote构成，不保存运行时MySQL ID。文件指纹固定；正式运行再核验原文及当前MySQL并唯一绑定。
- 自动复核覆盖重复ID、规范化/非常近似问法、跨集相同全套required_facts、章节/quote不存在或歧义、来源指纹/正文变更。语义近似另外按问题目标人工对照；共享文档或章节允许，但校准与测试没有同一近似问法。未以测试分数筛选或修改标注。

复核要点：B02/E01明确MH-LP50不含App，B11/E03明确MH-FD10无摄像头/远程App；通用设备激活说明不能覆盖具体型号例外。E07区分退款响应2小时、处理3个工作日与到账规则，禁止承诺到账。D05未用猫砂容量代替机身尺寸，D06/D07/D11/D12没有从近似型号或功能推断未记录参数。D08/D09/D10不编实时库存、未来优惠或精确到账时刻。

所有72题选定quote已精确匹配当前文档chunk；正式真实MySQL绑定及指定真实reranker验证27项通过（42.72s）。冻结依据为本次文件内容及Git版本，后续运行必须报告校准/测试文件与来源digest；不得根据测试成绩改题或门槛。重排器固定BAAI/bge-reranker-v2-m3，revision953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e，原始logits。
