# 退款原因

> 以下为固定下拉枚举，共 50 条。后端保存 `原因编码`，前端展示 `显示名称`；仅在没有更精确选项时使用 `other`。

| 原因编码 | 显示名称 |
|---|---|
| no_longer_needed | 不想要了 |
| ordered_by_mistake | 下错单了 |
| duplicate_order | 重复下单 |
| wrong_size_selected | 尺码选错 |
| wrong_model_selected | 型号/规格选错 |
| wrong_color_selected | 颜色选错 |
| found_better_price | 发现更优惠的价格 |
| delivery_too_slow | 配送太慢 |
| not_shipped_on_time | 商家未按约定时间发货 |
| seller_out_of_stock | 商家缺货 |
| address_issue | 收货信息有误 |
| refuse_delivery | 暂时无法收货/想拒收 |
| quality_issue | 商品质量问题 |
| cannot_power_on | 无法开机/无法启动 |
| function_failure | 功能故障 |
| intermittent_failure | 间歇性故障 |
| abnormal_noise | 异常噪音 |
| overheating | 异常发热 |
| battery_issue | 电池/续航问题 |
| screen_issue | 屏幕/显示问题 |
| connection_issue | 连接/网络问题 |
| damaged_item | 收到商品破损 |
| leaking_item | 商品漏液/渗漏 |
| deformed_item | 商品变形 |
| stained_item | 商品有污渍 |
| expired_item | 商品已过期 |
| near_expiry | 临近保质期 |
| spoiled_item | 食品/生鲜变质 |
| wrong_item | 发错商品 |
| wrong_variant | 发错型号/规格/颜色 |
| missing_item | 少发商品 |
| missing_accessory | 缺少配件 |
| missing_gift | 缺少赠品 |
| quantity_mismatch | 数量与订单不符 |
| not_as_described | 商品与描述不符 |
| counterfeit_suspected | 怀疑非正品 |
| material_mismatch | 材质与描述不符 |
| size_mismatch | 实际尺寸与描述不符 |
| used_item_received | 疑似收到已使用商品 |
| packaging_damaged | 商品包装严重破损 |
| seal_broken | 密封/封签异常 |
| logistics_lost | 物流疑似丢件 |
| logistics_no_update | 物流长时间无更新 |
| logistics_returned | 物流异常退回 |
| delivered_not_received | 显示签收但未收到 |
| refund_after_refusal | 拒收后申请退款 |
| repair_failed | 维修后问题仍未解决 |
| replacement_failed | 换货后仍有问题 |
| merchant_agreed_refund | 已与商家协商一致退款 |
| other | 其他 |
