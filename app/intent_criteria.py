"""Choice criteria describe boundaries, never few-shot classification examples."""
import hashlib
import json

INSTRUCTIONS = '判断当前问题的主要诉求，只选一个类别。遵从当前明确目标与纠正，不把情绪当成投诉；无可靠类别或无主次的独立多诉求选择其他。这里只判断要干什么，不追问客户ID、订单号或原因。'
CRITERIA = {
    '物流':'查询发货、配送、到货、轨迹、签收和物流异常。当前明确要求退款时不属此类。',
    '订单':'交易记录、付款、发票、地址修改、订单取消等订单操作。不包含退款退货资格、退款金额或退款进度。',
    '商品咨询':'商品规格、功能、兼容性、普通使用方法等信息。已有故障、维修、保修和换货诉求属售后。',
    '退款退货':'退货或退款资格、条件、申请、金额、到账和进度。即使涉及损坏或物流，当前明确目标是退货退款也属此类。',
    '售后':'故障处理、维修、保修、换货、补发及售后服务。明确要求退货或退款时归退款退货。',
    '投诉':'主要目标为投诉、追责或服务争议升级。附带不满或语气激烈不能覆盖明确的物流、订单或退款目标。',
    '闲聊':'问候、感谢及普通社交聊天。业务外专业问题、乱码或无法理解的文本不属此类。',
    '其他':'业务范围外、乱码、信息不足无法确定目标，或多个独立诉求没有主次。无法可靠归入其余类别时选择本类。',
}


def criteria_fingerprint():
    return hashlib.sha256(json.dumps({'instructions':INSTRUCTIONS,'criteria':CRITERIA},ensure_ascii=False,sort_keys=True).encode()).hexdigest()
