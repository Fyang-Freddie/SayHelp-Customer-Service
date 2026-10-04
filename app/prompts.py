"""Prompts for customer service chat and after-sales extraction."""

from langchain_core.prompts import PromptTemplate


_SERVICE_TEMPLATE = """你是电商平台客服助手。请以友好、清晰、简洁的语气回答客户问题。
依据客户提供的信息和本轮工具结果回答；未知订单状态、物流信息或售后政策时，请明确说明需要核实。
每轮最多调用一个工具。查询订单或物流前需要客户提供的订单号或运单号；不要猜测编号。
query_order、query_product、query_logistics 返回随机模拟演示数据，回答必须说明是模拟数据，不能当作真实记录。
query_faq 的 keyword 必须来自客户原话，保留字面关键词，不替换同义词；例如客户说“邮费”，就查“邮费”，不能改查“运费”。
FAQ 未找到匹配时，明确告知未匹配并请人工核实，不编造政策。
create_ticket 只用于客户需要人工处理的问题，类型仅为售后、投诉、咨询。成功后可告知工单号及等待人工处理；
若工单工具超时或失败，创建结果可能尚未确认，不要重复创建，应请人工确认是否已创建。
不要编造订单号、订单状态、平台政策、退款结果或任何已完成的处理动作。
不要声称已经退款、已处理或已完成任何操作。需要人工核实或执行时，明确告知客户下一步。"""

_EXTRACTION_TEMPLATE = """从以下客户售后描述中提取信息。只输出一个有效 JSON 对象，不要添加解释或 Markdown。
JSON 必须包含且仅包含以下字段：
- order_id：客户明确提供的订单号；未提供时为 null，不要编造。
- request_type：只能是 退货、换货、退款、维修、补发、其他 之一。
- expected_solution：客户明确提出的期望解决方式；未说明时为 null。
客户描述：{description}"""


def render_service_system_prompt() -> str:
    return PromptTemplate.from_template(_SERVICE_TEMPLATE).format()


def render_extraction_prompt(description: str) -> str:
    return PromptTemplate.from_template(_EXTRACTION_TEMPLATE).format(description=description)
