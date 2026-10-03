"""Prompts for customer service chat and after-sales extraction."""

from langchain_core.prompts import PromptTemplate


_SERVICE_TEMPLATE = """你是电商平台客服助手。请以友好、清晰、简洁的语气回答客户问题。
只能依据客户提供的信息讨论订单；未知订单状态、物流信息或售后政策时，请明确说明需要核实。
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
