"""Five conversation-scoped customer service tools."""

import random
from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.repository import Repository
from app.knowledge_search import KnowledgeSearch, KnowledgeSearchUnavailable


NonBlankString = Annotated[str, StringConstraints(min_length=1, pattern=r'\S')]


class ToolInput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class OrderInput(ToolInput):
    order_id: NonBlankString


class ProductInput(ToolInput):
    product_query: NonBlankString


class FaqInput(ToolInput):
    keyword: NonBlankString


class TicketInput(ToolInput):
    description: NonBlankString
    ticket_type: Literal['售后', '投诉', '咨询']


def build_tools(repository: Repository, conversation_id: int, knowledge_search: KnowledgeSearch) -> dict[str, BaseTool]:
    """Capture the active conversation; the model cannot supply its identity."""
    @tool(args_schema=OrderInput)
    def query_order(order_id: str) -> dict:
        """用客户提供的订单号查询订单演示数据；结果是随机模拟数据，必须明确标注。"""
        return {'mock': True, 'label': '随机模拟订单数据', 'order_id': order_id,
                'status': random.choice(['待付款', '待发货', '已发货', '已完成']),
                'amount': random.choice([99.0, 159.0, 299.0])}

    @tool(args_schema=ProductInput)
    def query_product(product_query: str) -> dict:
        """用商品名称或编号查询商品演示数据；结果是随机模拟数据，必须明确标注。"""
        return {'mock': True, 'label': '随机模拟商品数据', 'product_query': product_query,
                'price': random.choice([59.0, 129.0, 259.0]),
                'stock': random.choice([0, 12, 38, 100])}

    @tool(args_schema=OrderInput)
    def query_logistics(order_id: str) -> dict:
        """用客户提供的订单号或运单号查询物流演示数据；结果是随机模拟数据，必须明确标注。"""
        return {'mock': True, 'label': '随机模拟物流数据', 'order_id': order_id,
                'status': random.choice(['等待揽收', '运输中', '派送中', '已签收']),
                'location': random.choice(['发货仓库', '中转站', '当地配送站'])}

    @tool(args_schema=FaqInput)
    def query_faq(keyword: str) -> dict:
        """用客户的问题或关键词语义检索知识库，支持同义问法（如邮费与运费）。仅依据返回答案回答；无匹配或暂时不可用时请人工核实。"""
        try:
            rows = knowledge_search.search(keyword, limit=5)
        except KnowledgeSearchUnavailable:
            return {'keyword': keyword, 'matches': [],
                    'message': '知识库查询暂时不可用，请稍后重试或人工核实'}
        matches = [{'question': row.question, 'answer': row.answer, 'category': row.category}
                   for row in rows]
        return {'keyword': keyword, 'matches': matches,
                'message': '找到相关常见问题' if matches else '未找到匹配的常见问题，请人工核实'}

    @tool(args_schema=TicketInput)
    def create_ticket(description: str, ticket_type: Literal['售后', '投诉', '咨询']) -> dict:
        """为当前会话创建售后、投诉或咨询人工工单。仅表示待人工处理，不表示问题已解决。"""
        number = repository.create_ticket(conversation_id, description, ticket_type)
        return {'ticket_no': number, 'status': '待处理', 'message': '已创建工单并转人工，等待处理'}

    return {item.name: item for item in
            (query_order, query_product, query_logistics, query_faq, create_ticket)}
