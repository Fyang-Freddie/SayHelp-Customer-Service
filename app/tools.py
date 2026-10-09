"""Five conversation-scoped customer service tools."""

from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.repository import Repository
from app.knowledge_types import KnowledgeFilters
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
    filters: KnowledgeFilters | None = None


class TicketInput(ToolInput):
    description: NonBlankString
    ticket_type: Literal['售后', '投诉', '咨询']


def build_tools(repository: Repository, conversation_id: int, knowledge_search: KnowledgeSearch | None, *, knowledge_answer=None, ticket_request_key: str | None = None) -> dict[str, BaseTool]:
    """Capture the active conversation; the model cannot supply its identity."""
    @tool(args_schema=OrderInput)
    def query_order(order_id: str) -> dict:
        """用客户提供的订单号查询订单；当前数据源尚未接入，不能返回订单状态或金额。"""
        return {'order_id': order_id, 'available': False, 'reason': 'data_source_not_connected',
                'message': '数据源尚未接入，无法查询真实记录，请人工核实'}

    @tool(args_schema=ProductInput)
    def query_product(product_query: str) -> dict:
        """查询商品资料；未接入真实文档检索时说明数据源尚未接入，不编造价格、库存或规格。"""
        return {'product_query': product_query, 'available': False, 'reason': 'data_source_not_connected',
                'message': '数据源尚未接入，无法查询真实记录，请人工核实'}

    @tool(args_schema=OrderInput)
    def query_logistics(order_id: str) -> dict:
        """用客户提供的订单号或运单号查询物流；当前数据源尚未接入，不能返回物流状态或位置。"""
        return {'order_id': order_id, 'available': False, 'reason': 'data_source_not_connected',
                'message': '数据源尚未接入，无法查询真实记录，请人工核实'}

    @tool(args_schema=FaqInput)
    def query_faq(keyword: str, filters: KnowledgeFilters | None = None) -> dict:
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
        number = repository.create_ticket(conversation_id, description, ticket_type, request_key=ticket_request_key)
        return {'ticket_no': number, 'status': '待处理', 'message': '已创建工单，等待处理'}

    if knowledge_answer is not None:
        @tool('query_faq', args_schema=FaqInput)
        async def grounded_faq(keyword: str, filters: KnowledgeFilters | None = None) -> dict:
            """基于原文检索和充分性自评回答知识问题，可按显式品类过滤；useful=false不得据证据编答。"""
            return await knowledge_answer(keyword, filters)

        @tool('query_product', args_schema=ProductInput)
        async def grounded_product(product_query: str) -> dict:
            """从真实商品文档查询规格并验证充分性；不能查询实时库存、价格或返回模拟规格。"""
            return await knowledge_answer(product_query)
        query_faq = grounded_faq
        query_product = grounded_product

    return {item.name: item for item in
            (query_order, query_product, query_logistics, query_faq, create_ticket)}
