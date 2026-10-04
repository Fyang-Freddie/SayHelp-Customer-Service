"""Validate tool calls and bound execution without retrying ticket writes."""

import asyncio
import json
import math

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError


_READ_TOOLS = frozenset({'query_order', 'query_product', 'query_logistics', 'query_faq'})


class ToolExecutor:
    def __init__(self, tools: dict[str, BaseTool], timeout_seconds: float = 8,
                 max_read_attempts: int = 2):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be finite and positive')
        if isinstance(max_read_attempts, bool) or not isinstance(max_read_attempts, int) or max_read_attempts < 1:
            raise ValueError('max_read_attempts must be a positive integer')
        self.tools = tools
        self.timeout_seconds = timeout_seconds
        self.max_read_attempts = max_read_attempts

    async def execute(self, call: dict) -> ToolMessage:
        """Return a matching result even when validation or execution fails."""
        def error(content: str) -> ToolMessage:
            return ToolMessage(content=content, tool_call_id=call['id'], status='error')

        name = call.get('name')
        if not isinstance(name, str) or name not in self.tools:
            return error('工具名称无效，请核实后再试。')
        selected = self.tools[name]
        args = call.get('args')
        if not isinstance(args, dict):
            return error('工具参数无效，请核实客户提供的信息。')
        try:
            # Validate before invocation, so invalid requests cannot cause side effects.
            validated = selected.get_input_schema().model_validate(args, strict=True).model_dump()
        except (ValidationError, TypeError, ValueError):
            return error('工具参数无效，请核实客户提供的信息。')

        attempts = self.max_read_attempts if name in _READ_TOOLS else 1
        for _ in range(attempts):
            try:
                result = await asyncio.wait_for(selected.ainvoke(validated), timeout=self.timeout_seconds)
                content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                return ToolMessage(content=content, tool_call_id=call['id'], status='success')
            except Exception:
                # Sync tools run in workers that may finish after timeout. Never retry a write.
                if name == 'create_ticket':
                    return error('工单创建结果暂时无法确认，请人工核实是否已创建，避免重复提交。')
        if name == 'query_faq':
            # The outer execution deadline can fire before query_faq handles its
            # dependency exception. Preserve its public result shape here too.
            return error(json.dumps(
                {'keyword': validated['keyword'], 'matches': [],
                 'message': '知识库查询暂时不可用，请稍后重试或人工核实'},
                ensure_ascii=False))
        return error('工具查询暂时失败，请稍后重试或联系人工客服。')
