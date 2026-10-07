"""Bounded model/tool steps shared by the naked loop and later graph nodes."""
from dataclasses import dataclass
import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.tool_executor import ToolExecutor
from app.workflow_types import AgentModel, AgentResult, Emit, Usage, WorkflowState

READ_TOOLS = frozenset({'query_order', 'query_product', 'query_logistics', 'query_faq'})
SUGGEST_ACTIONS = {'type': 'function', 'function': {
    'name': 'suggest_actions', 'description': '建议用户选择人工或工单按钮；不会执行任何业务写操作。',
    'parameters': {'type': 'object', 'additionalProperties': False,
        'properties': {'actions': {'type': 'array', 'maxItems': 2, 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {'kind': {'enum': ['handoff', 'create_ticket']},
                'description': {'type': 'string'}, 'ticket_type': {'enum': ['售后', '投诉', '咨询']}},
            'required': ['kind']}}}, 'required': ['actions']}}}


@dataclass(frozen=True)
class AgentLimits:
    model_calls: int = 6
    tool_calls: int = 6
    turn_tokens: int = 12000
    response_tokens: int = 512
    selection_tokens: int = 512

    def __post_init__(self):
        for name in ('model_calls', 'tool_calls', 'turn_tokens', 'response_tokens', 'selection_tokens'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.model_calls > 6 or self.tool_calls > 6:
            raise ValueError('Chapter 5 permits at most six model calls and six tool executions')


def _usage(state):
    value = state.get('usage', {})
    return value if isinstance(value, Usage) else Usage(**value)


def _estimate(messages):
    return max(1, int(count_tokens_approximately(messages)))


def _account(usage, metadata, estimated):
    if metadata is not None:
        return Usage(usage.input_tokens + metadata['input_tokens'],
                     usage.output_tokens + metadata['output_tokens'], usage.estimated).to_dict()
    return Usage(usage.input_tokens, usage.output_tokens, usage.estimated + estimated).to_dict()


def _ids(messages):
    return {c.get('id') for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}


def _valid_calls(message, previous):
    calls = message.tool_calls
    ids = [c.get('id') for c in calls]
    return (not message.invalid_tool_calls and all(isinstance(i, str) and i.strip() for i in ids)
            and len(ids) == len(set(ids)) and not set(ids) & previous)


def _paired(messages):
    pending, seen = set(), set()
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                return False
            pending.remove(message.tool_call_id)
        else:
            if pending:
                return False
            if isinstance(message, AIMessage):
                if not _valid_calls(message, seen):
                    return False
                pending.update(c['id'] for c in message.tool_calls)
                seen.update(pending)
    return not pending


async def agent_step(state: WorkflowState, *, model: AgentModel, tools: dict,
                     limits: AgentLimits, emit: Emit) -> dict:
    """Select a next tool batch, preserving one paid final generation call."""
    messages = state['messages']
    if not _paired(messages):
        return {'status': 'stopped', 'stop_reason': 'invalid_messages'}
    if state.get('model_calls', 0) >= limits.model_calls - 1:
        return {'status': 'answer', 'stop_reason': 'model_budget'}
    schemas = [convert_to_openai_tool(t) for n,t in tools.items() if n in READ_TOOLS] + [SUGGEST_ACTIONS]
    schema_tokens = max(1, len(json.dumps(schemas, ensure_ascii=False)) // 3)
    input_tokens = _estimate(messages) + schema_tokens
    # Reserve selection output, then the expanded final input and its output.
    remaining = limits.turn_tokens - _usage(state).total
    needed = input_tokens + limits.selection_tokens + _estimate(messages) + limits.selection_tokens + limits.response_tokens
    if remaining < needed:
        return {'status': 'answer', 'stop_reason': 'token_budget'}
    calls = state.get('model_calls', 0) + 1
    await emit('model_start', {'phase': 'select', 'model_calls': calls})
    selected = await model.select(messages, schemas, max_tokens=limits.selection_tokens)
    update = {'model_calls': calls, 'usage': _account(_usage(state), selected.usage_metadata,
              input_tokens + _estimate([selected]))}
    if not _valid_calls(selected, _ids(messages)):
        return {**update, 'status': 'answer', 'stop_reason': 'invalid_tool_calls'}
    if not selected.tool_calls:
        # Selection text is an internal draft; final generation uses prior context.
        return {**update, 'status': 'answer', 'stop_reason': 'complete'}
    # Discard text accompanying tool requests too: only structured calls are fed back.
    selected = AIMessage(content='', tool_calls=selected.tool_calls)
    # Establish the complete fallback batch invariant before adding any pending
    # calls to history. Every result admission below preserves this same bound.
    minimum_results = [_result_budget_error(c) for c in selected.tool_calls]
    if not _fits_final(messages + [selected] + minimum_results, Usage(**update['usage']), limits):
        return {**update, 'status': 'answer', 'stop_reason': 'tool_result_budget'}
    return {**update, 'messages': [selected], 'status': 'tools'}


def _suggestions(args):
    if not isinstance(args, dict) or set(args) != {'actions'}:
        raise ValueError()
    actions = args['actions']
    if not isinstance(actions, list) or not 1 <= len(actions) <= 2:
        raise ValueError()
    for action in actions:
        if not isinstance(action, dict) or set(action) - {'kind', 'description', 'ticket_type'}:
            raise ValueError()
        if action.get('kind') == 'handoff':
            if set(action) != {'kind'}:
                raise ValueError()
        elif action.get('kind') == 'create_ticket':
            if (set(action) != {'kind', 'description', 'ticket_type'}
                or not isinstance(action['description'], str) or not action['description'].strip()
                or len(action['description']) > 2000 or action['ticket_type'] not in ('售后','投诉','咨询')):
                raise ValueError()
        else:
            raise ValueError()
    return actions


def _failure_streak(messages, call):
    """Count consecutive actual failed attempts for the same read operation."""
    calls = {c['id']: c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    count = 0
    for m in reversed(messages):
        if not isinstance(m, ToolMessage) or m.name == 'suggest_actions':
            continue
        prior = calls.get(m.tool_call_id)
        if prior is None or prior['name'] != call['name'] or prior['args'] != call['args'] or m.status != 'error':
            break
        count += max(1, m.additional_kwargs.get('execution_attempts', 0))
    return count


def _fits_final(messages, usage: Usage, limits: AgentLimits):
    return usage.total + _estimate(messages) + limits.response_tokens <= limits.turn_tokens


def _result_budget_error(call, *, attempts=0):
    return ToolMessage(content='本轮剩余输入预算不足，工具结果未采用，请核实。',
                       tool_call_id=call['id'], name=call['name'], status='error',
                       additional_kwargs={'execution_attempts': attempts})


async def execute_calls(state: WorkflowState, *, tools: dict, limits: AgentLimits, emit: Emit) -> dict:
    """Execute permitted reads serially; retries spend the same business budget."""
    messages = state['messages']
    selected = messages[-1]
    if not isinstance(selected, AIMessage) or not selected.tool_calls or not _valid_calls(selected, _ids(messages[:-1])):
        return {'status': 'answer', 'stop_reason': 'invalid_tool_calls'}
    count = state.get('tool_calls', 0)
    results, suggestions = [], list(state.get('suggestions', []))
    reason = None
    executor = ToolExecutor({n:t for n,t in tools.items() if n in READ_TOOLS})
    async def attempt(name, number):
        nonlocal count
        count += 1
        await emit('tool_attempt', {'name': name, 'attempt': number, 'tool_calls': count})
    # Validate the whole batch before the first side effect.
    denied = any(c['name'] not in READ_TOOLS | {'suggest_actions'} or
                 (c['name'] in READ_TOOLS and c['name'] not in tools) for c in selected.tool_calls)
    for index, call in enumerate(selected.tool_calls):
        if denied or reason:
            result = ToolMessage(content='本轮工具执行已停止，需要核实。', tool_call_id=call['id'], name=call['name'], status='error')
            reason = reason or 'denied_tool'
        elif call['name'] == 'suggest_actions':
            try:
                suggestions = _suggestions(call['args'])
                result = ToolMessage(content='已记录建议，等待用户自行选择；尚未执行任何操作。',
                                     tool_call_id=call['id'], name=call['name'])
                await emit('suggestions', {'suggestions': suggestions})
            except ValueError:
                result = ToolMessage(content='建议参数无效。', tool_call_id=call['id'], name=call['name'], status='error')
        elif count >= limits.tool_calls:
            result = ToolMessage(content='本轮工具预算已用完。', tool_call_id=call['id'], name=call['name'], status='error')
            reason = 'tool_budget'
        else:
            await emit('tool_start', {'name': call['name'], 'tool_call_id': call['id']})
            result = await executor.execute(call, max_attempts=limits.tool_calls-count, on_attempt=attempt)
            result.name = call['name']
            await emit('tool_end', {'name': call['name'], 'tool_call_id': call['id'], 'status': result.status,
                                    'attempts': result.additional_kwargs.get('execution_attempts', 0)})
            if result.status == 'error' and _failure_streak(messages + results + [result], call) >= 2:
                reason = 'repeated_tool_failure'
            elif count >= limits.tool_calls:
                reason = 'tool_budget'
        # Admission includes prior results and short paired results for every
        # pending call. A generated read can otherwise consume the allowance
        # reserved for the final input, even though its execution cost is paid.
        pending = [_result_budget_error(c) for c in selected.tool_calls[index + 1:]]
        if not _fits_final(messages + results + [result] + pending, _usage(state), limits):
            result = _result_budget_error(call, attempts=result.additional_kwargs.get('execution_attempts', 0))
            # This exact replacement was reserved at admission (or in the
            # preceding iteration), so the complete paired fallback must fit.
            assert _fits_final(messages + results + [result] + pending, _usage(state), limits)
            reason = 'tool_result_budget'
            await emit('tool_result_rejected', {'name': call['name'], 'tool_call_id': call['id'],
                                               'reason': reason})
        results.append(result)
    return {'messages': results, 'tool_calls': count, 'suggestions': suggestions,
            'status': 'answer' if reason else 'select', **({'stop_reason': reason} if reason else {})}


async def stream_answer(state: WorkflowState, *, model: AgentModel, limits: AgentLimits, emit: Emit) -> dict:
    """The terminal answer is a real, unbound model stream and costs one call."""
    messages = state['messages']
    if not _paired(messages):
        return {'status': 'stopped', 'stop_reason': 'invalid_messages'}
    if state.get('model_calls', 0) >= limits.model_calls:
        return {'status': 'stopped', 'stop_reason': 'model_budget'}
    input_tokens = _estimate(messages)
    if _usage(state).total + input_tokens + limits.response_tokens > limits.turn_tokens:
        return {'status': 'stopped', 'stop_reason': 'token_budget'}
    calls = state.get('model_calls', 0) + 1
    await emit('model_start', {'phase': 'final', 'model_calls': calls, 'tools_bound': False})
    answer, combined = '', None
    async for chunk in model.stream_reply(messages, max_tokens=limits.response_tokens):
        combined = chunk if combined is None else combined + chunk
        text = str(chunk.text)
        if text:
            answer += text
            await emit('token', {'text': text})
    metadata = combined.usage_metadata if combined is not None else None
    usage = _account(_usage(state), metadata, input_tokens + _estimate([AIMessage(content=answer)]))
    return {'messages': [AIMessage(content=answer)], 'answer': answer, 'usage': usage,
            'model_calls': calls, 'status': 'done', 'stop_reason': state.get('stop_reason', 'complete')}


def _apply(state, update):
    update = dict(update)
    if 'messages' in update:
        state['messages'] = state['messages'] + update.pop('messages')
    state.update(update)


async def run_bare_agent(messages: list, *, model: AgentModel, tools: dict,
                         limits: AgentLimits, usage: Usage, emit: Emit) -> AgentResult:
    """Explicit select -> execute -> feedback loop, without an agent executor."""
    state: WorkflowState = {'messages': list(messages), 'usage': usage.to_dict(), 'model_calls': 0,
                            'tool_calls': 0, 'answer': '', 'suggestions': [], 'status': 'select'}
    while state['status'] == 'select':
        _apply(state, await agent_step(state, model=model, tools=tools, limits=limits, emit=emit))
        if state['status'] == 'tools':
            _apply(state, await execute_calls(state, tools=tools, limits=limits, emit=emit))
    if state['status'] == 'answer':
        _apply(state, await stream_answer(state, model=model, limits=limits, emit=emit))
    return AgentResult(messages=state['messages'], answer=state['answer'], suggestions=state['suggestions'],
                       usage=_usage(state), model_calls=state['model_calls'], tool_calls=state['tool_calls'],
                       stop_reason=state.get('stop_reason', 'complete'))
