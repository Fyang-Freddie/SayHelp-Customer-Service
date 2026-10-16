from dataclasses import replace
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from app.context_types import ContextSnapshot, TurnRecord
from app.context_budget import calculate_budget


def ledger(limit):
    value=calculate_budget(window=18000,prefix=0,evidence=0,summary=0,output=1,user=1,
        steps=1,selection=1,tool_results=1,protocol=32,target_turns=32,steady=1024)
    return replace(value,history=limit,layer1=limit*7//10,layer2=limit-limit*7//10)


def turns():
    return (TurnRecord('1',1,2,(HumanMessage('原话订单123'),AIMessage('答'*200))),
        TurnRecord('2',3,4,(HumanMessage('查物流'),AIMessage('',tool_calls=[dict(id='q',name='query_order',args={})]),
            ToolMessage('物流结果'*200,tool_call_id='q'),AIMessage('请等待'))),
        TurnRecord('3',5,6,(HumanMessage('谢谢'),AIMessage('不客气'))))


def test_layer1_verbatim_and_layer2_rules():
    from app.context_history import build_context_view
    view=build_context_view(ContextSnapshot(1),turns(),ledger(700))
    assert view.layer1==turns()[-1].messages
    assert view.layer2[0].content=='原话订单123'
    assert view.layer2[1].content=='答'*60
    tools=[m for m in view.layer2 if isinstance(m,ToolMessage)]
    assert tools[0].tool_call_id=='q' and '\n' not in tools[0].content
    assert '物流结果' not in tools[0].content
    assert turns()[1].messages[2].content=='物流结果'*200


def test_no_downgrade_when_history_fits():
    from app.context_history import build_context_view
    view=build_context_view(ContextSnapshot(1),turns(),ledger(4000))
    assert not view.layer2
    assert len(view.layer1)==8
    assert not view.pending_turns


def test_fixed_prefix_and_background_order():
    from app.context_history import build_context_view, compose_model_messages
    snapshot=ContextSnapshot(1,summary_upto_msg_id=2,layer1_from_msg_id=5,
        summaries=({'ordinal':1,'content':'订单123要求退款','from_msg_id':1,'upto_msg_id':2},))
    view=build_context_view(snapshot,turns(),ledger(4000))
    messages=compose_model_messages(SystemMessage('固定'),view,HumanMessage('最初订单？'),[{'answer':'政策'}],[])
    assert len([m for m in messages if m.type=='system'])==1
    assert messages[-2].content=='最初订单？'
    assert messages[-1].type=='human' and '订单123要求退款' in messages[-1].content
    assert '政策' in messages[-1].content


def test_pending_summary_uses_remaining_capacity_or_fails_closed():
    from app.context_history import build_context_view, compose_model_messages
    from app.context_budget import validate_request
    from app.history import InputBudgetExceeded
    view=build_context_view(ContextSnapshot(1),turns(),ledger(100))
    # Intermediate budget overflow is visible, never silently discarded.
    assert view.pending_turns and view.layer2[0].content=='原话订单123'
    messages=compose_model_messages(SystemMessage('固定'),view,HumanMessage('现在'),[],[])
    with pytest.raises(InputBudgetExceeded):validate_request(messages,[],100,200)
def test_resolver_budget_payload_includes_summary_and_entire_view():
    from app.query_resolver import resolver_messages
    from types import SimpleNamespace
    from langchain_core.messages import HumanMessage
    context=SimpleNamespace(layer2=(HumanMessage('早期用户原话'),),layer1=(HumanMessage('最近原话'),),summary_text='订单ABC123456尚待退款')
    messages,rows,known=resolver_messages('那个订单',[],{},context=context)
    assert known['session_summary']=='订单ABC123456尚待退款'
    assert [r['content'] for r in rows]==['早期用户原话','最近原话']
    assert 'ABC123456' in str(messages)

\n