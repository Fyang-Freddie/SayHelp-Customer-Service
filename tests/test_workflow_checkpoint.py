"""Real official SQLite persistence, inherited child checkpoints and turn isolation."""
import asyncio
from contextlib import closing
import importlib
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from test_workflow_graph import Model, build, collect, selected, turn


def test_close_reopen_restores_complete_history_and_inherited_child_checkpoints(tmp_path):
    async def scenario():
        api=importlib.import_module('app.workflow_graph')
        path=tmp_path/'nested/checkpoints.sqlite'
        config={'configurable':{'thread_id':'1'}}
        first=Model([selected(), AIMessage(content='draft')])
        async with api.open_workflow_checkpointer(path) as saver:
            graph,*_=build(tmp_path, model=first, checkpointer=saver)
            result,_,_=await collect(graph,turn(),config)
            assert result['answer']=='有依据的回答'
            original=result['messages']
            assert len(original)==4
            assert len({m.id for m in original})==4
            assert [type(m) for m in original]==[HumanMessage,AIMessage,ToolMessage,AIMessage]
        # A distinct saver and compiled graph must load, never execute the turn again.
        second=Model([])
        async with api.open_workflow_checkpointer(path) as saver:
            graph,*_=build(tmp_path,model=second,checkpointer=saver)
            snapshot=await graph.aget_state(config)
            assert snapshot.next==()
            assert snapshot.values['messages']==original
            recovered=await graph.ainvoke(None,config)
            assert recovered['messages']==original and recovered['answer']=='有依据的回答'
            assert second.selected==second.finals==[]
            empty=await graph.aget_state({'configurable':{'thread_id':'2'}})
            assert empty.values=={}
            with closing(sqlite3.connect(path)) as connection:
                namespaces=[row[0] for row in connection.execute('SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id=?',('1',))]
            assert '' in namespaces and any(ns.startswith('agent:') for ns in namespaces)
            persisted=[item async for item in saver.alist(None)]
            assert any(item.config['configurable']['checkpoint_ns'].startswith('agent:') for item in persisted)
        # Closed connection really releases the file on Windows.
        path.rename(path.with_suffix('.closed.sqlite'))
    asyncio.run(scenario())


def test_new_turn_clears_old_suggestions_evidence_and_paid_budgets_without_duplicate_history(tmp_path):
    async def scenario():
        api=importlib.import_module('app.workflow_graph')
        path=tmp_path/'turns.sqlite'
        config={'configurable':{'thread_id':'1'}}
        async with api.open_workflow_checkpointer(path) as saver:
            complaint,*_=build(tmp_path,'投诉',checkpointer=saver)
            result=await complaint.ainvoke(turn('我要投诉','t1'),config)
            assert len(result['suggestions'])==2
            knowledge,*_=build(tmp_path,'商品咨询',checkpointer=saver)
            result=await knowledge.ainvoke(turn('饮水机容量','t2'),config)
            assert result['suggestions']==[] and result['model_calls']==2
            assert result['evidence'] and result['citations']
            chat,model,retrieval,*_=build(tmp_path,'闲聊',checkpointer=saver)
            result=await chat.ainvoke(turn('你好','t3'),config)
            assert result['suggestions']==result['evidence']==result['citations']==[]
            assert result['confidence']=={} and result['message_id']==''
            assert result['model_calls']==result['tool_calls']==0
            assert result['usage']=={'input_tokens':7,'output_tokens':3,'estimated':0}
            assert not model.selected and not retrieval.calls
            assert len(result['messages'])==6 and len({m.id for m in result['messages']})==6
            assert [m.content for m in result['messages'] if isinstance(m,HumanMessage)]==['我要投诉','饮水机容量','你好']
        async with api.open_workflow_checkpointer(path) as saver:
            business,model,*_=build(tmp_path,model=Model([selected('next-read'),AIMessage(content='draft')]),checkpointer=saver)
            result=await business.ainvoke(turn('查询订单','t4'),config)
            assert len(result['messages'])==10 and len({m.id for m in result['messages']})==10
            assert len([m for m in result['messages'] if isinstance(m,ToolMessage)])==1
            assert result['model_calls']==3 and result['tool_calls']==1
            assert not result['evidence'] and not result['suggestions']
            assert not any('完整原文证据' in str(m.content) for m in model.finals[0])
    asyncio.run(scenario())
