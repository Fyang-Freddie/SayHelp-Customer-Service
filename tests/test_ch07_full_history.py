"""Full checkpoint history must survive the model's smaller window."""
import asyncio
from types import SimpleNamespace
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from test_workflow_service import graph, service, Model, collect
from test_workflow_repository import sessions
from test_db import database_url
from app.repository import Repository
from app.workflow_graph import open_workflow_checkpointer


def test_full_history_survives_24_turns_and_reopen(sessions,tmp_path):
    cid=Repository(sessions).create_conversation('history')
    path=tmp_path/'history.sqlite'
    async def run():
        async with open_workflow_checkpointer(path) as saver:
            compiled=graph(saver,Model(),intent='闲聊')
            svc=service(sessions,compiled)
            for i in range(24): await collect(svc,cid,f'问题{i}')
        async with open_workflow_checkpointer(path) as saver:
            compiled=graph(saver,Model(),intent='闲聊')
            snapshot=await compiled.aget_state({'configurable':{'thread_id':str(cid)}})
            messages=snapshot.values['messages']
            assert [m.content for m in messages if isinstance(m,HumanMessage)]==[f'问题{i}' for i in range(24)]
    asyncio.run(run())


def test_legacy_partial_checkpoint_imports_only_missing_messages():
    from app.context_history import reconcile_history
    rows=[SimpleNamespace(id=i,role='user' if i%2 else 'assistant',content=str(i),tool_calls=None)
          for i in range(1,5)]
    records=reconcile_history(rows,[HumanMessage('3',id='3'),AIMessage('4',id='reply')],[])
    assert [t.first_msg_id for t in records]==[1,3]
    assert [m.content for t in records for m in t.messages]==['1','2','3','4']


def test_reply_reconciliation_after_mysql_failure_preserves_tool_trace(sessions,tmp_path):
    from app.workflow_service import WorkflowService
    cid=Repository(sessions).create_conversation('tool-history')
    async def run():
        async with open_workflow_checkpointer(tmp_path/'tool.sqlite') as saver:
            svc=service(sessions,graph(saver,Model()))
            uid=svc.workflow_repository.append_message_once(cid,'turn',0,'user','查询订单')
            result={'messages':[HumanMessage('查询订单',id=uid),
                AIMessage('',tool_calls=[dict(id='call',name='query_order',args={})]),
                ToolMessage('大块结果'*1000,tool_call_id='call'),AIMessage('订单待核实')],
                'answer':'订单待核实'}
            first=await svc._persist_reply(cid,'turn',uid,result)
            second=await svc._persist_reply(cid,'turn',uid,result)
            assert first==second
            rows=svc.repository.load_messages(cid)
            assert [r.role for r in rows]==['user','assistant']
            assert result['messages'][2].content=='大块结果'*1000
    asyncio.run(run())
