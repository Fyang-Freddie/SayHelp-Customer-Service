"""Twenty-four real graph/MySQL/checkpoint turns with oversized tool output."""
import asyncio
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.agent_runtime import AgentLimits
from app.config import Settings
from app.context_log import ContextLog
from app.context_repository import ContextRepository
from app.context_runtime import ContextRuntime
from app.init_ch07_db import initialize_ch07_database
from app.intent import IntentDecision
from app.repository import Repository
from app.workflow_graph import build_workflow,open_workflow_checkpointer
from app.workflow_log import WorkflowLog
from app.workflow_repository import WorkflowRepository
from app.workflow_service import WorkflowService
from app.workflow_types import Usage
from test_db import database_url
from test_workflow_repository import sessions


def test_24_tool_rich_turns_stay_bounded_and_keep_checkpoint_trace(sessions,database_url,tmp_path):
    initialize_ch07_database(database_url)
    repo=Repository(sessions)
    cid=repo.create_conversation('tool-rich-24')

    @tool
    def query_order() -> str:
        """Return an intentionally oversized synthetic order record."""
        return '模拟订单大结果'*1000

    class Model:
        calls=0
        async def select(self,messages,tools,**kwargs):
            self.calls+=1
            return AIMessage('',tool_calls=[{'id':f'call-{self.calls}',
                'name':'query_order','args':{}}])
        async def stream_reply(self,messages,**kwargs):
            yield AIMessageChunk(content='已记录，订单状态仍待核实。')
        async def resolve_query(self,messages):
            import json
            raw=json.loads(messages[-1].content)['raw']
            return AIMessage(content=json.dumps({'query':raw,'changed':False,
                'unresolved_references':[]},ensure_ascii=False))

    async def classify(question):
        return IntentDecision(intent='订单',confidence=1.0),Usage(input_tokens=1,output_tokens=1)

    async def run():
        settings=Settings('https://example.invalid/v1','test','test',layered_context=True,
            context_token_budget=18000,response_token_reserve=2000,max_user_input_tokens=2000,
            agent_turn_tokens=100000)
        model=Model()
        runtime=ContextRuntime(repo,WorkflowRepository(sessions),ContextRepository(sessions),None,
            settings,ContextLog(tmp_path/'app.log'))
        async with open_workflow_checkpointer(tmp_path/'tool-rich.sqlite') as saver:
            graph=build_workflow(model=model,classifier=classify,knowledge_gate=None,
                tools_factory=lambda cid:{'query_order':query_order},
                limits=AgentLimits(input_tokens=16000,turn_tokens=100000,
                    response_tokens=2000,tool_result_tokens=1200,max_agent_steps=3),
                log=WorkflowLog(tmp_path/'workflow'),checkpointer=saver,
                context_runtime=runtime)
            service=WorkflowService(repo,WorkflowRepository(sessions),graph,settings,
                context_runtime=runtime)
            for i in range(24):
                events=[e async for e in service.stream_turn(cid,f'第{i+1}轮查询订单 TB202611000001')]
                assert events[-1].kind=='completed'
            snapshot=await graph.aget_state({'configurable':{'thread_id':str(cid)}})
            full=snapshot.values['messages']
            assert len([m for m in full if isinstance(m,HumanMessage)])==24
            assert len([m for m in full if isinstance(m,ToolMessage)])==24
            assert all('模拟订单大结果' not in m.content for m in full if isinstance(m,ToolMessage))
            from app.context_budget import estimate_tokens
            assert all(estimate_tokens([m])<=1200 for m in full if isinstance(m,ToolMessage))
            assert len(repo.load_messages(cid))==48
            assert all(r.role in {'user','assistant'} for r in repo.load_messages(cid))
            assert model.calls==24
        import json
        logs=[json.loads(line) for line in (tmp_path/'app.log').read_text(encoding='utf8').splitlines()]
        assert len([row for row in logs if row['event']=='history_ctx'])==24
        assert len([row for row in logs if row['event']=='model_ctx' and row['phase']=='select'])==24
    asyncio.run(run())
