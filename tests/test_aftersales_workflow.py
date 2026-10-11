"""Deterministic order -> expansion -> evidence -> Agent, with real graph pauses."""
import asyncio, importlib
from dataclasses import replace
from decimal import Decimal
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
from app.workflow_types import WorkflowState, Usage
from app.workflow_graph import open_workflow_checkpointer
from app.agent_runtime import AgentLimits
from app.business_data import OrderItem
from app.ch06_types import Expansion, Eligibility

ITEM=OrderItem('C1','O1','I1','耳机','黑色',1,Decimal('99.00'),None,None,'已付款','无')
class Business:
    def __init__(self,events):self.events=events
    def list_items(self,customer):return [ITEM] if customer=='C1' else []
    def get_item(self,customer,order,item):
        self.events.append('order')
        if (customer,order,item)!=('C1','O1','I1'):raise KeyError('not found')
        return ITEM
    def load(self):
        from types import SimpleNamespace
        return SimpleNamespace(digest='data-v1')
class Expander:
    def __init__(self,events):self.events=events
    async def expand(self,query,item):
        self.events.append('expand');return Expansion(queries=['退货资格','退货时限','退货运费']),Usage(10,10)
class Policy:
    def __init__(self,events,sufficient=True):self.events,self.sufficient=events,sufficient
    async def prepare(self,query,queries,item):
        self.events.append('retrieve')
        return {'evidence':[{'id':1,'answer':'未拆封可退'}],'citations':[{'chunk_id':'1'}],'knowledge_confidence':{'sufficient':self.sufficient},'stop_reason':None}
class Model:
    def __init__(self,events):self.events=events
    async def ask_customer(self,query,**kwargs):self.events.append('ask_customer');return '请提供客户ID。',Usage(5,5)
    async def judge_eligibility(self,query,item,evidence,facts,**kwargs):
        self.events.append('judge');return Eligibility(status='eligible',explanation='根据条款可申请退款。',evidence_ids=['1'],missing_facts=[]),Usage(15,15)
async def emit(*args):pass

def graph(saver,events,sufficient=True):
    build=importlib.import_module('app.aftersales_workflow').build_aftersales_subgraph
    child=build(business=Business(events),expander=Expander(events),policy_retrieval=Policy(events,sufficient),model=Model(events),limits=AgentLimits(),emit=emit)
    parent=StateGraph(WorkflowState);parent.add_node('aftersales',child);parent.add_edge(START,'aftersales');parent.add_edge('aftersales',END)
    return parent.compile(checkpointer=saver)

def inputs(**values):return dict(conversation_id=1,turn_id='turn',raw_question='耳机能退吗',resolved_question='耳机能退吗',intent='退款退货',model_calls=1,usage=Usage(1,1).to_dict(),confirmed={'customer_id':'C1'},**values)

def test_order_interrupt_then_exact_deterministic_path(tmp_path):
    async def run():
        events=[];config={'configurable':{'thread_id':'1'}}
        async with open_workflow_checkpointer(tmp_path/'state.sqlite') as saver:
            compiled=graph(saver,events)
            first=await compiled.ainvoke(inputs(),config)
            assert first['__interrupt__'][0].value['kind']=='order_selection'
            assert events==[]
            result=await compiled.ainvoke(Command(resume={'customer_id':'C1','order_id':'O1','item_id':'I1','data_digest':'data-v1'}),config)
            assert events==['order','expand','retrieve','judge']
            assert result['eligibility']['status']=='eligible'
            assert result['model_calls']==3 and result['usage']['input_tokens']==26
    asyncio.run(run())

def test_weak_policy_never_calls_eligibility_agent(tmp_path):
    async def run():
        events=[]
        async with open_workflow_checkpointer(tmp_path/'state.sqlite') as saver:
            result=await graph(saver,events,False).ainvoke(inputs(selected_order={'customer_id':'C1','order_id':'O1','item_id':'I1','data_digest':'data-v1'}),{'configurable':{'thread_id':'1'}})
            assert events==['order','expand','retrieve']
            assert result['status']=='fallback' and not result.get('eligibility')
    asyncio.run(run())


def test_invalid_card_selection_keeps_waiting_without_paid_calls(tmp_path):
    async def run():
        events=[];config={'configurable':{'thread_id':'1'}}
        async with open_workflow_checkpointer(tmp_path/'state.sqlite') as saver:
            compiled=graph(saver,events)
            await compiled.ainvoke(inputs(),config)
            result=await compiled.ainvoke(Command(resume={'customer_id':'C2','order_id':'O1','item_id':'I1','data_digest':'data-v1'}),config)
            assert result['__interrupt__'] and events==[]
            result=await compiled.ainvoke(Command(resume={'customer_id':'C1','order_id':'O1','item_id':'I1','data_digest':'data-v1'}),config)
            assert result['eligibility']['status']=='eligible'
    asyncio.run(run())


def test_budget_survives_resume_and_blocks_expansion(tmp_path):
    async def run():
        events=[];config={'configurable':{'thread_id':'1'}};state=inputs();state['model_calls']=6
        async with open_workflow_checkpointer(tmp_path/'state.sqlite') as saver:
            compiled=graph(saver,events)
            await compiled.ainvoke(state,config)
            result=await compiled.ainvoke(Command(resume={'customer_id':'C1','order_id':'O1','item_id':'I1','data_digest':'data-v1'}),config)
            assert result['stop_reason']=='model_budget' and events==['order']
            assert result['model_calls']==6
    asyncio.run(run())


def test_stale_source_selection_cannot_load_order(tmp_path):
    async def run():
        events=[]
        async with open_workflow_checkpointer(tmp_path/'state.sqlite') as saver:
            state=inputs(selected_order={'customer_id':'C1','order_id':'O1','item_id':'I1','data_digest':'stale'})
            result=await graph(saver,events).ainvoke(state,{'configurable':{'thread_id':'1'}})
            assert result['stop_reason']=='order_data_changed' and events==[]
    asyncio.run(run())
