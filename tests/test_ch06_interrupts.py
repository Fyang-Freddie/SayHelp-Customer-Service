import asyncio
from langgraph.types import Command
from app.workflow_graph import open_workflow_checkpointer
from test_aftersales_workflow import graph,inputs


def test_sqlite_reopen_resumes_same_wait_without_repeating_paid_nodes(tmp_path):
    async def run():
        events=[];path=tmp_path/'state.sqlite';config={'configurable':{'thread_id':'1'}}
        state=inputs();state['confirmed']={}
        async with open_workflow_checkpointer(path) as saver:
            result=await graph(saver,events).ainvoke(state,config)
            assert result['__interrupt__'][0].value['kind']=='customer_input'
            assert events==['ask_customer']
        async with open_workflow_checkpointer(path) as saver:
            compiled=graph(saver,events)
            assert not (await compiled.aget_state({'configurable':{'thread_id':'2'}})).values
            result=await compiled.ainvoke(Command(resume={'customer_id':'C1'}),config)
            assert result['__interrupt__'][0].value['kind']=='order_selection'
            assert events==['ask_customer']
            final=await compiled.ainvoke(Command(resume={'customer_id':'C1','order_id':'O1','item_id':'I1','data_digest':'data-v1'}),config)
            assert events==['ask_customer','order','expand','retrieve','judge']
            assert final['model_calls']==4
    asyncio.run(run())


def test_outer_resolution_precedes_classifier_and_resume_does_not_reclassify(tmp_path):
    from app.workflow_graph import build_workflow
    from app.ch06_types import Resolution,IntentDecision
    from app.workflow_types import Usage
    from app.agent_runtime import AgentLimits
    from types import SimpleNamespace
    from test_aftersales_workflow import Business,Expander,Policy,Model,emit
    from app.aftersales_workflow import build_aftersales_subgraph
    async def run():
        events=[];logs=[];config={'configurable':{'thread_id':'1'}}
        class Resolver:
            async def resolve(self,raw,history,confirmed):
                events.append('resolve');return Resolution(query='耳机能退吗',changed=True,unresolved_references=[]),Usage(1,1)
        async def classify(query):
            assert query=='耳机能退吗';events.append('jev');return IntentDecision(intent='退款退货',confidence=1.0),Usage(2,2)
        model=Model(events)
        child=build_aftersales_subgraph(business=Business(events),expander=Expander(events),policy_retrieval=Policy(events),model=model,limits=AgentLimits(),emit=emit)
        async with open_workflow_checkpointer(tmp_path/'outer.sqlite') as saver:
            compiled=build_workflow(model=model,classifier=classify,resolver=Resolver(),aftersales=child,knowledge_gate=None,tools_factory=lambda cid:{},limits=AgentLimits(),log=SimpleNamespace(write=lambda event,payload:logs.append(event)),checkpointer=saver)
            state=inputs();state['raw_question']='这个能退吗';state['messages']=[]
            first=await compiled.ainvoke(state,config)
            assert first['__interrupt__'] and events==['resolve','jev']
            final=await compiled.ainvoke(Command(resume={'customer_id':'C1','order_id':'O1','item_id':'I1','data_digest':'data-v1'}),config)
            assert events==['resolve','jev','order','expand','retrieve','judge']
            assert final['intent_decision']=={'intent':'退款退货','confidence':1.0}
            assert 'node_error' not in logs and final['jev_calls']==1
    asyncio.run(run())


def test_explicit_classifier_failure_never_routes_business_decision(tmp_path):
    from app.intent import IntentClassificationError,IntentDecision
    from app.workflow_types import Usage
    from test_workflow_graph import build,collect,turn
    async def classifier(query):raise IntentClassificationError(Usage(),'low_confidence',IntentDecision(intent='订单',confidence=.2))
    compiled,model,_,_,attempts,_=build(tmp_path,classifier=classifier)
    result,_,_=asyncio.run(collect(compiled,turn()))
    assert result['route']=='fallback' and not attempts and not model.selected
