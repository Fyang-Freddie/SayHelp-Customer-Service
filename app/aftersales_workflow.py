"""Replay-safe core flow: exact item, expanded policy retrieval, then Agent judgment."""
import asyncio
import json
from langchain_core.messages import AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt
from app.workflow_types import WorkflowState, Usage
from app.agent_runtime import _estimate
from app.query_expansion import QueryExpansionError


def total_usage(state,usage):
    old=Usage(**state.get('usage',{}))
    return Usage(old.input_tokens+usage.input_tokens,old.output_tokens+usage.output_tokens,old.estimated+usage.estimated).to_dict()


def budget_reason(state,limits,input_tokens,output_tokens):
    if state.get('model_calls',0)>=limits.model_calls:return 'model_budget'
    if input_tokens>limits.input_tokens:return 'context_budget'
    if Usage(**state.get('usage',{})).total+input_tokens+output_tokens>limits.turn_tokens:return 'token_budget'
    return None


def stopped(reason):
    return {'status':'fallback','stop_reason':reason,'eligibility':{}}


def build_aftersales_subgraph(*,business,expander,policy_retrieval,model,limits,emit):
    async def paid(state,phase,inputs,operation,output_tokens):
        estimate=_estimate([AIMessage(content=json.dumps(inputs,ensure_ascii=False))])+200
        reason=budget_reason(state,limits,estimate,output_tokens)
        if reason:return None,stopped(reason)
        await emit('model_start',{'phase':phase,'model_calls':state.get('model_calls',0)+1})
        error=None
        try:value,usage=await operation()
        except Exception as exception:
            value=None;usage=getattr(exception,'usage',Usage(estimated=estimate+output_tokens));error=getattr(exception,'reason',phase+'_error')
        fields={'model_calls':state.get('model_calls',0)+1,'usage':total_usage(state,usage)}
        await emit('model_end',{'phase':phase,**fields})
        if error:fields.update(stopped(error))
        elif Usage(**fields['usage']).total>limits.turn_tokens:fields.update(stopped('token_budget'));value=None
        return value,fields

    async def prepare(state):
        return {'status':'order' if state.get('selected_order') else 'choose' if state.get('confirmed',{}).get('customer_id') else 'customer','eligibility':{}}

    async def ask_customer(state):
        question,fields=await paid(state,'ask_customer',state['resolved_question'],lambda:model.ask_customer(state['resolved_question'],max_tokens=limits.response_tokens),limits.response_tokens)
        if question is not None:fields.update(clarification=question,status='waiting_customer')
        return fields

    async def wait_customer(state):
        payload={'kind':'customer_input','message':state['clarification'],'turn_id':state['turn_id']}
        while True:
            value=interrupt(payload)
            customer=value.get('customer_id') if isinstance(value,dict) else None
            if isinstance(customer,str) and customer.strip() and len(customer)<=64:
                items=await asyncio.to_thread(business.list_items,customer)
                if items:return {'confirmed':{'customer_id':customer},'selected_order':{},'status':'choose'}
            payload={**payload,'error':'客户ID无效或没有可选订单，请核对。'}

    async def wait_order(state):
        customer=state['confirmed']['customer_id']
        while True:
            snapshot=await asyncio.to_thread(business.load)
            items=await asyncio.to_thread(business.list_items,customer)
            payload={'kind':'order_selection','turn_id':state['turn_id'],'message':'请选择需要处理的订单商品。','customer_id':customer,'data_digest':snapshot.digest,'orders':[item.to_dict() for item in items]}
            selected=interrupt(payload)
            if isinstance(selected,dict) and selected.get('customer_id')==customer and selected.get('data_digest')==snapshot.digest:
                match=next((item for item in items if item.order_id==selected.get('order_id') and item.item_id==selected.get('item_id')),None)
                if match:return {'selected_order':{k:selected[k] for k in ('customer_id','order_id','item_id','data_digest')},'status':'order'}

    async def order(state):
        selected=state.get('selected_order',{});customer=state.get('confirmed',{}).get('customer_id')
        if not customer or selected.get('customer_id')!=customer:return stopped('order_scope_mismatch')
        try:
            digest=(await asyncio.to_thread(business.load)).digest
            if selected.get('data_digest')!=digest:return stopped('order_data_changed')
            item=await asyncio.to_thread(business.get_item,customer,selected['order_id'],selected['item_id'])
            if item.issues:return stopped('order_data_inconsistent')
            return {'order_item':item.to_dict(),'data_digest':digest,'confirmed':{**state.get('confirmed',{}),'order_id':item.order_id,'item_id':item.item_id,'product_name':item.product_name,'variant':item.variant},'status':'expand'}
        except (KeyError,ValueError):return stopped('order_unavailable')

    async def item_for(state):
        selected=state['selected_order']
        if (await asyncio.to_thread(business.load)).digest!=state['data_digest']:raise ValueError('Order source changed')
        # Revalidate source snapshot without repeating externally visible order fetch.
        items=await asyncio.to_thread(business.list_items,selected['customer_id'])
        item=next((i for i in items if i.order_id==selected['order_id'] and i.item_id==selected['item_id']),None)
        if item is None or item.to_dict()!=state['order_item']:raise ValueError('Order source changed')
        return item

    async def expand(state):
        try:item=await item_for(state)
        except ValueError:return stopped('order_data_changed')
        result,fields=await paid(state,'expand',[state['resolved_question'],item.to_dict()],lambda:expander.expand(state['resolved_question'],item),1024)
        if result is not None:fields.update(queries=result.queries,status='retrieve')
        return fields

    async def retrieve(state):
        try:item=await item_for(state)
        except ValueError:return stopped('order_data_changed')
        result=await policy_retrieval.prepare(state['resolved_question'],state['queries'],item)
        return {**result,'status':'judge' if result.get('knowledge_confidence',{}).get('sufficient') and result.get('evidence') and result.get('citations') else 'fallback'}

    async def judge(state):
        try:item=await item_for(state)
        except ValueError:return stopped('order_data_changed')
        facts={**state.get('facts',{}),'intent':state['intent']}
        decision,fields=await paid(state,'eligibility',[state['resolved_question'],item.to_dict(),state['evidence'],facts],lambda:model.judge_eligibility(state['resolved_question'],item,state['evidence'],facts,max_tokens=limits.response_tokens),limits.response_tokens)
        if decision is None:return fields
        allowed={str(c.get('chunk_id')) for c in state['citations']}
        if not set(decision.evidence_ids)<=allowed or (decision.status=='eligible' and (decision.missing_facts or not decision.evidence_ids)):
            return {**fields,**stopped('eligibility_invalid_evidence')}
        fields['eligibility']=decision.model_dump()
        if decision.status=='needs_information':
            fields.update(clarification=decision.explanation,status='waiting_clarification')
        else:
            fields.update(answer=decision.explanation,messages=[AIMessage(content=decision.explanation,id=state['turn_id']+':assistant')],status='done',stop_reason='complete')
        return fields

    async def wait_clarification(state):
        while True:
            value=interrupt({'kind':'clarification','turn_id':state['turn_id'],'message':state['clarification']})
            text=value.get('text') if isinstance(value,dict) else None
            if isinstance(text,str) and text.strip() and len(text)<=2000:
                return {'facts':{**state.get('facts',{}),'customer_statement':text},'status':'judge'}

    graph=StateGraph(WorkflowState)
    for name,node in [('prepare',prepare),('ask_customer',ask_customer),('wait_customer',wait_customer),('wait_order',wait_order),('order',order),('expand',expand),('retrieve',retrieve),('judge',judge),('wait_clarification',wait_clarification)]:graph.add_node(name,node)
    graph.add_edge(START,'prepare')
    graph.add_conditional_edges('prepare',lambda s:s['status'],{'customer':'ask_customer','choose':'wait_order','order':'order'})
    graph.add_conditional_edges('ask_customer',lambda s:s['status'],{'waiting_customer':'wait_customer','fallback':END})
    graph.add_edge('wait_customer','wait_order');graph.add_edge('wait_order','order')
    graph.add_conditional_edges('order',lambda s:s['status'],{'expand':'expand','fallback':END})
    graph.add_conditional_edges('expand',lambda s:s['status'],{'retrieve':'retrieve','fallback':END})
    graph.add_conditional_edges('retrieve',lambda s:s['status'],{'judge':'judge','fallback':END})
    graph.add_conditional_edges('judge',lambda s:s['status'],{'done':END,'waiting_clarification':'wait_clarification','fallback':END})
    graph.add_edge('wait_clarification','judge')
    return graph.compile()
