"""Explicit user-confirmed local refund applications, with fresh reason review."""
import asyncio
from app.ch06_types import Eligibility
from app.workflow_types import Usage
from app.aftersales_workflow import budget_reason
from app.agent_runtime import _estimate
from langchain_core.messages import AIMessage
import json


class RefundService:
    def __init__(self,repository,business,model,limits,fingerprints):
        self.repository,self.business,self.model,self.limits,self.fingerprints=repository,business,model,limits,fingerprints

    def versions(self):return {**self.fingerprints(),'business':self.business.load().digest}

    def prepare_action(self,conversation_id,turn_id,item,eligibility,evidence,*,query='',facts=None,model_calls=0,usage=None):
        if eligibility.status!='eligible' or not evidence:raise ValueError('Refund is not eligible')
        versions=self.versions()
        context={'query':query,'facts':facts or {},'evidence':evidence,'model_calls':model_calls,'usage':usage or {},'reviews':{}}
        row=self.repository.issue_action(conversation_id,turn_id,item,eligibility,versions,context=context)
        return {'action_id':row['action_id'],'kind':'refund','status':row['status'],'item':row['item'],'reasons':self.business.refund_reasons()}

    def load_forms(self,conversation_id):
        versions=self.versions();reasons=self.business.refund_reasons()
        result={}
        for message_id,row in self.repository.load_forms(conversation_id).items():
            status=row['status'] if row['fingerprints']==versions else 'stale'
            result[message_id]={'action_id':row['action_id'],'kind':'refund','status':status,'item':row['item'],'reasons':reasons,'receipt':row.get('receipt')}
        return result

    async def submit(self,conversation_id,action_id,reason_code,request_id):
        existing=await asyncio.to_thread(self.repository.find_request,conversation_id,action_id,reason_code,request_id)
        if existing is not None:return existing
        action=await asyncio.to_thread(self.repository.get_action,conversation_id,action_id)
        reasons={reason['code']:reason['label'] for reason in await asyncio.to_thread(self.business.refund_reasons)}
        if reason_code not in reasons:raise ValueError('Invalid refund reason')
        versions=await asyncio.to_thread(self.versions)
        if action['status']!='active' or action['fingerprints']!=versions:raise ValueError('Refund action is stale')
        row=action['item']
        item=await asyncio.to_thread(self.business.get_item,row['customer_id'],row['order_id'],row['item_id'])
        if item.to_dict()!=row or item.issues or any(token in item.order_status+item.aftersales_status for token in ('已退款','退款中','已退货')):raise ValueError('Order cannot be refunded')
        context=action['eligibility']['context']
        facts={**context['facts'],'intent':'退款退货','selected_refund_reason':reasons[reason_code]}
        estimate=_estimate([AIMessage(content=json.dumps([context['query'],item.to_dict(),context['evidence'],facts],ensure_ascii=False))])+200
        if request_id not in context.get('reviews',{}) and budget_reason(context,self.limits,estimate,self.limits.response_tokens):raise ValueError('Refund review budget exhausted')
        review=await asyncio.to_thread(self.repository.reserve_review,conversation_id,action_id,request_id,reason_code,self.limits)
        if review['claimed']:
            try:
                decision,usage=await self.model.judge_eligibility(context['query'],item,context['evidence'],facts,max_tokens=self.limits.response_tokens)
                allowed={str(e.get('id',e.get('chunk_id'))) for e in context['evidence']}
                if not isinstance(decision,Eligibility) or not set(decision.evidence_ids)<=allowed:raise ValueError('Invalid review evidence')
            except Exception as error:
                usage=getattr(error,'usage',Usage(estimated=estimate+self.limits.response_tokens))
                await asyncio.to_thread(self.repository.finish_review,conversation_id,action_id,request_id,None,usage)
                raise ValueError('Refund requires reassessment') from None
            if Usage(**context.get('usage',{})).total+usage.total>self.limits.turn_tokens:decision=None
            await asyncio.to_thread(self.repository.finish_review,conversation_id,action_id,request_id,decision.model_dump() if decision else None,usage)
        else:
            if review['status']!='complete':raise ValueError('Refund review is processing or requires reassessment')
            decision=Eligibility.model_validate(review['decision'])
        if decision is None or decision.status!='eligible' or decision.missing_facts or not decision.evidence_ids:raise ValueError('Refund requires reassessment')
        if await asyncio.to_thread(self.versions)!=versions:raise ValueError('Refund data changed during review')
        return await asyncio.to_thread(self.repository.create_request,conversation_id,action_id,reason_code,request_id,item,versions)
