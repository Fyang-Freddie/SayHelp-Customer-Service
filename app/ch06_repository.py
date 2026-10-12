"""MySQL interaction records mirror checkpoints; claiming never executes a graph."""
import hashlib
import json
import re
from uuid import NAMESPACE_URL, uuid5
from sqlalchemy import select, text
from app.ch05_db import WorkflowTurn
from app.ch06_db import WorkflowInteraction, WorkflowResumeRequest, RefundAction, RefundRequest
from app.workflow_repository import WorkflowRepository, _active_conversation, _identity


def _opaque(value,limit=64):
    if not isinstance(value,str) or not value.strip() or len(value)>limit:
        raise ValueError('Invalid identity')


def _json(value):
    return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True))


def _interaction(row):
    return {'interrupt_id':row.interrupt_id,'conversation_id':str(row.conversation_id),
            'turn_id':row.turn_id,'message_id':str(row.message_id),'kind':row.kind,
            'payload':row.payload,'checkpoint_id':row.checkpoint_id,'status':row.status}


def _resume(row,claimed=False):
    return {'request_id':row.request_id,'status':row.status,'result':row.result,'claimed':claimed}


class InteractionRepository(WorkflowRepository):
    def publish_wait(self,conversation_id,turn_id,interrupt_id,kind,payload,checkpoint_id):
        _identity(turn_id,0);_opaque(interrupt_id);_opaque(checkpoint_id)
        if kind not in ('order_selection','customer_input','clarification') or not isinstance(payload,dict): raise ValueError('Invalid interaction')
        payload=_json(payload)
        message=payload.get('message','请选择需要处理的订单商品。')
        if not isinstance(message,str) or not message.strip(): raise ValueError('Invalid interaction message')
        def operation(s):
            turn=self._turn(s,conversation_id,turn_id)
            row=s.get(WorkflowInteraction,interrupt_id)
            if row is not None:
                if (row.conversation_id,row.turn_id,row.kind,row.payload,row.checkpoint_id)!=(conversation_id,turn_id,kind,payload,checkpoint_id):
                    raise ValueError('Interaction identity conflict')
                return _interaction(row)
            if turn.status not in ('running','waiting'): raise ValueError('Turn cannot wait')
            active=s.scalar(select(WorkflowInteraction).where(WorkflowInteraction.conversation_id==conversation_id,WorkflowInteraction.status.in_(['waiting','processing'])))
            if active is not None: raise ValueError('Another interaction is pending')
            # Reserved upper-half position is stable on replay and outside agent positions.
            position=2**63 + int.from_bytes(hashlib.sha256(interrupt_id.encode()).digest()[:8],'big') % 2**63
            display,_=self._message(s,conversation_id,turn_id,position,'assistant',message,None,None)
            row=WorkflowInteraction(interrupt_id=interrupt_id,conversation_id=conversation_id,turn_id=turn_id,message_id=display.id,kind=kind,payload=payload,checkpoint_id=checkpoint_id,status='waiting')
            s.add(row);turn.status='waiting';s.flush()
            return _interaction(row)
        return self._write(operation)

    def claim_resume(self,conversation_id,turn_id,interrupt_id,request_id,payload_digest):
        _identity(turn_id,0);_opaque(interrupt_id);_opaque(request_id,128)
        if not isinstance(payload_digest,str) or not re.fullmatch('[0-9a-f]{64}',payload_digest):raise ValueError('Invalid payload digest')
        def operation(s):
            _active_conversation(s,conversation_id)
            row=s.get(WorkflowInteraction,interrupt_id,with_for_update=True)
            if row is None or (row.conversation_id,row.turn_id)!=(conversation_id,turn_id):raise KeyError('Interaction not found')
            existing=s.get(WorkflowResumeRequest,request_id)
            if existing is not None:
                if (existing.conversation_id,existing.turn_id,existing.interrupt_id,existing.payload_digest)!=(conversation_id,turn_id,interrupt_id,payload_digest):raise ValueError('Resume identity conflict')
                return _resume(existing)
            if row.status!='waiting':raise ValueError('Interaction is not waiting')
            request=WorkflowResumeRequest(request_id=request_id,conversation_id=conversation_id,turn_id=turn_id,interrupt_id=interrupt_id,payload_digest=payload_digest,status='processing')
            s.add(request);row.status='processing';s.flush()
            return _resume(request,True)
        return self._write(operation)

    def get_resume(self,conversation_id,turn_id,interrupt_id,request_id,payload_digest):
        with self.session_factory.begin() as s:
            _active_conversation(s,conversation_id)
            row=s.get(WorkflowResumeRequest,request_id)
            if row is None:return None
            if (row.conversation_id,row.turn_id,row.interrupt_id,row.payload_digest)!=(conversation_id,turn_id,interrupt_id,payload_digest):raise ValueError('Resume identity conflict')
            return _resume(row)

    def processing_request(self,conversation_id,interrupt_id):
        with self.session_factory.begin() as s:
            _active_conversation(s,conversation_id)
            row=s.scalar(select(WorkflowResumeRequest).where(WorkflowResumeRequest.conversation_id==conversation_id,WorkflowResumeRequest.interrupt_id==interrupt_id,WorkflowResumeRequest.status=='processing'))
            return row.request_id if row else None

    def finish_resume(self,request_id,result):
        _opaque(request_id,128)
        if not isinstance(result,dict):raise ValueError('Invalid resume result')
        result=_json(result)
        # Read identity first, then lock conversation in the same order as claim/cancel.
        with self.session_factory() as s:
            request=s.get(WorkflowResumeRequest,request_id)
            if request is None:raise KeyError('Resume not found')
            cid=request.conversation_id
        def operation(s):
            _active_conversation(s,cid)
            request=s.get(WorkflowResumeRequest,request_id,with_for_update=True)
            if request.status=='complete':
                if request.result!=result:raise ValueError('Resume result conflict')
                return
            if request.status!='processing':raise ValueError('Resume was cancelled')
            interaction=s.get(WorkflowInteraction,request.interrupt_id,with_for_update=True)
            if interaction.status!='processing':raise ValueError('Interaction is not processing')
            request.result=result;request.status='complete';interaction.status='consumed'
        self._write(operation)

    def cancel_wait(self,conversation_id,interrupt_id):
        _opaque(interrupt_id)
        def operation(s):
            _active_conversation(s,conversation_id)
            row=s.get(WorkflowInteraction,interrupt_id,with_for_update=True)
            if row is None or row.conversation_id!=conversation_id:raise KeyError('Interaction not found')
            if row.status=='processing':raise ValueError('Resume is processing')
            if row.status=='consumed':raise ValueError('Interaction was consumed')
            row.status='cancelled'
            turn=s.get(WorkflowTurn,row.turn_id)
            turn.status='cancelled'
            for action in s.scalars(select(RefundAction).where(RefundAction.conversation_id==conversation_id,RefundAction.turn_id==row.turn_id,RefundAction.status=='active')):action.status='cancelled'
        self._write(operation)

    def pending(self,conversation_id):
        with self.session_factory.begin() as s:
            _active_conversation(s,conversation_id)
            row=s.scalar(select(WorkflowInteraction).where(WorkflowInteraction.conversation_id==conversation_id,WorkflowInteraction.status.in_(['waiting','processing'])))
            return _interaction(row) if row else None

    def load_interactions(self,conversation_id):
        with self.session_factory.begin() as s:
            _active_conversation(s,conversation_id)
            return {str(row.message_id):_interaction(row) for row in s.scalars(select(WorkflowInteraction).where(WorkflowInteraction.conversation_id==conversation_id))}


def _refund_action(row):
    return {'action_id':row.action_id,'turn_id':row.turn_id,'status':row.status,
            'item':row.item,'eligibility':row.eligibility,'fingerprints':row.fingerprints}


class RefundRepository(WorkflowRepository):
    def issue_action(self,conversation_id,turn_id,item,eligibility,fingerprints,*,context=None):
        from app.business_data import OrderItem
        from app.ch06_types import Eligibility
        _identity(turn_id,0)
        if not isinstance(item,OrderItem) or not isinstance(eligibility,Eligibility):raise ValueError('Invalid eligibility action')
        if item.issues or eligibility.status!='eligible' or eligibility.missing_facts or not eligibility.evidence_ids:raise ValueError('Eligibility is not established')
        if not isinstance(fingerprints,dict) or not fingerprints or any(not isinstance(v,str) or not v for v in fingerprints.values()):raise ValueError('Missing source fingerprints')
        data=_json(item.to_dict());decision=eligibility.model_dump();versions=_json(fingerprints)
        if context is not None:decision['context']=_json(context)
        identity=json.dumps([conversation_id,turn_id,item.customer_id,item.order_id,item.item_id],ensure_ascii=False)
        action_id=uuid5(NAMESPACE_URL,'sayhelp-refund:'+identity).hex
        def operation(s):
            turn=self._turn(s,conversation_id,turn_id)
            if turn.status in ('cancelled','failed'):raise ValueError('Turn is no longer active')
            row=s.get(RefundAction,action_id)
            if row is not None:
                if (row.conversation_id,row.turn_id,row.item,row.eligibility,row.fingerprints)!=(conversation_id,turn_id,data,decision,versions):raise ValueError('Refund action payload conflict')
                return _refund_action(row)
            row=RefundAction(action_id=action_id,conversation_id=conversation_id,turn_id=turn_id,item=data,eligibility=decision,fingerprints=versions,status='active')
            s.add(row);s.flush()
            return _refund_action(row)
        return self._write(operation)

    def cancel_actions(self,conversation_id):
        def operation(s):
            _active_conversation(s,conversation_id)
            for row in s.scalars(select(RefundAction).where(RefundAction.conversation_id==conversation_id,RefundAction.status=='active')):row.status='cancelled'
        self._write(operation)


    def get_action(self,conversation_id,action_id):
        with self.session_factory.begin() as s:
            _active_conversation(s,conversation_id)
            row=s.get(RefundAction,action_id)
            if row is None or row.conversation_id!=conversation_id:raise KeyError('Refund action not found')
            return _refund_action(row)

    def load_forms(self,conversation_id):
        with self.session_factory.begin() as s:
            _active_conversation(s,conversation_id)
            pairs=s.execute(select(RefundAction,WorkflowTurn.final_message_id).join(WorkflowTurn,WorkflowTurn.turn_id==RefundAction.turn_id).where(RefundAction.conversation_id==conversation_id,WorkflowTurn.final_message_id.is_not(None))).all()
            receipts={row.action_id:{**_refund_request(row),'reason_code':row.reason_code} for row in s.scalars(select(RefundRequest).where(RefundRequest.conversation_id==conversation_id))}
            return {str(message_id):{**_refund_action(row),'receipt':receipts.get(row.action_id)} for row,message_id in pairs}

    def find_request(self,conversation_id,action_id,reason_code,request_id):
        with self.session_factory.begin() as s:
            _active_conversation(s,conversation_id)
            row=s.get(RefundRequest,request_id)
            if row is None:return None
            if (row.conversation_id,row.action_id,row.reason_code)!=(conversation_id,action_id,reason_code):raise ValueError('Refund request identity conflict')
            return _refund_request(row)

    def reserve_review(self,conversation_id,action_id,request_id,reason_code,limits):
        def operation(s):
            _active_conversation(s,conversation_id)
            row=s.get(RefundAction,action_id,with_for_update=True)
            if row is None or row.conversation_id!=conversation_id:raise KeyError('Refund action not found')
            if row.status!='active':raise ValueError('Refund action is not active')
            value=_json(row.eligibility);context=value.get('context',{});reviews=context.setdefault('reviews',{})
            prior=reviews.get(request_id)
            if prior is not None:
                if prior['reason_code']!=reason_code:raise ValueError('Refund review identity conflict')
                return {**prior,'claimed':False}
            if any(review['status']=='processing' for review in reviews.values()):raise ValueError('Refund review is processing')
            from app.workflow_types import Usage
            if context.get('model_calls',0)>=limits.model_calls or Usage(**context.get('usage',{})).total>=limits.turn_tokens:raise ValueError('Refund review budget exhausted')
            context['model_calls']=context.get('model_calls',0)+1
            reviews[request_id]={'reason_code':reason_code,'status':'processing'}
            value['context']=context;row.eligibility=value
            return {**reviews[request_id],'claimed':True}
        return self._write(operation)

    def finish_review(self,conversation_id,action_id,request_id,decision,usage):
        def operation(s):
            _active_conversation(s,conversation_id)
            row=s.get(RefundAction,action_id,with_for_update=True)
            if row is None or row.conversation_id!=conversation_id:raise KeyError('Refund action not found')
            value=_json(row.eligibility);context=value['context'];review=context['reviews'][request_id]
            if review['status']!='processing':raise ValueError('Refund review already ended')
            from app.aftersales_workflow import total_usage
            context['usage']=total_usage(context,usage)
            review.update(status='complete' if decision else 'failed',decision=decision)
            row.eligibility=value
        self._write(operation)

    def create_request(self,conversation_id,action_id,reason_code,request_id,item,expected_fingerprints):
        _opaque(request_id,128);_opaque(action_id);_opaque(reason_code)
        identity=json.dumps([item.customer_id,item.order_id,item.item_id],ensure_ascii=False,separators=(',',':'))
        lock_name='sh_refund:'+hashlib.sha256(identity.encode()).hexdigest()[:54]
        digest=hashlib.sha256(json.dumps([conversation_id,action_id,reason_code,item.to_dict()],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        def operation(s):
            _active_conversation(s,conversation_id)
            existing=s.get(RefundRequest,request_id)
            if existing is not None:
                if existing.payload_digest!=digest:raise ValueError('Refund request identity conflict')
                return _refund_request(existing)
            action=s.get(RefundAction,action_id,with_for_update=True)
            if action is None or action.conversation_id!=conversation_id:raise KeyError('Refund action not found')
            if action.status!='active' or action.item!=item.to_dict() or action.fingerprints!=expected_fingerprints:raise ValueError('Refund action is stale')
            review=action.eligibility.get('context',{}).get('reviews',{}).get(request_id,{})
            decision=review.get('decision') or {}
            if review.get('status')!='complete' or review.get('reason_code')!=reason_code or decision.get('status')!='eligible' or decision.get('missing_facts'):raise ValueError('Refund requires reassessment')
            if item.issues or item.quantity<1 or item.paid_amount<=0 or any(token in item.order_status+item.aftersales_status for token in ('已退款','退款中','已退货')):raise ValueError('Order cannot be refunded')
            duplicate=s.scalar(select(RefundRequest).where(RefundRequest.customer_id==item.customer_id,RefundRequest.order_id==item.order_id,RefundRequest.item_id==item.item_id))
            if duplicate is not None:raise ValueError('An active refund request already exists')
            row=RefundRequest(request_id=request_id,action_id=action_id,conversation_id=conversation_id,refund_no='RF'+uuid5(NAMESPACE_URL,request_id).hex,customer_id=item.customer_id,order_id=item.order_id,item_id=item.item_id,amount=item.paid_amount,quantity=item.quantity,reason_code=reason_code,payload_digest=digest,status='pending')
            s.add(row);action.status='submitted';s.flush()
            return _refund_request(row)
        # Named lock belongs to this connection, not the transaction; release explicitly.
        engine=self.session_factory.kw['bind']
        with engine.connect() as connection:
            locked=connection.execute(text('SELECT GET_LOCK(:name, 10)'),{'name':lock_name}).scalar()
            connection.commit()
            if locked!=1:raise TimeoutError('Refund item is busy')
            try:return self._write(operation)
            finally:
                try:
                    connection.rollback()
                    released=connection.execute(text('SELECT RELEASE_LOCK(:name)'),{'name':lock_name}).scalar()
                    connection.commit()
                    if released!=1:connection.invalidate()
                except BaseException:
                    connection.invalidate()
                    raise


def _refund_request(row):
    return {'refund_no':row.refund_no,'status':row.status,'amount':str(row.amount),'quantity':row.quantity,
            'message':'退款申请已提交，等待处理'}
