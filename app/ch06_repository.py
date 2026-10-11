"""MySQL interaction records mirror checkpoints; claiming never executes a graph."""
import hashlib
import json
import re
from uuid import NAMESPACE_URL, uuid5
from sqlalchemy import select
from app.ch05_db import WorkflowTurn
from app.ch06_db import WorkflowInteraction, WorkflowResumeRequest, RefundAction
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
    def issue_action(self,conversation_id,turn_id,item,eligibility,fingerprints):
        from app.business_data import OrderItem
        from app.ch06_types import Eligibility
        _identity(turn_id,0)
        if not isinstance(item,OrderItem) or not isinstance(eligibility,Eligibility):raise ValueError('Invalid eligibility action')
        if item.issues or eligibility.status!='eligible' or eligibility.missing_facts or not eligibility.evidence_ids:raise ValueError('Eligibility is not established')
        if not isinstance(fingerprints,dict) or not fingerprints or any(not isinstance(v,str) or not v for v in fingerprints.values()):raise ValueError('Missing source fingerprints')
        data=_json(item.to_dict());decision=eligibility.model_dump();versions=_json(fingerprints)
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
