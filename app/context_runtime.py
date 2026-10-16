"""The sole bridge between durable history, model-only views and background summaries."""
import asyncio
from dataclasses import replace
import hashlib
import json
from langchain_core.messages import HumanMessage, SystemMessage
from app.context_budget import estimate_tokens, settings_budget, validate_request
from app.context_history import reconcile_history,build_context_view,compose_model_messages
from app.context_log import ContextLog
from app.prompts import render_workflow_system_prompt
from app.summary_prompts import summary_messages
from app.history import InputBudgetExceeded


class ContextRuntime:
    def __init__(self,repository,workflow_repository,context_repository,worker,settings,log=None,schemas=()):
        self.repository=repository
        self.workflow_repository=workflow_repository
        self.context_repository=context_repository
        self.worker=worker
        self.settings=settings
        self.log=log or ContextLog()
        self.schemas=list(schemas)
        self.snapshots={}

    async def recover(self,graph):
        for cid in await asyncio.to_thread(self.context_repository.expired_conversations):
            try:
                saved=await graph.aget_state({'configurable':{'thread_id':str(cid)}})
                if not saved.values.get('messages'):
                    self.log.write('summary skip',{'conversation_id':cid,'reason':'checkpoint_missing','duration_ms':0})
                    continue
                self.log.write('summary recovery',{'conversation_id':cid,'duration_ms':0})
                await self.history_view(saved.values)
            except Exception as error:
                self.log.write('summary failed',{'conversation_id':cid,'reason':'recovery_'+type(error).__name__,'duration_ms':0})

    async def history_view(self,state,*,evidence_tokens=None):
        cid,turn=state['conversation_id'],state['turn_id']
        current=next((m for m in reversed(state['messages']) if m.type=='human'),HumanMessage(state['raw_question']))
        if estimate_tokens([current])>self.settings.max_user_input_tokens:
            raise InputBudgetExceeded('上下文预算不足: 当前输入超过 MAX_USER_INPUT_TOKENS')
        key=(cid,turn)
        if key not in self.snapshots:
            self.snapshots={k:v for k,v in self.snapshots.items() if k[0]!=cid}
            self.snapshots[key]=await asyncio.to_thread(self.context_repository.load,cid)
        snapshot=self.snapshots[key]
        rows=await asyncio.to_thread(self.repository.load_messages,cid)
        excluded=await asyncio.to_thread(self.workflow_repository.incomplete_message_ids,cid) if self.workflow_repository else set()
        rows=[r for r in rows if str(r.id) not in excluded]
        turns=reconcile_history(rows,state.get('messages',[]),[])
        summary='\n'.join(s['content'] for s in snapshot.summaries)
        summary_tokens=estimate_tokens([HumanMessage(summary)]) if summary else 0
        ledger=settings_budget(self.settings,prefix=estimate_tokens([SystemMessage(render_workflow_system_prompt())],self.schemas),
            evidence=evidence_tokens,summary=summary_tokens+64,user=estimate_tokens([current]))
        view=build_context_view(snapshot,turns,ledger)
        boundary=view.layer1_from_msg_id
        if boundary is not None and boundary!=snapshot.layer1_from_msg_id:
            moved=await asyncio.to_thread(self.context_repository.advance_layer1,cid,snapshot.version,boundary)
            if moved:
                initializing=snapshot.layer1_from_msg_id is None and not view.layer2
                self.log.write('context initialized' if initializing else '层1 降级',{'conversation_id':cid,'turn_id':turn,
                    'message':f'层1 初始化 {boundary}' if initializing else f'层1 降级 {snapshot.layer1_from_msg_id}→{boundary}',
                    'from_msg_id':snapshot.layer1_from_msg_id,'to_msg_id':boundary,
                    'initializing':snapshot.layer1_from_msg_id is None and not view.layer2})
                # Freeze summary contents for this turn, but remember its monotonic L1 boundary.
                snapshot=replace(snapshot,version=snapshot.version+1,layer1_from_msg_id=boundary)
                self.snapshots[key]=snapshot
                view=replace(view,snapshot=snapshot)
            else:
                latest=await asyncio.to_thread(self.context_repository.load,cid)
                snapshot=replace(snapshot,version=latest.version,
                    layer1_from_msg_id=max(boundary,latest.layer1_from_msg_id or boundary))
                self.snapshots[key]=snapshot
                view=build_context_view(snapshot,turns,ledger)
        used=estimate_tokens(view.layer2)
        if self.worker and view.pending_turns and used>ledger.layer2:
            pending=list(view.pending_turns)
            # Summarize a contiguous prefix; every batch must itself fit the model.
            while pending:
                try:
                    validate_request(summary_messages(view.summary_text,pending),[],512,self.settings.context_token_budget)
                    break
                except InputBudgetExceeded: pending.pop()
            if pending:
                digest=hashlib.sha256(json.dumps([m.model_dump(mode='json') for t in pending for m in t.messages],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                claim=await asyncio.to_thread(self.context_repository.claim,cid,pending[0].first_msg_id,pending[-1].last_msg_id,digest,
                    excluded_message_ids=excluded)
                self.log.write('summary trigger' if claim else 'summary skip',{'conversation_id':cid,'turn_id':turn,
                    'from_msg_id':pending[0].first_msg_id,'upto_msg_id':pending[-1].last_msg_id,
                    'message':f'summary trigger 层2 约 {used} token > 预算 {ledger.layer2}',
                    'reason':'over_budget' if claim else 'running_or_covered','duration_ms':0})
                if claim:self.worker.submit(claim,pending,view.summary_text)
        return view

    async def model_state(self,state,tools,*,phase):
        evidence=state.get('evidence',[])
        # Count the actual serialized evidence envelope, not its row count.
        r=estimate_tokens([HumanMessage(json.dumps({'本轮检索证据':evidence},ensure_ascii=False))]) if evidence else 0
        view=await self.history_view(state,evidence_tokens=r)
        all_messages=state['messages']
        index=next(i for i in range(len(all_messages)-1,-1,-1) if all_messages[i].type=='human')
        messages=compose_model_messages(SystemMessage(render_workflow_system_prompt()),view,
            all_messages[index],evidence,all_messages[index+1:])
        tokens=validate_request(messages,tools,self.settings.response_token_reserve,self.settings.context_token_budget)
        def log_request(actual_messages,actual_tools):
            self.log.model_request(state['conversation_id'],state['turn_id'],phase,actual_messages,actual_tools,
                {**vars(view.ledger),'actual_input':estimate_tokens(actual_messages,actual_tools)},view.summary_text)
        # This transient callback is never returned as a graph update or checkpointed.
        return {**state,'messages':messages,'_log_model_request':log_request}
