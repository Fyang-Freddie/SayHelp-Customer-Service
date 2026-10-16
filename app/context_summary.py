"""Lifecycle-owned tasks, fenced by MySQL leases; the reply never waits for a summary."""
import asyncio
import re
from time import perf_counter
from langchain_core.messages import HumanMessage
from app.context_budget import estimate_tokens
from app.context_log import ContextLog

IDENTIFIER=re.compile(r'[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*|(?<!\d)\d{6,}(?!\d)')


def validate_summary(content,turns):
    if not isinstance(content,str) or len(content)>200:
        raise ValueError('Summary must be text of at most 200 characters')
    source='\n'.join(str(m.content) for t in turns for m in t.messages)
    if not set(IDENTIFIER.findall(content))<=set(IDENTIFIER.findall(source)):
        raise ValueError('Summary invented an identifier')
    return content.strip()


class SummaryWorker:
    def __init__(self,repository,model,*,log=None,timeout=45.0):
        self.repository,self.model=repository,model
        self.log=log or ContextLog().write
        self.timeout=timeout
        self.tasks={}
        self.closing=False
        self.recover_callback=None

    async def recover(self):
        if self.recover_callback and not self.closing:
            started=perf_counter()
            try:await self.recover_callback()
            except Exception as error:
                self.log('summary failed',{'reason':'recovery_'+type(error).__name__,
                    'from_msg_id':None,'upto_msg_id':None,
                    'duration_ms':(perf_counter()-started)*1000})

    def submit(self,claim,turns,background):
        payload={'conversation_id':claim.conversation_id,'task_id':claim.lease_token,
            'from_msg_id':claim.from_msg_id,'upto_msg_id':claim.upto_msg_id,'ordinal':claim.ordinal}
        if self.closing or claim.conversation_id in self.tasks:
            self.log('summary skip',{**payload,'reason':'closing_or_running','duration_ms':0})
            return
        task=asyncio.create_task(self._run(claim,tuple(turns),background,payload))
        self.tasks[claim.conversation_id]=task
        def done(completed):
            self.tasks.pop(claim.conversation_id,None)
            if not completed.cancelled():completed.exception()
        task.add_done_callback(done)

    async def _run(self,claim,turns,background,payload):
        started=perf_counter()
        try:
            self.log('summary start',{**payload,'duration_ms':0})
            for attempt in range(3):
                # A request lasts at most 45s, below the 120s lease. Renew
                # before every attempt, including retries, and fence the commit.
                if not await asyncio.to_thread(self.repository.renew,claim):
                    self.log('summary skip',{**payload,'reason':'lost_lease',
                        'duration_ms':(perf_counter()-started)*1000})
                    return
                try:
                    from app.ch07_evaluation import evaluation_scope
                    with evaluation_scope('background' if attempt==0 else 'retry'):
                        content=await asyncio.wait_for(self.model.summarize(background=background,turns=turns),self.timeout)
                    break
                except (ConnectionError,TimeoutError) as error:
                    if attempt==2:raise
                    self.log('summary retry',{**payload,'attempt':attempt+2,'reason':type(error).__name__,
                        'duration_ms':(perf_counter()-started)*1000})
            content=validate_summary(content,turns)
            committed=await asyncio.to_thread(self.repository.complete,claim,content,
                estimate_tokens([HumanMessage(content)]))
            self.log('summary done' if committed else 'summary skip',{**payload,
                'message':f'summary done 第{claim.ordinal}段' if committed else 'stale_claim',
                'duration_ms':(perf_counter()-started)*1000})
        except BaseException as error:
            # Release only our lease; a newer owner's claim is never touched.
            try:await asyncio.shield(asyncio.to_thread(self.repository.fail,claim,type(error).__name__))
            finally:self.log('summary failed',{**payload,'reason':type(error).__name__,
                'duration_ms':(perf_counter()-started)*1000})
            if isinstance(error,asyncio.CancelledError):raise

    async def close(self,timeout=5.0):
        self.closing=True
        pending=list(self.tasks.values())
        if not pending:return
        _,remaining=await asyncio.wait(pending,timeout=timeout)
        for task in remaining:task.cancel()
        await asyncio.gather(*pending,return_exceptions=True)
