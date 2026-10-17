import asyncio
from datetime import datetime, UTC, timedelta
from langchain_core.messages import HumanMessage, AIMessage
from app.context_types import SummaryClaim, TurnRecord


class Repo:
    def __init__(self): self.committed=[]; self.failed=[]
    def complete(self,claim,content,tokens): self.committed.append(content); return True
    def fail(self,claim,reason): self.failed.append(reason)
    def renew(self,claim): return True


def claim():
    return SummaryClaim(1,1,1,2,None,'lease','a'*64,datetime.now(UTC)+timedelta(seconds=120))


def batch():
    return (TurnRecord('1',1,2,(HumanMessage('订单ABC123要求退款'),AIMessage('待核实'))),)


def test_slow_summary_does_not_block_reply(tmp_path):
    from app.context_summary import SummaryWorker
    async def run():
        entered,release=asyncio.Event(),asyncio.Event()
        class Model:
            async def summarize(self,**kwargs):
                entered.set(); await release.wait(); return '用户要求订单ABC123退款，尚待核实。'
        repo=Repo(); worker=SummaryWorker(repo,Model(),log=lambda *a:None)
        worker.submit(claim(),batch(),'')
        await asyncio.wait_for(entered.wait(),1)
        assert not repo.committed
        # Caller can complete independently while summary is still suspended.
        reply=asyncio.create_task(asyncio.sleep(0,result='当前轮回复'))
        assert await asyncio.wait_for(reply,1)=='当前轮回复'
        worker.submit(claim(),batch(),'')  # same conversation already running
        release.set(); await worker.close()
        assert len(repo.committed)==1
    asyncio.run(run())


def test_summary_rejects_invented_ids_and_keeps_boundary():
    from app.context_summary import SummaryWorker
    async def run():
        class Model:
            async def summarize(self,**kwargs):return '订单ZZ999已退款。'
        repo=Repo(); worker=SummaryWorker(repo,Model(),log=lambda *a:None)
        worker.submit(claim(),batch(),'旧摘要有订单ZZ999')
        await worker.close()
        assert not repo.committed and repo.failed
    asyncio.run(run())


def test_timeout_and_shutdown_release_claim():
    from app.context_summary import SummaryWorker
    async def run():
        class Model:
            async def summarize(self,**kwargs):await asyncio.Event().wait()
        repo=Repo(); worker=SummaryWorker(repo,Model(),log=lambda *a:None,timeout=.01)
        worker.submit(claim(),batch(),'')
        await worker.close()
        assert repo.failed and not repo.committed
    asyncio.run(run())


def test_lost_lease_never_calls_or_commits_model():
    from app.context_summary import SummaryWorker
    async def run():
        class LostRepo(Repo):
            def renew(self,claim): return False
        class Model:
            async def summarize(self,**kwargs): raise AssertionError('lease already lost')
        repo=LostRepo(); events=[]
        worker=SummaryWorker(repo,Model(),log=lambda event,payload:events.append((event,payload)))
        worker.submit(claim(),batch(),'')
        await worker.close()
        assert not repo.committed
        assert any(e=='summary skip' and p.get('reason')=='lost_lease' for e,p in events)
    asyncio.run(run())


def test_transient_failure_retries_at_most_twice():
    from app.context_summary import SummaryWorker
    async def run():
        class Model:
            calls=0
            async def summarize(self,**kwargs):
                self.calls+=1
                if self.calls<3: raise ConnectionError('synthetic')
                return '用户要求订单ABC123退款，待核实。'
        model=Model();repo=Repo()
        worker=SummaryWorker(repo,model,log=lambda *a:None)
        worker.submit(claim(),batch(),'')
        await worker.close()
        assert model.calls==3 and len(repo.committed)==1
    asyncio.run(run())


def test_recovery_scan_failure_is_logged_and_next_scan_can_run():
    from app.context_summary import SummaryWorker
    async def run():
        events=[]; scans=[]
        worker=SummaryWorker(Repo(),None,log=lambda e,p:events.append((e,p)))
        async def scan():
            scans.append(1)
            if len(scans)==1:raise ConnectionError('temporary database failure')
        worker.recover_callback=scan
        await worker.recover()
        await worker.recover()
        assert len(scans)==2
        assert any(e=='summary failed' and p['reason']=='recovery_ConnectionError' for e,p in events)
    asyncio.run(run())
