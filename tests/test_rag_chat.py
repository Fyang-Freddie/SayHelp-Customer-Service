"""Atomic source snapshots and gated chat independent of old fabricated tools."""
import asyncio,sys
from pathlib import Path
from types import SimpleNamespace
from threading import Event
from sqlalchemy import select,event,func
from sqlalchemy.orm import undefer
import pytest
from app.repository import Repository
from app.db import Message
from app.ch04_db import LowConfidenceQuestion
from app.chat_service import ChatService
from app.config import Settings
from app.knowledge_types import PreparedQuery,KnowledgeFilters,ConfidenceDecision,KnowledgeAnswer,RankedChunk,RetrievalResult
from app.rag_generation import RagGenerator,REFUSAL
from app.evidence import select_prompt_evidence
from test_db import database_url,sessions
from test_ch04_ingest import corpus


def settings(): return Settings(chat_base_url='https://example.invalid',chat_model='test',chat_api_key='placeholder')


def sample():
    from test_rag_evidence import result
    q,r=result(3);return select_prompt_evidence(q,r,settings()).citations


def atomic(repo,*args):
    assert hasattr(repo,'commit_knowledge_answer'),'Atomic assistant/evidence/pool port missing'
    return repo.commit_knowledge_answer(*args)


@pytest.mark.parametrize('source',['retrieval_low_conf','self_check'])
def test_false_answer_pool_and_full_snapshot_are_atomic_and_preserve_raw_question(sessions,source):
    repo=Repository(sessions);cid=repo.create_conversation('guest');raw='气死了，到底哪天能到账啊！'
    evidence=sample();answer=KnowledgeAnswer(False,REFUSAL,'标注：缺少具体到账证据',source,evidence)
    mid=atomic(repo,cid,answer,raw)
    assert isinstance(mid,str) and mid.isdecimal()
    with sessions() as session:
        saved=session.get(Message,int(mid),options=[undefer('*')]);pool=list(session.scalars(select(LowConfidenceQuestion)))
        assert saved.content==REFUSAL and saved.citations==evidence and saved.conversation_id==cid
        assert len(pool)==1 and pool[0].raw_question==raw and pool[0].source==source and pool[0].conversation_id==cid
        assert pool[0].reason==answer.reason and pool[0].created_at is not None


def test_true_answer_saves_uncited_evidence_and_never_inserts_pool(sessions):
    repo=Repository(sessions);cid=repo.create_conversation('guest');evidence=sample()
    mid=atomic(repo,cid,KnowledgeAnswer(True,'仅引用首条[1]','充分',None,evidence),'用户原话')
    rows=repo.load_messages(cid)
    assert str(rows[0].id)==mid and rows[0].citations==evidence and len(rows[0].citations)==3
    with sessions() as session: assert session.scalar(select(func.count()).select_from(LowConfidenceQuestion))==0


def test_pool_write_failure_rolls_back_final_assistant_snapshot(sessions):
    repo=Repository(sessions);cid=repo.create_conversation('guest');repo.append_message(cid,'user','保留用户原话')
    def fail(connection,cursor,statement,parameters,context,executemany):
        if 'insert into low_confidence_questions' in statement.lower(): raise RuntimeError('forced pool insert failure')
    engine=sessions.kw['bind'];event.listen(engine,'before_cursor_execute',fail)
    try:
        with pytest.raises(RuntimeError,match='forced pool'): atomic(repo,cid,KnowledgeAnswer(False,REFUSAL,'不足','self_check',sample()),'保留用户原话')
    finally: event.remove(engine,'before_cursor_execute',fail)
    with sessions() as session:
        assert [(r.role,r.content) for r in session.scalars(select(Message))]==[('user','保留用户原话')]
        assert session.scalar(select(func.count()).select_from(LowConfidenceQuestion))==0


class Model:
    def __init__(self,useful=True): self.useful=useful;self.prompts=[]
    async def choose_tool(self,*args): raise AssertionError('Knowledge/chitchat/clarify cannot select fabricated tools')
    async def stream_chat(self,messages): yield '你好'
    async def generate_knowledge(self,messages):
        self.prompts.append(messages)
        return {'useful':self.useful,'answer':'依据本轮证据[1]','reason':'充分' if self.useful else '缺少条件'}

class Query:
    def __init__(self,intent='knowledge'): self.intent=intent;self.calls=[]
    async def prepare(self,raw):
        self.calls.append(raw);return PreparedQuery(raw,'标准问法','检索扩展',intent=self.intent,clarification='请补充型号' if self.intent=='clarify' else None)

class Retrieval:
    def __init__(self,snapshot): self.snapshot=snapshot;self.calls=[]
    def retrieve(self,q,strategy,filters):
        self.calls.append((q,strategy,filters));ranked=[RankedChunk(c,i+1,10-i) for i,c in enumerate(list(self.snapshot.chunks.values())[:3])]
        return RetrievalResult(strategy,q,ranked,ranked,{})


def service(sessions,corpus,intent='knowledge',sufficient=True,useful=True):
    ingest,manifest,_=corpus;snapshot=ingest.ingest_corpus(sessions,manifest)
    repo=Repository(sessions);cid=repo.create_conversation('guest');model=Model(useful);query=Query(intent);retrieval=Retrieval(snapshot)
    policy=SimpleNamespace(assess=lambda result:ConfidenceDecision(sufficient,'通过' if sufficient else '检索低置信'))
    chat=ChatService(repo,model,settings(),None,query_understanding=query,rag_retrieval=retrieval,rag_generator=RagGenerator(model),confidence_policy=policy)
    return chat,repo,cid,model,query,retrieval


def collect(chat,cid,raw,filters=None):
    async def run(): return [e async for e in chat.stream_turn(cid,raw,filters)]
    return asyncio.run(run())


def test_knowledge_uses_only_current_question_and_evidence_and_commits_before_citations_tokens(sessions,corpus):
    chat,repo,cid,model,query,retrieval=service(sessions,corpus)
    repo.append_message(cid,'assistant','错误历史事实：保证1天到账')
    raw='我问的原话，不许用历史事实！';filters=KnowledgeFilters(product_category='猫砂盆')
    async def run():
        events=[]
        async for item in chat.stream_turn(cid,raw,filters):
            if item.kind in ('citations','token'):
                assert any(row.role=='assistant' and row.content=='依据本轮证据[1]' for row in repo.load_messages(cid))
            events.append(item)
        return events
    events=asyncio.run(run());kinds=[e.kind for e in events]
    assert query.calls==[raw] and retrieval.calls[0][0].raw_question==raw and retrieval.calls[0][2] is filters
    assert kinds.index('citations')<kinds.index('token')
    citation=next(e.data for e in events if e.kind=='citations')
    assert citation['message_id'].isdecimal() and len(citation['citations'])==3
    body=''.join(m.content for m in model.prompts[0])
    assert raw in body and '错误历史事实' not in body and '检索扩展' not in body
    with sessions() as session: assert session.scalar(select(func.count()).select_from(LowConfidenceQuestion))==0


@pytest.mark.parametrize('sufficient,useful,source',[(False,True,'retrieval_low_conf'),(True,False,'self_check')])
def test_knowledge_refusal_commits_one_pool_row_with_raw_question(sessions,corpus,sufficient,useful,source):
    chat,repo,cid,model,query,retrieval=service(sessions,corpus,sufficient=sufficient,useful=useful)
    raw='啥玩意呀，我就是问这个原话！';events=collect(chat,cid,raw)
    assert ''.join(e.data['text'] for e in events if e.kind=='token')==REFUSAL
    with sessions() as session:
        rows=list(session.scalars(select(LowConfidenceQuestion)))
        assert len(rows)==1 and rows[0].raw_question==raw and rows[0].source==source and rows[0].conversation_id==cid
    assert len(model.prompts)==int(sufficient)


@pytest.mark.parametrize('intent',['clarify','chitchat'])
def test_nonknowledge_routes_never_retrieve_or_fill_pool(sessions,corpus,intent):
    chat,repo,cid,model,query,retrieval=service(sessions,corpus,intent=intent)
    events=collect(chat,cid,'你好' if intent=='chitchat' else '它咋用')
    assert retrieval.calls==[] and model.prompts==[]
    assert any(e.kind=='token' for e in events)
    with sessions() as session: assert session.scalar(select(func.count()).select_from(LowConfidenceQuestion))==0


def test_repeated_cancellation_waits_for_atomic_write_worker(sessions,corpus,monkeypatch):
    chat,repo,cid,model,query,retrieval=service(sessions,corpus,useful=False)
    assert hasattr(repo,'commit_knowledge_answer'),'Atomic assistant/evidence/pool port missing'
    original=repo.commit_knowledge_answer;entered=Event();release=Event()
    def slow(*args): entered.set();assert release.wait(5);return original(*args)
    monkeypatch.setattr(repo,'commit_knowledge_answer',slow)
    async def run():
        task=asyncio.create_task(_drain(chat,cid))
        assert await asyncio.to_thread(entered.wait,5)
        task.cancel();await asyncio.sleep(.03);task.cancel();await asyncio.sleep(.03)
        assert not task.done(),'Cancellation released while synchronous pool commit was still pending'
        release.set()
        with pytest.raises(asyncio.CancelledError): await task
    try: asyncio.run(run())
    finally: release.set()
    with sessions() as session:
        assert len(list(session.scalars(select(LowConfidenceQuestion))))==1
        assert len([r for r in session.scalars(select(Message)) if r.role=='assistant'])==1

async def _drain(chat,cid): return [e async for e in chat.stream_turn(cid,'原始口语')]


@pytest.mark.parametrize('name,args',[('query_faq',{'keyword':'模型擅自替换问题'}),('query_product',{'product_query':'随机库存'})])
def test_operation_misclassified_knowledge_tool_uses_current_gated_pipeline(sessions,corpus,name,args):
    from langchain_core.messages import AIMessage
    import json
    chat,repo,cid,model,query,retrieval=service(sessions,corpus,intent='operation',useful=False)
    async def choose(*unused):
        return AIMessage(content='',tool_calls=[{'id':'first','name':name,'args':args},
                         {'id':'skip','name':'create_ticket','args':{'description':'不能执行','ticket_type':'咨询'}}])
    async def no_ungated(*unused):
        raise AssertionError('Final knowledge must not go through ordinary generation')
        yield ''
    model.choose_tool=choose;model.stream_chat=no_ungated
    events=collect(chat,cid,'用户当前原话')
    assert len(retrieval.calls)==1 and retrieval.calls[0][0].raw_question=='用户当前原话'
    rows=repo.load_messages(cid);tools=[row for row in rows if row.role=='tool']
    assert [row.tool_call_id for row in tools]==['first','skip']
    payload=json.loads(tools[0].content)
    assert payload['useful'] is False and payload['matches']==[] and payload['citations']
    assert '随机模拟商品' not in tools[0].content
    with sessions() as session:
        from app.db import Ticket
        assert session.scalar(select(func.count()).select_from(Ticket))==0
        assert session.scalar(select(func.count()).select_from(LowConfidenceQuestion))==1
    assert ''.join(e.data['text'] for e in events if e.kind=='token')==REFUSAL


def test_generation_cancelled_before_result_never_writes_final_or_pool(sessions,corpus):
    chat,repo,cid,model,query,retrieval=service(sessions,corpus)
    async def run():
        entered=asyncio.Event()
        async def pending(messages): entered.set();await asyncio.Event().wait()
        model.generate_knowledge=pending
        task=asyncio.create_task(_drain(chat,cid));await asyncio.wait_for(entered.wait(),2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
    asyncio.run(run())
    assert [r.role for r in repo.load_messages(cid)]==['user']
    with sessions() as session: assert session.scalar(select(func.count()).select_from(LowConfidenceQuestion))==0



def test_pure_faq_tool_retains_shape_and_gates_evidence_with_explicit_filters(sessions,corpus):
    from app.tools import build_tools
    chat,repo,cid,model,query,retrieval=service(sessions,corpus,useful=False)
    assert hasattr(chat,'knowledge_tool_answer'),'Shared pure knowledge tool gate missing'
    tools=build_tools(repo,cid,None,knowledge_answer=chat.knowledge_tool_answer)
    async def run(): return await tools['query_faq'].ainvoke({'keyword':'工具原话','filters':{'product_category':'猫砂盆'}})
    result=asyncio.run(run())
    assert result['keyword']=='工具原话' and result['matches']==[] and result['useful'] is False
    assert result['citations'] and query.calls==['工具原话'] and retrieval.calls[0][2].product_category=='猫砂盆'
    assert repo.load_messages(cid)==[]
    with sessions() as session: assert session.scalar(select(func.count()).select_from(LowConfidenceQuestion))==0
