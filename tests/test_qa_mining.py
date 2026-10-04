import json
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import event, select

from app.db import Conversation, Message
from app.knowledge_db import KnowledgeChunk, QaExtractionStaging
from tests.test_knowledge_ingest import sessions
from tests.test_knowledge_db import database_url


class Model:
    def __init__(self, callback=None):
        self.inputs = []
        self.callback = callback
    def invoke(self, messages):
        data = json.loads(messages[-1].content)
        self.inputs.append(data)
        if self.callback:
            return AIMessage(content=json.dumps(self.callback(data)))
        return AIMessage(content=json.dumps({'pairs': [
            {'source_ref': t['source_ref'], 'question': t['user'], 'answer': t['assistant']}
            for c in data['conversations'] for t in c['turns']]}))


class Embeddings:
    def encode(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts]


def seed(sessions, pairs):
    with sessions() as s, s.begin():
        c = Conversation(user_id='mining-test')
        s.add(c)
        s.flush()
        for role, content, calls, call_id in pairs:
            s.add(Message(conversation_id=c.id, role=role, content=content,
                          tool_calls=calls, tool_call_id=call_id))
        return c.id


def turn(q='运费多少？', a='运费为10元。'):
    return [('user', q, None, None), ('assistant', a, None, None)]


def rows(sessions, model):
    with sessions() as s:
        return s.scalars(select(model).order_by(model.id)).all()


def mine(sessions, model, **kwargs):
    from app.qa_mining import mine_conversations
    return mine_conversations(sessions, model, embedder=Embeddings(), **kwargs)


def test_mining_batches_complete_conversations_only(sessions):
    c1 = seed(sessions, turn() + [('user', 'unfinished', None, None)])
    c2 = seed(sessions, [('user', '退货期限？', None, None),
        ('assistant', 'private payload', [{'id':'call1','name':'query_faq','args':{},'type':'tool_call'}], None),
        ('tool', 'secret payload', None, 'call1'), ('assistant', '七天内。', None, None)])
    seed(sessions, [('user', 'partial', None, None)])
    model = Model()
    result = mine(sessions, model, batch_size=1)
    assert result.kept == 2
    assert [x['conversations'][0]['id'] for x in model.inputs] == [c1, c2]
    assert 'unfinished' not in json.dumps(model.inputs)
    assert 'payload' not in json.dumps(model.inputs)
    assert [r.question for r in rows(sessions, QaExtractionStaging)] == ['运费多少？', '退货期限？']


def test_staging_precedes_global_dedup(sessions):
    seed(sessions, turn())
    seed(sessions, turn(q='邮费多少？'))
    class CheckEmbedding(Embeddings):
        def encode(self, texts):
            assert [r.status for r in rows(sessions, QaExtractionStaging)] == ['extracted', 'extracted']
            return super().encode(texts)
    from app.qa_mining import mine_conversations
    model = Model(lambda d: {'equivalent':True} if 'left' in d else {'pairs':[
        {'source_ref':t['source_ref'],'question':t['user'],'answer':t['assistant']}
        for c in d['conversations'] for t in c['turns']]})
    result = mine_conversations(sessions, model, batch_size=1, embedder=CheckEmbedding())
    assert (result.kept, result.discarded) == (1, 1)
    assert [r.status for r in rows(sessions, QaExtractionStaging)] == ['kept', 'discarded']
    k = rows(sessions, KnowledgeChunk)[0]
    assert (k.category, k.questions, k.answer, k.content_type, k.vectorize_status) == (
        '历史客服对话', '运费多少？', '运费为10元。', 'faq', 'pending')


def test_conflicting_answers_survive_dedup(sessions):
    seed(sessions, turn())
    seed(sessions, turn(a='运费为20元。'))
    model = Model(lambda data: {'equivalent': False} if 'left' in data else {'pairs': [
        {'source_ref':t['source_ref'], 'question':t['user'], 'answer':t['assistant']}
        for c in data['conversations'] for t in c['turns']]})
    result = mine(sessions, model, batch_size=1)
    assert result.kept == 2
    assert [r.answer for r in rows(sessions, KnowledgeChunk)] == ['运费为10元。', '运费为20元。']


def test_mining_rerun_is_idempotent(sessions):
    seed(sessions, turn())
    first = mine(sessions, Model())
    second_model = Model()
    second = mine(sessions, second_model)
    assert first.kept == 1 and second.kept == 0
    assert len(rows(sessions, KnowledgeChunk)) == len(rows(sessions, QaExtractionStaging)) == 1
    assert second_model.inputs == []
    # New complete turn is eligible without re-extracting the old one.
    with sessions() as s, s.begin():
        c = s.scalar(select(Conversation).order_by(Conversation.id))
        s.add_all([Message(conversation_id=c.id, role='user', content='期限？'),
                   Message(conversation_id=c.id, role='assistant', content='七天。')])
    assert mine(sessions, Model(lambda d: {'equivalent':False} if 'left' in d else {'pairs':[
        {'source_ref':t['source_ref'],'question':t['user'],'answer':t['assistant']}
        for c in d['conversations'] for t in c['turns']]})).kept == 1


def test_knowledge_and_status_roll_back_together_and_resume(sessions):
    seed(sessions, turn())
    engine = sessions.kw['bind']
    def fail(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith('UPDATE QA_EXTRACTION_STAGING'):
            raise RuntimeError('injected finalize failure')
    event.listen(engine, 'before_cursor_execute', fail)
    try:
        with pytest.raises(RuntimeError, match='finalize failure'):
            mine(sessions, Model())
    finally:
        event.remove(engine, 'before_cursor_execute', fail)
    assert rows(sessions, KnowledgeChunk) == []
    assert [r.status for r in rows(sessions, QaExtractionStaging)] == ['extracted']
    assert mine(sessions, Model()).kept == 1


@pytest.mark.parametrize('output', [{'pairs':[{'source_ref':'invented','question':'x','answer':'y'}]},
    {'pairs':[{'source_ref':'COPY','question':'invented','answer':'invented'}]}, {'pairs':'wrong'}])
def test_invalid_or_unsupported_extraction_never_stages(sessions, output):
    seed(sessions, turn())
    def response(data):
        if isinstance(output['pairs'], list) and output['pairs'][0]['source_ref'] == 'COPY':
            output['pairs'][0]['source_ref'] = data['conversations'][0]['turns'][0]['source_ref']
        return output
    with pytest.raises(ValueError):
        mine(sessions, Model(response))
    assert rows(sessions, QaExtractionStaging) == []
    assert rows(sessions, KnowledgeChunk) == []


def test_oversized_turn_and_credentials_are_excluded(sessions):
    seed(sessions, turn(q='a' * 20000))
    seed(sessions, turn(q='api_key=privatecredential'))
    assert mine(sessions, Model()).kept == 0


def test_scheduler_runs_one_daily_job(monkeypatch):
    from apscheduler.schedulers.blocking import BlockingScheduler
    from app import qa_scheduler
    scheduler = BlockingScheduler(timezone='Asia/Shanghai')
    monkeypatch.setattr(scheduler, 'start', lambda: None)
    calls = []
    monkeypatch.setattr(qa_scheduler, 'run_once', lambda *a, **k: calls.append(1))
    qa_scheduler.run_scheduled_mining(None, None, None, None, scheduler=scheduler)
    jobs = scheduler.get_jobs()
    assert len(jobs) == 1 and jobs[0].max_instances == 1 and jobs[0].coalesce is True
    next_time = jobs[0].trigger.get_next_fire_time(None, datetime(2026, 10, 5, 0, 0, tzinfo=scheduler.timezone))
    assert (next_time.hour, next_time.minute, str(next_time.tzinfo)) == (2, 0, 'Asia/Shanghai')
    jobs[0].func(*jobs[0].args, **jobs[0].kwargs)
    assert calls == [1]


def test_near_duplicates_require_both_meanings_and_global_existing_knowledge(sessions):
    with sessions() as s, s.begin():
        s.add(KnowledgeChunk(category='配送', questions='运费多少？', answer='运费为10元。', content_type='faq'))
    seed(sessions, turn(q='邮费多少？', a='配送费为10元。'))
    model = Model(lambda d: {'equivalent':True} if 'left' in d else {'pairs':[
        {'source_ref':t['source_ref'],'question':t['user'],'answer':t['assistant']}
        for c in d['conversations'] for t in c['turns']]})
    assert mine(sessions, model).discarded == 1
    assert len(rows(sessions, KnowledgeChunk)) == 1
    assert [r.status for r in rows(sessions, QaExtractionStaging)] == ['discarded']


def test_exact_duplicates_do_not_require_embedding_inference(sessions):
    seed(sessions, turn())
    seed(sessions, turn())
    class Unavailable:
        def encode(self, texts):
            raise RuntimeError('embedding unavailable')
    from app.qa_mining import mine_conversations
    assert mine_conversations(sessions, Model(), embedder=Unavailable()).discarded == 1


def test_embedding_failure_preserves_all_committed_batches(sessions):
    seed(sessions, turn())
    seed(sessions, turn(q='退货期限？', a='七天内。'))
    class Unavailable:
        def encode(self, texts):
            raise RuntimeError('embedding unavailable')
    from app.qa_mining import mine_conversations
    with pytest.raises(RuntimeError, match='embedding unavailable'):
        mine_conversations(sessions, Model(), batch_size=1, embedder=Unavailable())
    assert [r.status for r in rows(sessions, QaExtractionStaging)] == ['extracted', 'extracted']
    assert rows(sessions, KnowledgeChunk) == []
    model = Model(lambda d: {'equivalent':False})
    assert mine(sessions, model).kept == 2
    assert all('left' in d for d in model.inputs)


def test_numeric_conflict_survives_even_positive_model_judgment(sessions):
    seed(sessions, turn())
    seed(sessions, turn(a='运费为20元。'))
    model = Model(lambda d: {'equivalent':True} if 'left' in d else {'pairs':[
        {'source_ref':t['source_ref'],'question':t['user'],'answer':t['assistant']}
        for c in d['conversations'] for t in c['turns']]})
    assert mine(sessions, model).kept == 2


def test_one_shot_indexes_staged_knowledge_and_rejects_no_progress(sessions, monkeypatch):
    from app import qa_scheduler
    seed(sessions, turn())
    class Store:
        def ensure_collection(self):
            pass
        def upsert(self, id, vector):
            return id
    assert qa_scheduler.run_once(sessions, Model(), Embeddings(), Store()).kept == 1
    assert rows(sessions, KnowledgeChunk)[0].vectorize_status == 'done'
    with sessions() as s, s.begin():
        s.get(KnowledgeChunk, rows(sessions, KnowledgeChunk)[0].id).vectorize_status = 'pending'
    monkeypatch.setattr(qa_scheduler, 'index_pending', lambda *a: 0)
    with pytest.raises(RuntimeError, match='no progress'):
        qa_scheduler.run_once(sessions, Model(), Embeddings(), Store())


def test_prompt_batches_bound_complete_turns_without_truncating(sessions):
    for _ in range(5):
        seed(sessions, turn(q='问' * 1500, a='答' * 1500))
    model = Model(lambda d: {'pairs':[]})
    assert mine(sessions, model, batch_size=20).kept == 0
    assert len(model.inputs) >= 2
    assert sum(len(c['turns']) for d in model.inputs for c in d['conversations']) == 5
    assert all(len(json.dumps(d, ensure_ascii=False)) <= 12000 for d in model.inputs)
    assert all(len(t['user']) == len(t['assistant']) == 1500 for d in model.inputs for c in d['conversations'] for t in c['turns'])


def test_plain_language_credentials_do_not_reach_llm(sessions):
    seed(sessions, turn(q='我的密码是privatecredential，请登录。'))
    model = Model(lambda d: {'pairs':[]})
    mine(sessions, model)
    assert model.inputs == []


def test_competing_miner_is_rejected_during_model_io(sessions):
    seed(sessions, turn())
    def response(data):
        with pytest.raises(RuntimeError, match='already running'):
            mine(sessions, Model())
        return {'pairs':[{'source_ref':t['source_ref'],'question':t['user'],'answer':t['assistant']}
            for c in data['conversations'] for t in c['turns']]}
    assert mine(sessions, Model(response)).kept == 1


def test_dedup_snapshot_serializes_other_knowledge_ingestion_without_open_transaction(sessions):
    from sqlalchemy import text
    seed(sessions, turn())
    seed(sessions, turn(q='邮费多少？'))
    active = set()
    engine = sessions.kw['bind']
    def begin(connection):
        active.add(connection)
    def finish(connection):
        active.discard(connection)
    event.listen(engine, 'begin', begin)
    event.listen(engine, 'commit', finish)
    event.listen(engine, 'rollback', finish)
    def response(data):
        if 'left' in data:
            assert not active
            with engine.connect() as connection:
                acquired = connection.execute(text("SELECT GET_LOCK(CONCAT('sayhelp_ingest_', MD5(DATABASE())), 0)")).scalar()
                if acquired == 1:
                    connection.execute(text("SELECT RELEASE_LOCK(CONCAT('sayhelp_ingest_', MD5(DATABASE())))"))
                assert acquired == 0
            return {'equivalent':True}
        return {'pairs':[{'source_ref':t['source_ref'],'question':t['user'],'answer':t['assistant']}
            for c in data['conversations'] for t in c['turns']]}
    try:
        assert mine(sessions, Model(response)).discarded == 1
    finally:
        event.remove(engine, 'begin', begin)
        event.remove(engine, 'commit', finish)
        event.remove(engine, 'rollback', finish)


def test_exact_existing_match_is_checked_before_any_semantic_candidate(sessions):
    with sessions() as s, s.begin():
        s.add_all([KnowledgeChunk(category='退货', questions='期限？', answer='七天内。', content_type='faq'),
            KnowledgeChunk(category='配送', questions='运费多少？', answer='运费为10元。', content_type='faq')])
    seed(sessions, turn())
    class Unavailable:
        def encode(self, texts):
            raise RuntimeError('embedding unavailable')
    from app.qa_mining import mine_conversations
    assert mine_conversations(sessions, Model(), embedder=Unavailable()).discarded == 1
    assert len(rows(sessions, KnowledgeChunk)) == 2


def test_cli_once_uses_mining_and_indexing_without_starting_scheduler(sessions, monkeypatch, capsys):
    from types import SimpleNamespace
    from app import qa_scheduler
    seed(sessions, turn())
    class Store:
        def ensure_collection(self):
            pass
        def upsert(self, id, vector):
            return id
    class ConfiguredModel(Model):
        def bind(self, **kwargs):
            return self
    monkeypatch.setattr(qa_scheduler.Settings, 'from_env', lambda: SimpleNamespace(
        database_url='configured',chat_base_url='configured',chat_model='configured',
        chat_api_key='test-placeholder',bge_cache_dir=None,milvus_uri='configured'))
    monkeypatch.setattr(qa_scheduler,'initialize_knowledge_database',lambda _: None)
    monkeypatch.setattr(qa_scheduler,'make_session_factory',lambda _: sessions)
    monkeypatch.setattr(qa_scheduler,'ChatOpenAI',lambda **_: ConfiguredModel())
    monkeypatch.setattr(qa_scheduler,'BgeM3Embedder',lambda **_: Embeddings())
    monkeypatch.setattr(qa_scheduler,'MilvusKnowledgeStore',lambda **_: Store())
    def unexpected_schedule(*args, **kwargs):
        raise AssertionError('scheduler started during --once')
    monkeypatch.setattr(qa_scheduler,'run_scheduled_mining',unexpected_schedule)
    qa_scheduler.main(['--once','--batch-size','1'])
    assert 'staged=1, kept=1, discarded=0' in capsys.readouterr().out
    assert rows(sessions, KnowledgeChunk)[0].vectorize_status == 'done'


def test_escaped_control_chars_turn_exceeding_json_budget_is_excluded(sessions):
    # Raw evidence fits the 4000-character turn bound, but JSON escapes each
    # control character as six characters and exceeds the request budget.
    seed(sessions, turn(q='\u0001' * 1997, a='\u0001' * 1997))
    model = Model(lambda data: {'pairs': []})
    assert mine(sessions, model).kept == 0
    assert model.inputs == []
    assert rows(sessions, QaExtractionStaging) == []
