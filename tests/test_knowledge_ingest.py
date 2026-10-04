import pytest
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker

from app.db import Faq
from app.init_knowledge_db import initialize_knowledge_database
from app.knowledge_db import KnowledgeChunk
from tests.test_knowledge_db import database_url


@pytest.fixture
def sessions(database_url):
    from sqlalchemy import create_engine
    initialize_knowledge_database(database_url)
    engine = create_engine(database_url)
    try:
        yield sessionmaker(engine, expire_on_commit=False)
    finally:
        engine.dispose()


def markdown(tmp_path, text):
    path = tmp_path / 'policy.md'
    path.write_text(text, encoding='utf-8')
    return path


def test_faq_seed_ingests_once_with_real_question(sessions):
    from app.knowledge_ingest import ingest_faq
    ids = ingest_faq(sessions)
    assert ingest_faq(sessions) == ids
    with sessions() as session:
        rows = session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)).all()
        assert len(rows) == session.scalar(select(func.count()).select_from(Faq))
        row = next(row for row in rows if '运费' in row.questions)
        assert (row.category, row.questions, row.answer) == (
            '配送', '订单运费如何计算？', '运费根据收货地区与订单金额计算，请以结算页显示为准。')
        assert (row.content_type, row.section_path, row.is_key_clause) == ('faq', None, 0)
        assert all(row.vectorize_status == 'pending' and row.vector_id is None for row in rows)
        assert all(row.prev_chunk_id is None and row.next_chunk_id is None for row in rows)
        row.vectorize_status, row.vector_id = 'done', str(row.id)
        session.commit()
    assert ingest_faq(sessions) == ids
    with sessions() as session:
        assert session.get(KnowledgeChunk, row.id).vectorize_status == 'done'


def test_markdown_ingestion_links_chunks_atomically(sessions, tmp_path):
    from app.knowledge_ingest import ingest_markdown
    path = markdown(tmp_path, '# 政策\n## 配送\n首段。\n\n第二段。\n\n> [!IMPORTANT]\n> 关键条款。\n')
    ids = ingest_markdown(sessions, path, 'policy')
    assert len(ids) == 3
    assert ingest_markdown(sessions, path, 'policy') == ids
    # CRLF, trailing whitespace, and surrounding empty lines are non-authoritative.
    path.write_bytes(('\n# 政策  \r\n## 配送\r\n首段。  \r\n\r\n第二段。\r\n\r\n> [!IMPORTANT]\r\n> 关键条款。\r\n\r\n').encode('utf-8'))
    assert ingest_markdown(sessions, path, 'policy') == ids
    with sessions() as session:
        rows = [session.get(KnowledgeChunk, id_) for id_ in ids]
        assert session.scalar(select(func.count()).select_from(KnowledgeChunk)) == 3
        assert [row.answer for row in rows] == ['首段。', '第二段。', '关键条款。']
        assert [row.prev_chunk_id for row in rows] == [None, ids[0], ids[1]]
        assert [row.next_chunk_id for row in rows] == [ids[1], ids[2], None]
        assert all((row.category, row.questions, row.section_path, row.content_type) ==
                   ('政策', '配送', '政策 / 配送', 'policy') for row in rows)
        assert [row.is_key_clause for row in rows] == [0, 0, 1]
        assert all(row.vectorize_status == 'pending' and row.vector_id is None for row in rows)


def test_failed_ingestion_rolls_back_links(sessions, tmp_path):
    from app.knowledge_ingest import ingest_markdown
    path = markdown(tmp_path, '# 政策\n首段。\n\n第二段。')
    engine = sessions.kw['bind']
    def fail_on_link(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith('UPDATE KNOWLEDGE_CHUNKS'):
            raise RuntimeError('injected link failure')
    event.listen(engine, 'before_cursor_execute', fail_on_link)
    try:
        with pytest.raises(RuntimeError, match='injected link failure'):
            ingest_markdown(sessions, path, 'policy')
    finally:
        event.remove(engine, 'before_cursor_execute', fail_on_link)
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(KnowledgeChunk)) == 0
    # A successful retry verifies recovery after the failed transaction.
    assert len(ingest_markdown(sessions, path, 'policy')) == 2


def test_changed_document_preserves_old_chain_and_complete_metadata(sessions, tmp_path):
    from app.knowledge_ingest import ingest_markdown
    path = markdown(tmp_path, '# 政策\n首段。\n\n旧条款。')
    old_ids = ingest_markdown(sessions, path, 'policy')
    with sessions() as session:
        for id_ in old_ids:
            row = session.get(KnowledgeChunk, id_)
            row.vectorize_status, row.vector_id = 'done', str(id_)
        session.commit()
    path.write_text('# 政策\n首段。\n\n新条款。', encoding='utf-8')
    new_ids = ingest_markdown(sessions, path, 'policy')
    assert not set(new_ids) & set(old_ids)
    assert ingest_markdown(sessions, path, 'policy') == new_ids
    manual_ids = ingest_markdown(sessions, path, 'manual')
    assert not set(manual_ids) & set(new_ids)
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(KnowledgeChunk)) == 6
        assert session.get(KnowledgeChunk, old_ids[0]).next_chunk_id == old_ids[1]
        old = session.get(KnowledgeChunk, old_ids[1])
        assert (old.answer, old.vectorize_status, old.vector_id) == ('旧条款。', 'done', str(old.id))


def test_repeated_identical_chunks_keep_source_order(sessions, tmp_path):
    from app.knowledge_ingest import ingest_markdown
    path = markdown(tmp_path, '# 政策\n相同段落。\n\n相同段落。')
    ids = ingest_markdown(sessions, path, 'policy')
    assert len(set(ids)) == 2
    assert ingest_markdown(sessions, path, 'policy') == ids


def test_markdown_preserves_indentation_and_error_line(sessions, tmp_path):
    from app.knowledge_ingest import ingest_markdown
    path = markdown(tmp_path, '    # literal\nBody.')
    ids = ingest_markdown(sessions, path, 'manual')
    with sessions() as session:
        row = session.get(KnowledgeChunk, ids[0])
        assert row.questions == ''
        assert '# literal' in row.answer
    path.write_text('\n\n# Manual\n' + 'A' * 900 + '.', encoding='utf-8')
    with pytest.raises(ValueError, match=r'line 4: oversized indivisible sentence'):
        ingest_markdown(sessions, path, 'manual')


def test_lock_released_if_commit_after_acquisition_fails(sessions, database_url):
    from app.knowledge_ingest import _ingestion_session
    engine = sessions.kw['bind']
    state = {'raised': False}

    def fail_first_commit(connection):
        if not state['raised']:
            state['raised'] = True
            raise RuntimeError('injected lock acquisition commit failure')

    event.listen(engine, 'commit', fail_first_commit)
    try:
        with pytest.raises(RuntimeError, match='injected lock acquisition commit failure'):
            with _ingestion_session(sessions):
                pass
    finally:
        event.remove(engine, 'commit', fail_first_commit)

    independent = create_engine(database_url, poolclass=NullPool)
    try:
        with independent.connect() as connection:
            lock_name = "CONCAT('sayhelp_ingest_', MD5(DATABASE()))"
            assert connection.execute(text(f'SELECT GET_LOCK({lock_name}, 0)')).scalar() == 1
            assert connection.execute(text(f'SELECT RELEASE_LOCK({lock_name})')).scalar() == 1
    finally:
        independent.dispose()


def test_ingested_fenced_code_preserves_whitespace_and_section(sessions, tmp_path):
    from app.knowledge_ingest import ingest_markdown
    block = '```sh\n# reboot system\n  reboot  \n```'
    path = markdown(tmp_path, '# Manual\n## Restart\n' + block + '\nAfter.')
    ids = ingest_markdown(sessions, path, 'manual')
    assert ingest_markdown(sessions, path, 'manual') == ids
    with sessions() as session:
        rows = [session.get(KnowledgeChunk, id) for id in ids]
        assert [row.answer for row in rows] == [block, 'After.']
        assert all(row.section_path == 'Manual / Restart' for row in rows)
