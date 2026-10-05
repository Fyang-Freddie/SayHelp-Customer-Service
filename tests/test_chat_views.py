"""Chat history and source previews exercise real storage and HTTP routes."""
import asyncio
import json

import httpx
import pytest

from app.db import make_session_factory
from app.init_db import initialize_database
from app.main import create_app
from app.repository import Repository
from test_chat_api import settings, FakeModel, parse_events, post
from test_db import database_url
from test_tools import FakeKnowledgeSearch


@pytest.fixture
def factory(database_url):
    from app.init_ch04_db import initialize_ch04_database
    initialize_ch04_database(database_url)
    result = make_session_factory(database_url)
    yield result
    result.kw['bind'].dispose()


def get(app, path):
    async def fetch():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            return await client.get(path)
    return asyncio.run(fetch())


def test_history_lists_recent_activity_and_restores_only_public_messages(factory):
    repo = Repository(factory)
    older, newer = repo.create_conversation('guest-old'), repo.create_conversation('guest-new')
    repo.append_message(older, 'user', '退货怎么申请？')
    repo.append_message(older, 'assistant', '内部草稿', tool_calls=[{'id': 'faq-1', 'name': 'query_faq', 'args': {'keyword': '退货'}}])
    repo.append_message(older, 'tool', json.dumps({'matches': [{'question': '退货条件', 'answer': '完整原文', 'category': '售后'}]}), tool_call_id='faq-1')
    repo.append_message(older, 'assistant', '请看条件[1]。')
    repo.append_message(newer, 'user', '另一个会话')
    repo.append_message(newer, 'assistant', '另一条答复')
    repo.append_message(older, 'user', '继续旧会话')
    repo.append_message(older, 'assistant', '继续答复')
    app = create_app(settings(), FakeModel(), factory, FakeKnowledgeSearch())
    response = get(app, '/v1/conversations?limit=1')
    assert response.status_code == 200
    listing = response.json()
    assert listing['conversations'][0]['id'] == str(older)
    assert listing['conversations'][0]['title'] == '退货怎么申请？'
    more = get(app, '/v1/conversations?limit=1&before=' + listing['next_cursor']).json()
    assert [item['id'] for item in more['conversations']] == [str(newer)]
    restored = get(app, f'/v1/conversations/{older}/messages')
    assert restored.status_code == 200
    messages = restored.json()['messages']
    assert [(item['role'], item['content']) for item in messages] == [
        ('user', '退货怎么申请？'), ('assistant', '请看条件[1]。'), ('user', '继续旧会话'), ('assistant', '继续答复')]
    assert messages[1]['sources'][0]['answer'] == '完整原文'
    assert messages[-1]['sources'] == []
    assert '内部草稿' not in restored.text and 'faq-1' not in restored.text and 'keyword' not in restored.text
    events = parse_events(asyncio.run(post(app, {'message': '恢复后续聊', 'conversation_id': str(older)})).text)
    assert events[0][1]['conversation_id'] == str(older)
    assert repo.load_messages(older)[-1].content == '你好'


@pytest.mark.parametrize('suffix,status', [('999999',404), ('0',422), ('bad',422), ('18446744073709551616',422)])
def test_history_missing_or_invalid_id(factory, suffix, status):
    app = create_app(settings(), FakeModel(), factory, FakeKnowledgeSearch())
    assert get(app, '/v1/conversations/' + suffix + '/messages').status_code == status


@pytest.mark.parametrize('query', ['limit=0','limit=101','before=0','before=oops'])
def test_history_rejects_invalid_pagination(factory, query):
    app = create_app(settings(), FakeModel(), factory, FakeKnowledgeSearch())
    assert get(app, '/v1/conversations?' + query).status_code == 422


def test_document_preview_returns_actual_markdown_and_rejects_paths(factory):
    app = create_app(settings(), FakeModel(), factory, FakeKnowledgeSearch())
    response = get(app, '/v1/knowledge/documents/product-specs.md')
    assert response.status_code == 200
    payload = response.json()
    assert payload['filename'] == 'product-specs.md'
    assert '# 商品规格手册' in payload['text']
    assert 'MH-LP100' in payload['text'] and 'MH-W60' in payload['text']
    for name in ['.env','missing.md','%2e%2e%5c.env','%2e%2e%2f.env']:
        assert get(app, '/v1/knowledge/documents/' + name).status_code in (404,422)


def test_source_mapping_uses_exact_original_text_and_never_guesses(tmp_path):
    from app.chat_views import sources_from_tool
    (tmp_path / 'manual.md').write_text('# 手册\n\n## 型号 MH-W60\n\n6L，建议每周清洗。\n', encoding='utf-8')
    content = json.dumps({'matches': [{'question':'型号 MH-W60','answer':'6L，建议每周清洗。','category':'手册'}]})
    source = sources_from_tool(content, tmp_path)[0]
    assert (source['n'],source['source_file'],source['section_path'],source['source_line']) == (1,'manual.md','手册 / 型号 MH-W60',5)
    assert source['answer'] == '6L，建议每周清洗。'
    missing = sources_from_tool(json.dumps({'matches':[{'question':'猜测','answer':'不存在的规格','category':'手册'}]}), tmp_path)[0]
    assert missing['source_file'] is None and missing['section_path'] is None
    (tmp_path / 'copy.md').write_text('# 复制\n\n6L，建议每周清洗。\n', encoding='utf-8')
    assert sources_from_tool(content, tmp_path)[0]['source_file'] is None
    assert sources_from_tool('not json', tmp_path) == []


def test_faq_source_event_precedes_answer_and_history_keeps_snapshot(factory):
    from langchain_core.messages import AIMessage
    class Model(FakeModel):
        async def choose_tool(self, messages, tools):
            return AIMessage(content='', tool_calls=[{'id':'faq','name':'query_faq','args':{'keyword':'退货'}}])
    app = create_app(settings(), Model(), factory, FakeKnowledgeSearch())
    events = parse_events(asyncio.run(post(app, {'message':'退货'})).text)
    kinds = [kind for kind,_ in events]
    assert 'sources' in kinds
    assert kinds.index('sources') < kinds.index('token')
    snapshot = next(data['sources'] for kind,data in events if kind == 'sources')
    assert snapshot[0]['answer'] == '七天内申请'
    cid = events[0][1]['conversation_id']
    history = get(app, f'/v1/conversations/{cid}/messages').json()
    assert history['messages'][-1]['sources'][0]['answer'] == '七天内申请'


def test_invalid_source_entries_do_not_change_original_citation_numbers(tmp_path):
    from app.chat_views import sources_from_tool
    payload = {'matches': [None, {'question':'空答案','answer':'','category':'售后'},
                           {'question':'有效条目','answer':'原文条件','category':'售后'}]}
    result = sources_from_tool(json.dumps(payload), tmp_path)
    assert [(source['n'], source['answer']) for source in result] == [(3, '原文条件')]
