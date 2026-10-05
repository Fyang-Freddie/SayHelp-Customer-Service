"""Exact parser positions and manifest-only ingestion, using disposable MySQL."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from app.knowledge_chunking import chunk_markdown
from app.knowledge_db import KnowledgeChunk
from test_db import database_url, sessions


def module(name):
    try:
        return importlib.import_module('app.' + name)
    except ModuleNotFoundError:
        pytest.fail('Chapter 4 component missing: ' + name)


@pytest.mark.parametrize('body,limit,want,legacy', [
    ('# 手册\n相同正文。\n\n## 第二节\n相同正文。', 800, [(2,2),(5,5)], [2,5]),
    ('# 手册\n甲句跨\n行完成。\n乙句。', 9, [(2,3),(4,4)], [2,4]),
    ('# 政策\n> [!IMPORTANT]\n> 甲。\n> 乙。', 800, [(3,4)], [3]),
    ('# 手册\n```text\n# literal\n\ncode\n```', 800, [(2,6)], [2]),
    ('# 手册\n| A | B |\n| --- | --- |\n| 1 | x |\n| 2 | y |', 35, [(2,4),(2,5)], [4,5]),
])
def test_parser_tracks_real_ranges_including_repeated_headers(body, limit, want, legacy):
    chunks = chunk_markdown(body, content_type='manual', max_chars=limit, overlap_chars=0)
    assert all(hasattr(c, 'source_start_line') and hasattr(c, 'source_end_line') for c in chunks), 'Exact source ranges missing'
    assert [(c.source_start_line,c.source_end_line) for c in chunks] == want
    assert [c.source_line for c in chunks] == legacy


def test_overlap_range_includes_actual_previous_sentence():
    chunks = chunk_markdown('# 手册\n甲句。\n乙句。\n丙句。', content_type='manual', max_chars=7, overlap_chars=3)
    assert all(hasattr(c, 'source_end_line') for c in chunks), 'Source end positions missing'
    assert [(c.source_start_line,c.source_end_line) for c in chunks] == [(2,3),(3,4)]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    def ingest_call(session_factory, manifest):
        ingest = module('ch04_ingest')
        monkeypatch.setattr(ingest, 'ROOT', tmp_path)
        return ingest.ingest_corpus(session_factory, manifest)
    ingest = SimpleNamespace(ingest_corpus=ingest_call)
    directory = tmp_path / 'knowledge_db'
    directory.mkdir()
    originals = {
        'returns-policy.md': ('policy', '# 退货退款政策\n## 无理由退货\n自签收起7天内，商品完好可申请退货。'),
        'after-sales-manual.md': ('manual', '# 售后手册\n## 保修说明\n电子商品保修12个月，人为损坏除外。'),
        'product-specs.md': ('manual', '# 商品规格手册\n## 智能猫砂盆 Lite(型号 MH-LP50)\nLite 无 App 联网功能。\n## 自动饮水机(型号 MH-W20)\n水箱容量2L。'),
        'product-faq.md': ('faq', '# 商品与购物 FAQ\n## 运费怎么算\n订单满99元包邮，偏远地区另计。'),
    }
    docs = []
    for name,(kind,body) in originals.items():
        (directory/name).write_text(body,encoding='utf-8')
        docs.append({'source_file': 'knowledge_db/'+name, 'content_type': kind})
    manifest = tmp_path / 'corpus.json'
    manifest.write_text(json.dumps({'documents': docs}, ensure_ascii=False),encoding='utf-8')
    return ingest, manifest, directory


def test_real_manifest_does_not_copy_seed_faq_or_duplicate_queries(sessions, corpus):
    ingest, manifest, directory = corpus
    first = ingest.ingest_corpus(sessions, manifest)
    second = ingest.ingest_corpus(sessions, manifest)
    assert len(first.files) == 4 and len(first.chunks) == 5
    assert first.chunks == second.chunks and first.corpus_digest == second.corpus_digest
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(KnowledgeChunk)) == 5
    products = [c for c in first.chunks.values() if c.source_file == 'knowledge_db/product-specs.md']
    assert {c.product_category for c in products} == {'猫砂盆','饮水机'}
    assert all(c.source_start_line and c.source_end_line and len(c.source_digest)==64 for c in first.chunks.values())
    assert {c.source_file for c in first.chunks.values()} == {'knowledge_db/'+name for name in ('returns-policy.md','after-sales-manual.md','product-specs.md','product-faq.md')}


def test_new_file_version_preserves_old_rows_but_excludes_them_from_current_corpus(sessions, corpus):
    ingest, manifest, directory = corpus
    first = ingest.ingest_corpus(sessions, manifest)
    path = directory/'product-specs.md'
    path.write_text(path.read_text(encoding='utf-8').replace('2L','3L'),encoding='utf-8')
    second = ingest.ingest_corpus(sessions, manifest)
    old = {c.id for c in first.chunks.values() if c.source_file.endswith('product-specs.md')}
    new = {c.id for c in second.chunks.values() if c.source_file.endswith('product-specs.md')}
    assert old.isdisjoint(new) and len(second.chunks)==5
    assert first.corpus_digest != second.corpus_digest
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(KnowledgeChunk)) == 7
        assert session.scalar(select(KnowledgeChunk.answer).where(KnowledgeChunk.id.in_(old), KnowledgeChunk.questions.contains('MH-W20'))) == '水箱容量2L。'


def test_unique_legacy_done_chain_is_backfilled_without_resetting_old_status(sessions, corpus):
    ingest, manifest, directory = corpus
    with sessions.begin() as session:
        row = KnowledgeChunk(category='退货退款政策', questions='无理由退货', answer='自签收起7天内，商品完好可申请退货。', section_path='退货退款政策 / 无理由退货', content_type='policy', vectorize_status='done')
        session.add(row)
        session.flush()
        old_id = row.id
    snapshot = ingest.ingest_corpus(sessions, manifest)
    assert old_id in snapshot.chunks
    assert snapshot.chunks[old_id].source_file == 'knowledge_db/returns-policy.md'
    with sessions() as session:
        assert session.get(KnowledgeChunk, old_id).vectorize_status == 'done'
        assert session.scalar(select(func.count()).select_from(KnowledgeChunk)) == 5


@pytest.mark.parametrize('bad_path', ['../../.env', 'D:/private.md', 'knowledge_db/unknown.md'])
def test_manifest_rejects_paths_outside_the_explicit_real_corpus(sessions, corpus, bad_path):
    ingest, manifest, directory = corpus
    data = json.loads(manifest.read_text(encoding='utf-8'))
    data['documents'][0]['source_file'] = bad_path
    manifest.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError, match='[Mm]anifest|[Ss]ource|[Cc]orpus'):
        ingest.ingest_corpus(sessions, manifest)
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(KnowledgeChunk)) == 0


def test_product_overview_is_not_tagged_as_aftersales(sessions, corpus):
    ingest,manifest,directory=corpus
    path=directory/'product-specs.md'
    path.write_text(path.read_text(encoding='utf-8').replace('# 商品规格手册\n','# 商品规格手册\n本手册收录型号与规格。\n'),encoding='utf-8')
    snapshot=ingest.ingest_corpus(sessions,manifest)
    overview=next(c for c in snapshot.chunks.values() if c.source_file.endswith('product-specs.md') and c.question=='商品规格手册')
    assert overview.product_category=='商品'
