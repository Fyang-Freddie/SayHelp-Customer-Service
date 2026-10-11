from pathlib import Path
import pytest
from app.ch04_ingest import load_documents

ROOT=Path(__file__).resolve().parents[1]

def test_current_corpus_has_only_three_knowledge_sources():
    docs=load_documents(ROOT/'eval/ch06/corpus.json')
    assert {d[0] for d in docs}=={'knowledge_db/returns-policy.md','knowledge_db/after-sales-manual.md','knowledge_db/product-specifications.md'}
    assert all(d[2].strip() for d in docs)

def test_historical_manifest_is_not_current_business_corpus():
    with pytest.raises(ValueError): load_documents(ROOT/'eval/ch04/corpus.json')


def test_build_cli_selects_new_collection_explicitly(capsys):
    from app.build_ch04_knowledge import main
    main(['--manifest',str(ROOT/'eval/ch06/corpus.json'),'--collection','knowledge_ch06','--dry-run'])
    import json
    report=json.loads(capsys.readouterr().out)
    assert report['documents']==3 and report['collection']=='knowledge_ch06'


def test_legacy_source_mapping_never_reads_business_documents(tmp_path):
    import json
    from app.chat_views import sources_from_tool
    (tmp_path/'orders.md').write_text('# Orders\n\n## Private\n\nprivate order data\n',encoding='utf-8')
    payload=json.dumps({'matches':[{'question':'Private','answer':'private order data','category':'order'}]})
    assert sources_from_tool(payload,tmp_path)[0]['source_file'] is None
