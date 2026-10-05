"""Ground-truth anchors bind exact current source chunks in real MySQL."""
import importlib
import json
from dataclasses import replace

import pytest
from sqlalchemy.orm import undefer
from app.knowledge_db import KnowledgeChunk
from test_db import database_url,sessions
from test_ch04_ingest import corpus


def api(): return importlib.import_module('app.evaluation.dataset')


def case(id_='A01',**changes):
    row={'eval_id':id_,'bucket':'A_policy','difficulty':'easy','query':'签收后多久可以申请无理由退货？','filters':{},'answerable':True,
         'ground_truth':'签收起7天内，商品完好可申请。','required_facts':['签收起7天内','商品完好'],'forbidden_claims':['任意时间都能退'],
         'evidence':[{'source_file':'knowledge_db/returns-policy.md','section_path':'退货退款政策 / 无理由退货','evidence_quote':'自签收起7天内，商品完好可申请退货。'}]}
    return {**row,**changes}


def write(path,cases,files=None,split='test'):
    path.write_text(json.dumps({'version':1,'split':split,'source_digests':files or {},'cases':cases},ensure_ascii=False),encoding='utf-8')
    return path


def test_duplicate_id_is_rejected(tmp_path):
    path=write(tmp_path/'cases.json',[case(),case()])
    with pytest.raises(ValueError,match='[Dd]uplicate'): api().load_cases(path)


@pytest.mark.parametrize('patch',[
    {'eval_id':'X'*17},{'bucket':'bad'},{'difficulty':'impossible'}, {'filters':{'unsafe':'x'}},
    {'answerable':'true'},{'answerable':False}, {'required_facts':[]}, {'evidence':[]},
    {'evidence':[{'source_file':'../../.env','section_path':'x','evidence_quote':'x'}]},
],ids=['id','bucket','difficulty','filter','bool','unknown_gold','empty_facts','empty_gold','path'])
def test_case_schema_rejects_invalid_annotation(tmp_path,patch):
    with pytest.raises(ValueError): api().load_cases(write(tmp_path/'cases.json',[case(**patch)]))


def test_cross_split_id_or_identical_query_is_rejected(tmp_path):
    write(tmp_path/'calibration.json',[case('K01')],split='calibration')
    path=write(tmp_path/'test.json',[case('A01')])
    with pytest.raises(ValueError,match='[Ll]eak|[Ss]plit|[Oo]verlap'): api().load_cases(path)
    write(path,[case('K01',query='不同的问题')])
    with pytest.raises(ValueError,match='[Ll]eak|[Ss]plit|[Oo]verlap'): api().load_cases(path)


def test_exact_gold_binds_current_mysql_row(sessions,corpus,tmp_path):
    ingest,manifest,_=corpus
    snapshot=ingest.ingest_corpus(sessions,manifest)
    cases=api().load_cases(write(tmp_path/'cases.json',[case()],snapshot.files))
    bound=api().bind_gold(cases,snapshot,sessions)
    expected=next(c.id for c in snapshot.chunks.values() if c.source_file.endswith('returns-policy.md'))
    assert bound[0].gold_ids==[expected] and bound[0].corpus_digest==snapshot.corpus_digest


@pytest.mark.parametrize('kind',['missing','ambiguous','mysql_changed','source_digest'])
def test_gold_refuses_missing_ambiguous_or_changed_source(sessions,corpus,tmp_path,kind):
    ingest,manifest,_=corpus
    snapshot=ingest.ingest_corpus(sessions,manifest)
    files=dict(snapshot.files)
    annotation=case()
    expected=next(c for c in snapshot.chunks.values() if c.source_file.endswith('returns-policy.md'))
    if kind=='missing': annotation['evidence'][0]['evidence_quote']='原文完全不存在的句子'
    if kind=='ambiguous':
        chunks={**snapshot.chunks,999:replace(expected,id=999)}
        snapshot=replace(snapshot,chunks=chunks)
    if kind=='mysql_changed':
        with sessions.begin() as session: session.get(KnowledgeChunk,expected.id).answer='已改变的来源正文'
    if kind=='source_digest': files[expected.source_file]='0'*64
    cases=api().load_cases(write(tmp_path/'cases.json',[annotation],files))
    with pytest.raises(ValueError): api().bind_gold(cases,snapshot,sessions)


def test_unknown_annotation_has_no_gold(sessions,corpus,tmp_path):
    ingest,manifest,_=corpus
    snapshot=ingest.ingest_corpus(sessions,manifest)
    cases=api().load_cases(write(tmp_path/'cases.json',[case('D01',bucket='D_unknown',answerable=False,required_facts=[],evidence=[],ground_truth='明确拒答：原文没有此信息。')],snapshot.files))
    assert api().bind_gold(cases,snapshot,sessions)[0].gold_ids==[]


def test_frozen_real_dataset_balance_and_all_anchors(sessions):
    from pathlib import Path
    from app.ch04_ingest import ingest_corpus,ROOT
    dataset=api()
    calibration=dataset.load_cases(ROOT/'eval/ch04/calibration.json')
    test=dataset.load_cases(ROOT/'eval/ch04/test.json')
    dataset.validate_splits(calibration,test)
    assert len(calibration)==12 and len(test)==60
    snapshot=ingest_corpus(sessions,ROOT/'eval/ch04/corpus.json')
    bound=dataset.bind_gold([*calibration,*test],snapshot,sessions)
    assert len(bound)==72 and all(bool(row.gold_ids)==row.case.answerable for row in bound)
    assert any('MH-LP50' in row.case.query and 'App' in row.case.query for row in bound)
    assert any(row.case.bucket=='E_multi' and len(row.gold_ids)>=2 for row in bound)
