import json
from importlib import import_module
import pytest

def api():
    try: return import_module('app.knowledge_policy')
    except ModuleNotFoundError: pytest.fail('Chapter 6 knowledge calibration missing')

def test_policy_and_faq_thresholds_are_selected_separately():
    rows=[{'split':'calibration','mode':mode,'score':score,'answerable':label,'gold_found':label,'error':None} for mode,score,label in [('policy',5.,True),('policy',1.,False),('faq',2.,True),('faq',-4.,False)]]
    result=api().calibrate_knowledge(rows,{'corpus':'v1'})
    assert 1<result['modes']['policy']['threshold']<=5
    assert -4<result['modes']['faq']['threshold']<=2

def test_acceptance_cannot_change_knowledge_thresholds():
    with pytest.raises(ValueError): api().calibrate_knowledge([{'split':'acceptance'}],{})

def test_missing_or_stale_knowledge_artifact_fails_explicitly(tmp_path):
    path=tmp_path/'policy.json'
    with pytest.raises(Exception): api().load_policy(path,{'corpus':'new'},'policy')
    path.write_text(json.dumps({'version':1,'fingerprints':{'corpus':'old'},'modes':{}}),encoding='utf-8')
    with pytest.raises(Exception): api().load_policy(path,{'corpus':'new'},'policy')


def test_generation_window_changes_preserve_independent_knowledge_budget(tmp_path):
    from types import SimpleNamespace
    from app.config import Settings
    corpus=SimpleNamespace(corpus_digest='same')
    frozen=api().knowledge_fingerprints(corpus,Settings('x','same','x'))
    path=tmp_path/'frozen.json'
    path.write_text(json.dumps({'version':1,'fingerprints':frozen,
        'modes':{'faq':{'enabled':True,'threshold':6.1}}}),encoding='utf8')
    current=api().knowledge_fingerprints(corpus,Settings('x','same','x',
        context_token_budget=18000,response_token_reserve=2000,layered_context=True))
    assert current==frozen
    assert api().load_policy(path,current,'faq').thresholds['hybrid_rerank']==6.1
    for key in ('corpus_digest','query_model','prompt_digest','generation_k','context_token_budget','response_token_reserve'):
        with pytest.raises(Exception):
            api().load_policy(path,current|{key:'changed'},'faq')
