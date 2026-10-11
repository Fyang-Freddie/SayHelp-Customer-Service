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
