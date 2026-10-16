import importlib.util,json
from pathlib import Path


def test_profiles_override_all_peak_inputs_and_share_24_frozen_turns():
    path=Path('scripts/demo_ch07.py')
    spec=importlib.util.spec_from_file_location('demo_ch07',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    demo=module.profile_environment('demo')
    assert demo['MODEL_CONTEXT_WINDOW']=='18000'
    assert [demo[k] for k in ('MAX_OUTPUT_TOKENS','MAX_USER_INPUT_TOKENS','MAX_AGENT_STEPS','TOOL_RESULT_MAX_TOKENS','RERANK_TOP_K')]==['2000','2000','3','1200','5']
    assert module.profile_environment('default')['MODEL_CONTEXT_WINDOW']=='65536'
    cases=json.loads(Path('eval/ch07/dialogue-cases.json').read_text(encoding='utf-8-sig'))
    assert len(cases['turns'])==24 and cases['expected_order'] in cases['turns'][0]
    assert cases['expected_order'] not in cases['turns'][-1]


def test_sse_events_preserve_chinese_and_done_identity():
    spec=importlib.util.spec_from_file_location('demo_ch07',Path('scripts/demo_ch07.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert list(module.events('event: token\ndata: {"text":"原文"}\n\nevent: done\ndata: {"conversation_id":"1"}\n'))==[
        ('token',{'text':'原文'}),('done',{'conversation_id':'1'})]


def test_live_preflight_rejects_remaining_allowance_before_requests(tmp_path):
    import pytest
    from app.ch07_evaluation import EvaluationBudget,EvaluationLimitExceeded
    from scripts.demo_ch07 import preflight
    from app.config import Settings
    meter=EvaluationBudget(tmp_path/'meter.sqlite')
    settings=Settings('x','x','x',layered_context=True,context_token_budget=18000,response_token_reserve=2000)
    for _ in range(25):meter.reserve('jev','jev_dialogue')
    with pytest.raises(EvaluationLimitExceeded):preflight(settings,meter,24)


def test_initialization_failure_retains_report(tmp_path):
    import asyncio
    import pytest
    from types import SimpleNamespace
    from scripts.demo_ch07 import run
    output=tmp_path/'failed.json'
    args=SimpleNamespace(profile='demo',live=True,dry_run=False,init_db=False,
        env_file=str(tmp_path/'missing.env'),output=str(output))
    with pytest.raises(FileNotFoundError):asyncio.run(run(args))
    assert json.loads(output.read_text(encoding='utf8'))['error_type']=='FileNotFoundError'
