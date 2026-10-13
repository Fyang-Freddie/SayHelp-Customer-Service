import pytest
from scripts import demo_ch06 as demo


def test_demo_requires_explicit_live_before_loading_credentials(monkeypatch,capsys):
    def forbidden(*args,**kwargs):raise AssertionError('credentials read')
    monkeypatch.setattr(demo,'dotenv_values',forbidden)
    assert demo.main(['--case','order-picker'])==0
    assert 'dry_run' in capsys.readouterr().out


def test_demo_missing_env_fails_without_starting_server(tmp_path,capsys):
    assert demo.main(['--live','--env-file',str(tmp_path/'absent.env')])==2
    assert 'configuration' in capsys.readouterr().err


def test_demo_unknown_case_is_rejected():
    with pytest.raises(SystemExit):demo.main(['--case','invented'])
