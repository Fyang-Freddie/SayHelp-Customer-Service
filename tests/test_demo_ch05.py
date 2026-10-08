"""CLI acceptance: real runtime/graph, isolated scripted inputs, no provider by default."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode', ['bare', 'workflow'])
@pytest.mark.parametrize('case,minimum_tools', [('logistics', 1), ('multi-step', 2),
    ('policy', 0), ('complaint', 0), ('chitchat', 0), ('weak-evidence', 0), ('missing-info', 0)])
def test_demo_is_standalone_simulation_with_measured_runtime_counts(mode, case, minimum_tools, tmp_path):
    process = subprocess.run([sys.executable, str(ROOT/'scripts/demo_ch05.py'),
        '--mode', mode, '--case', case], cwd=tmp_path, capture_output=True, text=True, encoding='utf-8')
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout.splitlines()[-1])
    assert result['model_source'] == 'simulation' and result['business_data'] == 'seeded random simulation'
    assert result['logical_model_requests'] == 0 and result['ticket_writes'] == 0
    assert minimum_tools <= result['tool_calls'] <= 6 and result['model_calls'] <= 6
    assert result['answer'] and result['final_stream_chunks'] >= (case not in {'complaint','chitchat','weak-evidence'} or mode == 'bare')
    if case == 'multi-step':
        assert result['tool_names'] == ['query_order', 'query_logistics']
    if mode == 'workflow':
        assert result['node_count'] == len(result['nodes']) > 0
        if case == 'weak-evidence':
            assert result['model_calls'] == result['tool_calls'] == 0
            assert result['stop_reason'] == 'weak_evidence'
        if case == 'missing-info':
            assert result['tool_calls'] == 0 and '订单号' in result['answer']
