"""Controlled clock exercises real CLI/native benchmark reporting, not performance claims."""
import json
from unittest.mock import patch
import pytest
from click.testing import CliRunner
from ruvector.cli import main

@pytest.mark.parametrize('queries,expected_ms',[(2,2),(3,3),(10,10),(100,99)])
def test_p99_reports_nearest_rank_including_small_query_counts(queries,expected_ms):
    clock=[0.,0.1]
    for i in range(queries):
        start=1.+i
        clock.extend([start,start+(i+1)/1000])
    with patch('time.perf_counter', side_effect=clock):
        result=CliRunner().invoke(main,['benchmark','-n','8','--dim','2','--queries',str(queries),'--k','2','--recall-k','2','--json'])
    assert result.exit_code == 0, repr(result.exception)
    data=json.loads(result.output)
    assert data['p99_ms'] == pytest.approx(expected_ms)
    assert data['p99_ms'] >= data['p50_ms']
