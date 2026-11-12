import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('trigger', Path(__file__).parents[1] / 'release_requested.py')
trigger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trigger)

@pytest.mark.parametrize('event,ref,previous,expected', [
    ('push','refs/heads/main',None,True),
    ('push','refs/heads/main','0.0.9',True),
    ('push','refs/heads/main','0.1.0',False),
    ('push','refs/heads/feature','0.0.9',False),
    ('pull_request','refs/pull/1185/merge',None,False),
    ('workflow_dispatch','refs/heads/main',None,False),
    ('push','refs/tags/python-portfolio-v0.1.0','0.1.0',True),
    ('push','refs/tags/python-portfolio-v9.9.9',None,False),
])
def test_release_scope(event,ref,previous,expected):
    assert trigger.requested(event,ref,{},'0.1.0',previous) is expected
