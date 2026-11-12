"""Fail-closed release gate contracts. No network or GitHub mutations."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("release_supervisor", Path(__file__).parents[1] / "release_supervisor.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)
SHA = "a" * 40


def run(**updates):
    return dict({"head_sha": SHA, "path": ".github/workflows/python-portfolio.yml", "event": "push", "status": "completed", "conclusion": "success"}, **updates)


def test_exact_run_success():
    assert supervisor.check_run(run(), SHA, "python-portfolio.yml", "push")


@pytest.mark.parametrize("changes", [
    {"head_sha": "b" * 40}, {"path": ".github/workflows/other.yml"},
    {"event": "pull_request"}, {"conclusion": "failure"},
    {"conclusion": "cancelled"}, {"conclusion": "skipped"},
])
def test_run_gate_refuses_wrong_identity_or_failed_run(changes):
    with pytest.raises(supervisor.GateError):
        supervisor.check_run(run(**changes), SHA, "python-portfolio.yml", "push")


def test_pending_run_waits():
    assert supervisor.check_run(run(status="queued", conclusion=None), SHA, "python-portfolio.yml", "push") is False


@pytest.mark.parametrize("jobs", [[],
    [{"name": "publish", "status": "completed", "conclusion": "failure"}],
    [{"name": "publish", "status": "completed", "conclusion": "skipped"}],
    [{"name": "publish", "status": "in_progress", "conclusion": None}],
    [{"name": "publish", "status": "completed", "conclusion": "success"}] * 2,
])
def test_required_jobs_missing_failed_or_ambiguous_refused(jobs):
    with pytest.raises(supervisor.GateError):
        supervisor.check_jobs(jobs, {"publish"})


def test_all_platform_verifications_required():
    names = {"publish"} | supervisor.VERIFY
    jobs = [{"name": n, "status": "completed", "conclusion": "success"} for n in names]
    assert supervisor.check_jobs(jobs, names)
    with pytest.raises(supervisor.GateError):
        supervisor.check_jobs(jobs[:-1], names)


def test_pr_head_change_or_already_merged_refused():
    pr = {"head": {"sha": SHA}, "base": {"ref": "main", "repo": {"full_name": supervisor.REPO}}, "state": "open", "merged": False}
    supervisor.check_pr(pr, SHA)
    for changed in [dict(pr, head={"sha": "b" * 40}), dict(pr, merged=True), dict(pr, state="closed")]:
        with pytest.raises(supervisor.GateError):
            supervisor.check_pr(changed, SHA)
