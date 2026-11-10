#!/usr/bin/env python3
"""One-shot, fail-closed supervisor for an explicitly authorized two-stage release.

Uses existing gh authentication. Never reads credentials or retrieves action logs.
A lock and immutable state path prevent duplicate launches. Failed or uncertain
mutations require operator inspection; this program never retries uploads/merges.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request

REPO = "ruvnet/RuVector"
API = f"repos/{REPO}"
VERIFY = {f"verify-published ({os_name})" for os_name in
          ("ubuntu-latest", "macos-latest", "windows-latest")}


class GateError(RuntimeError):
    """A sanitized release gate failure, safe to write to the receipt."""


def check_run(run, head, workflow, event=None):
    if run.get("head_sha") != head:
        raise GateError("workflow head differs from authorized commit")
    if run.get("path", "").split("@")[0] != f".github/workflows/{workflow}":
        raise GateError("unexpected workflow identity")
    if event and run.get("event") != event:
        raise GateError("unexpected workflow event")
    if run.get("status") != "completed":
        return False
    if run.get("conclusion") != "success":
        raise GateError("workflow did not succeed")
    return True


def check_jobs(jobs, required):
    for name in required:
        found = [job for job in jobs if job.get("name") == name]
        if len(found) != 1 or found[0].get("status") != "completed" or found[0].get("conclusion") != "success":
            raise GateError(f"required job missing or unsuccessful: {name}")
    return True


def check_pr(pr, head):
    if pr.get("head", {}).get("sha") != head:
        raise GateError("companion PR head changed")
    if pr.get("base", {}).get("ref") != "main" or pr.get("base", {}).get("repo", {}).get("full_name", "").lower() != REPO.lower():
        raise GateError("unexpected companion PR destination")
    if pr.get("state") != "open" or pr.get("merged"):
        raise GateError("companion PR is no longer open")


def gh(*args, body=None):
    try:
        result = subprocess.run(["gh", *args], input=json.dumps(body) if body is not None else None,
                                text=True, capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateError("gh execution unavailable or timed out; inspect remote state before retry") from exc
    if result.returncode:
        raise GateError("GitHub operation refused; inspect protections or authentication")
    try:
        return json.loads(result.stdout) if result.stdout.strip() else None
    except ValueError as exc:
        raise GateError("unexpected GitHub response") from exc


def api(path, method="GET", body=None):
    args = ["api", "--method", method, path]
    if body is not None:
        args += ["--input", "-"]
    return gh(*args, body=body)


def jobs_for(run):
    pages = gh("api", "--paginate", "--slurp", f"{API}/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs?per_page=100")
    return [job for page in pages for job in page["jobs"]]


def published(package, version):
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise GateError("PyPI metadata HTTP error") from exc
    except (OSError, ValueError) as exc:
        raise GateError("PyPI metadata unavailable") from exc
    return data.get("info", {}).get("version") == version and bool(data.get("urls")) and all(not f.get("yanked", False) for f in data["urls"])


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.deadline = time.monotonic() + args.timeout_seconds
        self.state = {"status": "running", "stage": "native", "native_run_id": args.native_run_id,
                      "native_head_sha": args.native_head_sha, "companion_pr": args.companion_pr,
                      "companion_head_sha": args.companion_head_sha, "native_version": args.native_version,
                      "portfolio_version": args.portfolio_version, "rerun_used": False}

    def save(self, **updates):
        self.state.update(updates)
        self.state["updated_at_unix"] = int(time.time())
        temp = self.args.state_path.with_suffix(self.args.state_path.suffix + ".tmp")
        temp.write_text(json.dumps(self.state, indent=2) + "\n")
        os.chmod(temp, 0o600)
        temp.replace(self.args.state_path)

    def pause(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise GateError("release supervisor deadline exceeded")
        time.sleep(min(30, remaining))

    def latest(self, head, event):
        response = api(f"{API}/actions/workflows/python-portfolio.yml/runs?head_sha={head}&event={event}&per_page=100")
        runs = [r for r in response["workflow_runs"] if r.get("head_sha") == head and r.get("event") == event]
        return max(runs, key=lambda r: r["id"]) if runs else None

    def run(self):
        self.save()
        while True:
            run = api(f"{API}/actions/runs/{self.args.native_run_id}")
            if check_run(run, self.args.native_head_sha, "python-wheels.yml"):
                check_jobs(jobs_for(run), {"Publish to PyPI"})
                break
            self.pause()
        self.save(stage="runtime_visibility")
        while not published("ruvector", self.args.native_version):
            self.pause()
        self.save(stage="companion_validation")
        previous_attempt = None
        while True:
            pr = api(f"{API}/pulls/{self.args.companion_pr}")
            check_pr(pr, self.args.companion_head_sha)
            run = self.latest(self.args.companion_head_sha, "pull_request")
            if run is None:
                self.pause()
                continue
            # A rerun can take time to increment its attempt counter. Never
            # interpret the stale failed attempt as the rerun's outcome.
            identity = (run["id"], run["run_attempt"])
            if identity == previous_attempt:
                self.pause()
                continue
            if run.get("status") != "completed":
                self.pause()
                continue
            try:
                check_run(run, self.args.companion_head_sha, "python-portfolio.yml", "pull_request")
                check_jobs(jobs_for(run), {"test-build", "native-audit"})
            except GateError:
                if self.state["rerun_used"]:
                    raise
                # PR workflows cannot upload. Rerun once only after the
                # exact native runtime was observed on PyPI.
                self.save(rerun_used=True, companion_run_id=run["id"])
                previous_attempt = identity
                api(f"{API}/actions/runs/{run['id']}/rerun", "POST")
                self.pause()
                continue
            self.save(companion_run_id=run["id"], companion_attempt=run["run_attempt"])
            break
        # Fetch the exact authorized manifest; never use a mutable checkout.
        encoded = api(f"{API}/contents/python/portfolio/manifest.json?ref={self.args.companion_head_sha}")
        manifest = json.loads(base64.b64decode(encoded["content"]))
        if manifest.get("runtime") != self.args.native_version or manifest.get("version") != self.args.portfolio_version:
            raise GateError("manifest versions differ from authorized release")
        packages = manifest.get("packages")
        if not isinstance(packages, list) or not packages or len(set(packages)) != len(packages) or any(not isinstance(p, str) or not re.fullmatch(r"ruvector-[a-z0-9-]+", p) for p in packages):
            raise GateError("invalid package manifest")
        self.save(stage="merge", packages=packages)
        pr = api(f"{API}/pulls/{self.args.companion_pr}")
        check_pr(pr, self.args.companion_head_sha)
        if pr.get("draft"):
            # gh pr ready emits prose; suppress it rather than parse or log it.
            result = subprocess.run(["gh", "pr", "ready", str(self.args.companion_pr), "--repo", REPO], capture_output=True, timeout=60)
            if result.returncode:
                raise GateError("marking companion PR ready failed")
        check_pr(api(f"{API}/pulls/{self.args.companion_pr}"), self.args.companion_head_sha)
        merged = api(f"{API}/pulls/{self.args.companion_pr}/merge", "PUT",
                     {"sha": self.args.companion_head_sha, "merge_method": "squash"})
        if not merged.get("merged") or not re.fullmatch(r"[0-9a-f]{40}", merged.get("sha", "")):
            raise GateError("merge not confirmed; inspect remote state")
        merge_sha = merged["sha"]
        self.save(stage="portfolio_publication", merge_sha=merge_sha)
        while True:
            run = self.latest(merge_sha, "push")
            if run and check_run(run, merge_sha, "python-portfolio.yml", "push"):
                check_jobs(jobs_for(run), {"test-build", "native-audit", "publish"} | VERIFY)
                self.save(portfolio_run_id=run["id"])
                break
            self.pause()
        self.save(stage="portfolio_visibility")
        while not all(published(p, self.args.portfolio_version) for p in packages):
            self.pause()
        self.save(status="success", stage="complete")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-run-id", type=int, required=True)
    parser.add_argument("--native-head-sha", required=True)
    parser.add_argument("--companion-pr", type=int, required=True)
    parser.add_argument("--companion-head-sha", required=True)
    parser.add_argument("--native-version", required=True)
    parser.add_argument("--portfolio-version", required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 14400 or args.native_run_id <= 0 or args.companion_pr <= 0:
        parser.error("positive IDs and timeout of 1 through 14400 seconds required")
    for sha in (args.native_head_sha, args.companion_head_sha):
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            parser.error("full lowercase commit SHA required")
    for version in (args.native_version, args.portfolio_version):
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
            parser.error("exact stable X.Y.Z versions required")
    return args


def main():
    args = parse_args()
    args.state_path = args.state_path.resolve()
    args.state_path.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(str(args.state_path) + ".lock")
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError:
        raise SystemExit("supervisor lock already exists; inspect existing execution")
    supervisor = Supervisor(args)
    try:
        if args.state_path.exists():
            raise SystemExit("state path already exists; inspect prior receipt")
        if not hasattr(signal, "SIGALRM"):
            raise GateError("supervisor requires POSIX alarm support for its hard deadline")
        def expired(signum, frame):
            raise GateError("release supervisor hard deadline exceeded")
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(args.timeout_seconds)
        supervisor.run()
    except Exception as exc:
        # Never serialize subprocess output, HTTP responses, or arbitrary
        # exception strings which could contain authentication material.
        message = str(exc) if isinstance(exc, GateError) else "unexpected supervisor failure; inspect code and remote state"
        supervisor.save(status="failed", error=message)
        raise SystemExit(message) from None
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        lock.rmdir()


if __name__ == "__main__":
    main()
