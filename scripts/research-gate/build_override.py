#!/usr/bin/env python3
"""Create an exact-SHA, short-lived override from environment review history."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from research_gate import GateError, validate_override


def build_override(
    approvals: list[dict],
    permission: str,
    codeowners_value: str,
    base_sha: str,
    head_sha: str,
    failed_checks: list[str],
    rationale: str,
    expires_at: str,
    now: dt.datetime | None = None,
) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    approved = [
        item for item in approvals
        if item.get("state") == "approved"
        and any(env.get("name") == "research-gate-override" for env in item.get("environments", []))
    ]
    if len(approved) != 1:
        raise GateError("exactly one recorded research-gate-override environment approval is required")
    approver = approved[0].get("user", {}).get("login")
    if not approver:
        raise GateError("environment approval has no recorded reviewer")
    codeowners = {item.strip().lstrip("@").lower() for item in codeowners_value.split(",") if item.strip()}
    if approver.lower() not in codeowners:
        raise GateError("recorded environment reviewer is not in RESEARCH_GATE_CODEOWNERS")
    value = {
        "schema_version": 1,
        "scope": "red-base-only",
        "approver": approver,
        "approval_source": "github-environment:research-gate-override",
        "approver_permission": permission,
        "approver_in_research_gate_codeowners": True,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "failed_checks": failed_checks,
        "rationale": rationale,
        "approved_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": expires_at,
    }
    validate_override(value, base_sha, head_sha, failed_checks, now)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approvals", required=True)
    parser.add_argument("--permission", required=True)
    parser.add_argument("--codeowners", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--failed-checks", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--expires-at", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    approvals = json.loads(Path(args.approvals).read_text(encoding="utf-8"))
    now = dt.datetime.now(dt.timezone.utc)
    value = build_override(
        approvals,
        args.permission,
        args.codeowners,
        args.base_sha,
        args.head_sha,
        json.loads(Path(args.failed_checks).read_text(encoding="utf-8")),
        args.rationale,
        args.expires_at,
        now,
    )
    Path(args.output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
