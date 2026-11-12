#!/usr/bin/env python3
"""Reduce GitHub check/status responses to a deterministic red-base check set.

This repository routinely carries more than one page of check runs per commit,
so the caller must fetch with ``gh api --paginate``. Depending on the ``gh``
version that produces either one JSON object per page concatenated together or,
with ``--slurp``, a single array of page objects; both forms are accepted. Each
page's ``total_count`` is cross-checked against the entries actually seen, so a
truncated response fails closed instead of certifying a red base as green.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PASSING_CONCLUSIONS = {"success", "neutral", "skipped"}


class BaseHealthError(ValueError):
    """Raised when the base-health snapshot cannot be trusted."""


def load_pages(path: str | Path) -> list[dict[str, Any]]:
    """Parse concatenated JSON objects, a slurped array, or a single object."""
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    pages: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            value, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as error:
            raise BaseHealthError(f"{path}: malformed JSON at offset {index}: {error}") from error
        if isinstance(value, list):
            pages.extend(value)
        else:
            pages.append(value)
    if not pages:
        raise BaseHealthError(f"{path}: contained no JSON pages")
    for page in pages:
        if not isinstance(page, dict):
            raise BaseHealthError(f"{path}: expected page objects, found {type(page).__name__}")
    return pages


def collect_paginated(pages: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Concatenate ``key`` across pages, proving completeness against total_count."""
    entries: list[dict[str, Any]] = []
    totals: set[int] = set()
    for page in pages:
        items = page.get(key)
        if not isinstance(items, list):
            raise BaseHealthError(f"response page is missing the '{key}' array")
        entries.extend(items)
        total = page.get("total_count")
        if not isinstance(total, int) or isinstance(total, bool):
            raise BaseHealthError(
                f"'{key}' response page has no integer total_count; "
                "pagination completeness cannot be proven, refusing to certify the base"
            )
        totals.add(total)
    if len(totals) != 1:
        raise BaseHealthError(
            f"'{key}' total_count changed across pages ({sorted(totals)}); "
            "the base-health snapshot is not consistent, refusing to certify the base"
        )
    total = totals.pop()
    if len(entries) != total:
        raise BaseHealthError(
            f"'{key}' returned {len(entries)} of {total} entries; the response was truncated. "
            "Fetch with 'gh api --paginate' so a red base cannot be certified green."
        )
    return entries


def failed_base_checks(check_runs: dict | list[dict], statuses: dict | list[dict]) -> list[str]:
    check_pages = check_runs if isinstance(check_runs, list) else [check_runs]
    status_pages = statuses if isinstance(statuses, list) else [statuses]

    latest = {}
    for check in collect_paginated(check_pages, "check_runs"):
        name = str(check.get("name", "")).strip()
        if not name:
            continue
        key = (check.get("completed_at") or check.get("started_at") or "", int(check.get("id", 0)))
        if name not in latest or key > latest[name][0]:
            latest[name] = (key, check)
    failed = []
    for name, (_, check) in latest.items():
        status = check.get("status")
        conclusion = check.get("conclusion")
        if status != "completed" or conclusion not in PASSING_CONCLUSIONS:
            failed.append(f"base/check:{name}:{conclusion or status or 'unknown'}")

    latest_status = {}
    for status in collect_paginated(status_pages, "statuses"):
        context = str(status.get("context", "")).strip()
        if context and context not in latest_status:
            # The combined-status endpoint returns statuses newest first.
            latest_status[context] = status
    for context, status in latest_status.items():
        state = status.get("state")
        if state != "success":
            failed.append(f"base/status:{context}:{state or 'unknown'}")
    return sorted(set(failed))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-runs", required=True)
    parser.add_argument("--statuses", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        checks = load_pages(args.check_runs)
        statuses = load_pages(args.statuses)
        failed = failed_base_checks(checks, statuses)
    except BaseHealthError as error:
        print(f"base health: {error}", file=sys.stderr)
        return 2
    Path(args.output).write_text(json.dumps(failed, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
