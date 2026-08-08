#!/usr/bin/env python3
"""Build an immutable ADR-282 artifact index without following symlinks."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from research_gate import FULL_SHA_RE, GateError, sha256_file

RETENTION_DAYS = {"candidate": 365, "accepted": 2555, "publication": -1}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--workflow-revision", required=True)
    parser.add_argument("--evaluator-version", required=True)
    parser.add_argument("--retention-class", choices=RETENTION_DAYS, required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    if not FULL_SHA_RE.fullmatch(args.commit):
        raise GateError("commit must be a full lowercase SHA")
    root = Path(args.root).resolve()
    entries = []
    for value in sorted(set(args.files)):
        path = Path(value)
        target = (root / path).resolve()
        if path.is_absolute() or ".." in path.parts or not target.is_relative_to(root):
            raise GateError(f"artifact escapes root: {value}")
        if target.is_symlink() or not target.is_file():
            raise GateError(f"artifact must be a regular non-symlink file: {value}")
        entries.append({
            "path": path.as_posix(),
            "sha256": sha256_file(target),
            "size_bytes": target.stat().st_size,
        })

    index = {
        "schema_version": 1,
        "candidate_commit": args.commit,
        "candidate_ref": args.candidate_ref,
        "workflow_revision": args.workflow_revision,
        "evaluator_version": args.evaluator_version,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "retention_class": args.retention_class,
        "retention_days": RETENTION_DAYS[args.retention_class],
        "immutable_storage_required": True,
        "artifacts": entries,
    }
    output = Path(args.output)
    output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
