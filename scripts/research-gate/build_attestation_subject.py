#!/usr/bin/env python3
"""Bind trusted check outcomes and artifact hashes to an immutable commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import schema_validate
from research_gate import FULL_SHA_RE, GateError, load_json, sha256_file, validate_base_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-index", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-gate", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--workflow-revision", required=True)
    parser.add_argument("--evaluator-version", required=True)
    parser.add_argument("--required-check", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not FULL_SHA_RE.fullmatch(args.commit):
        raise GateError("commit must be a full SHA")
    index = load_json(args.artifact_index)
    if index.get("candidate_commit") != args.commit:
        raise GateError("artifact index commit mismatch")
    if index.get("candidate_ref") != args.candidate_ref:
        raise GateError("artifact index candidate ref mismatch")
    base_gate = load_json(args.base_gate)
    validate_base_gate(base_gate, base_gate.get("base_sha", ""), args.commit)
    if any("=" not in check for check in args.required_check):
        raise GateError("required checks must use name=success")
    checks = dict(check.split("=", 1) for check in args.required_check)
    if not checks or any(result != "success" for result in checks.values()):
        raise GateError("every required check must be present and successful")
    envelope = {
        "schema_version": 1,
        "candidate_commit": args.commit,
        "candidate_ref": args.candidate_ref,
        "base_sha": base_gate["base_sha"],
        "base_gate_state": base_gate["state"],
        "override_sha256": base_gate["override_sha256"],
        "workflow_revision": args.workflow_revision,
        "manifest_sha256": sha256_file(args.manifest),
        "evaluator_version": args.evaluator_version,
        "artifact_index_sha256": sha256_file(args.artifact_index),
        "required_checks": checks,
    }
    schema_validate.validate_document(envelope, schema_validate.ATTESTATION_SUBJECT_SCHEMA)
    Path(args.output).write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
