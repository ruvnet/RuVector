#!/usr/bin/env python3
"""Verify downloaded trusted evidence without importing candidate code."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from research_gate import (
    FULL_SHA_RE,
    PROMOTION_UNINDEXED,
    GateError,
    load_json,
    sha256_file,
    validate_artifact_index,
    validate_base_gate,
    validate_manifest,
    validate_override,
)

REF_RE = re.compile(r"^research/(?:candidate|nightly)/[A-Za-z0-9._/-]+$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.evidence).resolve()
    subject = load_json(root / "attestation-subject.json")
    index = load_json(root / "artifact-index.json")
    manifest = load_json(root / "research-manifest.json")
    commit = subject.get("candidate_commit", "")
    ref = subject.get("candidate_ref", "")
    if not FULL_SHA_RE.fullmatch(commit):
        raise GateError("attested candidate commit is invalid")
    if not REF_RE.fullmatch(ref):
        raise GateError("attested candidate ref is outside research namespaces")
    if index.get("candidate_commit") != commit or index.get("candidate_ref") != ref:
        raise GateError("attestation and artifact index candidate mismatch")
    if manifest.get("commit") != commit:
        raise GateError("attested manifest commit mismatch")
    validate_manifest(manifest)
    if sha256_file(root / "artifact-index.json") != subject.get("artifact_index_sha256"):
        raise GateError("attested artifact-index digest mismatch")
    if sha256_file(root / "research-manifest.json") != subject.get("manifest_sha256"):
        raise GateError("attested manifest digest mismatch")
    base_gate = load_json(root / "base-gate.json")
    validate_base_gate(base_gate, subject.get("base_sha", ""), commit)
    if base_gate.get("state") != subject.get("base_gate_state"):
        raise GateError("attested base gate state mismatch")
    if base_gate.get("override_sha256") != subject.get("override_sha256"):
        raise GateError("attested override digest mismatch")
    if base_gate["state"] == "authorized-red":
        override_path = root / "override.json"
        if not override_path.is_file() or sha256_file(override_path) != base_gate["override_sha256"]:
            raise GateError("authorized red base is missing its exact override")
        validate_override(
            load_json(override_path),
            base_gate["base_sha"],
            commit,
            base_gate["failed_checks"],
        )
    checks = subject.get("required_checks", {})
    if not checks or any(value != "success" for value in checks.values()):
        raise GateError("attestation does not bind a complete green check set")
    # The downloaded attested bundle also carries the signed attestation subject.
    validate_artifact_index(index, root, allow_unindexed=PROMOTION_UNINDEXED)
    Path(args.output).write_text(
        f"candidate_sha={commit}\ncandidate_ref={ref}\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
