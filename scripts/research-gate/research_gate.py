#!/usr/bin/env python3
"""Trusted validation and evaluation for ADR-282 research candidates.

Apart from the pinned ``jsonschema`` validator installed from
``scripts/research-gate/requirements.txt``, this module uses only the Python
standard library, so the trusted gate runs without executing dependency
lifecycle scripts from a candidate. Every document the gate reads or writes is
checked against its schema in ``schemas/`` first; the hand-rolled checks below
then enforce the cross-field invariants a schema cannot express.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import sys
import unicodedata
from pathlib import Path
from typing import Any

import schema_validate
from schema_validate import SchemaValidationError

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
IDENTITY_FIELDS = {
    "schema_version",
    "provider",
    "model_id",
    "model_artifact_sha256",
    "model_graph_sha256",
    "tokenizer_sha256",
    "prompt_template_sha256",
    "pooling_strategy",
    "normalize",
    "truncation_tokens",
    "output_dimension",
    "output_dtype",
    "runtime_revision",
    "distance_metric",
    "role_policy",
    "prefix_policy",
    "prefix_policy_version",
}
MEMORY_COMPONENTS = {
    "encoded_payload_bytes",
    "index_graph_bytes",
    "codebooks_quantizer_bytes",
    "container_bookkeeping_bytes",
    "allocator_overhead_bytes",
    "temporary_build_bytes",
    "temporary_query_bytes",
    "process_peak_rss_bytes",
}
MEMORY_PROTOCOL_FIELDS = {
    "version",
    "isolation",
    "baseline_subtracted",
    "warmup_operations",
    "sampling_interval_ms",
    "allocator",
    "cgroup_limit_bytes",
    "page_size_bytes",
    "process_count",
    "thread_count",
    "child_process_policy",
    "filesystem_cache_included",
    "cache_state",
    "kernel",
    "alternating_order",
}
# Files that are legitimately absent from the artifact index because a different
# cryptographic mechanism binds them: an index cannot contain its own digest, and
# the attestation subject carries artifact_index_sha256 and is itself signed.
# Nothing else may sit in the evidence root without a recorded hash. A candidate
# that needs to retain an extra artifact adds it to the trusted index, not here.
UNINDEXED_BY_DESIGN = frozenset({"artifact-index.json"})
PROMOTION_UNINDEXED = UNINDEXED_BY_DESIGN | {"attestation-subject.json"}
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


class GateError(ValueError):
    """Raised when a candidate violates a research-gate invariant."""


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise GateError(f"{path}: JSON root must be an object")
    return value


def canonical_json(value: Any) -> bytes:
    # EmbeddingSpaceIdentity contains integers, booleans, strings, and objects,
    # for which this is RFC 8785-compatible. NaN and infinities are forbidden.
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedding_space_id(identity: dict[str, Any]) -> str:
    validate_embedding_identity(identity)
    return sha256_bytes(b"ruvector.embedding-space.v1\0" + canonical_json(identity))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _schema(document: Any, schema_name: str) -> None:
    """Fail closed against the declared JSON Schema before any semantic check."""
    try:
        schema_validate.validate_document(document, schema_name)
    except SchemaValidationError as error:
        raise GateError(str(error)) from error


def _sha(value: Any, name: str) -> None:
    _require(isinstance(value, str) and bool(SHA256_RE.fullmatch(value)), f"{name} must be lowercase SHA-256")


def validate_embedding_identity(identity: dict[str, Any]) -> None:
    _schema(identity, schema_validate.EMBEDDING_IDENTITY_SCHEMA)
    missing = IDENTITY_FIELDS - set(identity)
    _require(not missing, f"embedding identity missing fields: {sorted(missing)}")
    _require(set(identity) == IDENTITY_FIELDS,
             f"embedding identity has unknown fields: {sorted(set(identity) - IDENTITY_FIELDS)}")
    def require_nfc(value: Any, path: str) -> None:
        if isinstance(value, str):
            _require(unicodedata.normalize("NFC", value) == value, f"{path} must be NFC-normalized")
        elif isinstance(value, dict):
            for key, nested in value.items():
                _require(unicodedata.normalize("NFC", key) == key, f"{path} key must be NFC-normalized")
                require_nfc(nested, f"{path}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                require_nfc(nested, f"{path}[{index}]")

    require_nfc(identity, "embedding identity")
    _require(identity["schema_version"] == 1, "embedding identity schema_version must be 1")
    for field in (
        "model_artifact_sha256",
        "model_graph_sha256",
        "tokenizer_sha256",
        "prompt_template_sha256",
    ):
        _sha(identity[field], f"embedding identity {field}")
    _require(identity["role_policy"] in {"symmetric", "asymmetric"}, "invalid role_policy")
    _require(identity["prefix_policy"] in {"none", "required", "query-recommended", "custom"},
             "invalid prefix_policy")
    _require(identity["output_dtype"] in {"f32", "f16", "bf16", "i8", "u8"}, "invalid output_dtype")
    _require(identity["distance_metric"] in {"cosine", "dot", "euclidean", "manhattan"},
             "invalid distance_metric")
    _require(isinstance(identity["prefix_policy_version"], int) and identity["prefix_policy_version"] >= 1,
             "prefix_policy_version must be a positive integer")
    _require(isinstance(identity["output_dimension"], int) and identity["output_dimension"] > 0,
             "output_dimension must be positive")
    _require(isinstance(identity["truncation_tokens"], int) and identity["truncation_tokens"] > 0,
             "truncation_tokens must be positive")


def _validate_decision_rule(rule: dict[str, Any]) -> None:
    required = {
        "primary_metric",
        "minimum_meaningful_effect",
        "expected_direction",
        "alpha",
        "comparison",
        "sample_size_or_power_rationale",
        "outcome_rules",
    }
    _require(not (required - set(rule)), f"decision_rule missing: {sorted(required - set(rule))}")
    _require(rule["expected_direction"] in {"greater", "less"}, "expected_direction must be greater or less")
    _require(rule["comparison"] == "paired", "current evaluator requires paired comparison")
    _require(isinstance(rule["minimum_meaningful_effect"], (int, float))
             and rule["minimum_meaningful_effect"] > 0, "minimum meaningful effect must be positive")
    _require(rule["alpha"] == 0.05, "evaluator v1 supports preregistered alpha=0.05 only")
    expected_rules = {
        "pass": "ci_lower_at_or_above_minimum_meaningful_effect",
        "fail": "ci_upper_at_or_below_zero",
        "inconclusive": "otherwise",
    }
    _require(rule["outcome_rules"] == expected_rules, "outcome_rules must match evaluator v1")
    _require(bool(str(rule["sample_size_or_power_rationale"]).strip()), "power rationale cannot be empty")


def _validate_memory(manifest: dict[str, Any]) -> None:
    accounting = manifest.get("memory_accounting")
    _require(isinstance(accounting, dict), "memory_accounting must be an object")
    _require(not (MEMORY_COMPONENTS - set(accounting.get("components", []))),
             f"memory components missing: {sorted(MEMORY_COMPONENTS - set(accounting.get('components', [])))}")
    protocol = accounting.get("peak_rss_protocol")
    _require(isinstance(protocol, dict), "peak_rss_protocol must be an object")
    _require(not (MEMORY_PROTOCOL_FIELDS - set(protocol)),
             f"peak RSS protocol missing: {sorted(MEMORY_PROTOCOL_FIELDS - set(protocol))}")
    _require(0 < protocol["sampling_interval_ms"] <= 10, "RSS sampling interval must be <=10ms")
    _require(protocol["baseline_subtracted"] is True, "RSS baseline subtraction is required")
    _require(protocol["alternating_order"] is True, "paired runs must use alternating order")


def validate_manifest(manifest: dict[str, Any], prior: dict[str, Any] | None = None) -> None:
    _schema(manifest, schema_validate.MANIFEST_SCHEMA)
    _require(manifest.get("schema_version") == 1, "manifest schema_version must be 1")
    _require(isinstance(manifest.get("commit"), str) and bool(FULL_SHA_RE.fullmatch(manifest["commit"])),
             "commit must be a full lowercase 40-character SHA")
    _require(isinstance(manifest.get("revision"), int) and manifest["revision"] >= 1,
             "revision must be a positive integer")
    _require(manifest.get("phase") in {"exploration", "confirmation"}, "invalid phase")
    _require(bool(str(manifest.get("claim", "")).strip()), "claim must be falsifiable and non-empty")
    _require(bool(str(manifest.get("independent_variable", "")).strip()),
             "independent_variable is required")
    _validate_decision_rule(manifest.get("decision_rule", {}))

    exploration = manifest.get("exploration_seeds")
    confirmation = manifest.get("confirmation_seeds")
    _require(isinstance(exploration, list) and len(exploration) >= 5 and len(set(exploration)) == len(exploration),
             "at least five unique exploration seeds are required")
    _require(isinstance(confirmation, list) and len(confirmation) >= 5 and len(set(confirmation)) == len(confirmation),
             "at least five unique confirmation seeds are required")
    _require(set(exploration).isdisjoint(confirmation), "confirmation seeds must be fresh")

    datasets = manifest.get("datasets")
    _require(isinstance(datasets, list) and datasets, "at least one real dataset is required")
    for index, dataset in enumerate(datasets):
        for field in ("name", "source", "sampling"):
            _require(bool(str(dataset.get(field, "")).strip()), f"datasets[{index}].{field} is required")
        _sha(dataset.get("sha256"), f"datasets[{index}].sha256")
        _require(dataset.get("kind") == "real", "production claims require kind=real")

    embedding = manifest.get("embedding_space", {})
    identity = embedding.get("identity")
    _require(isinstance(identity, dict), "complete embedding_space.identity is required")
    expected_identity = embedding_space_id(identity)
    _require(embedding.get("embedding_space_id") == expected_identity,
             "embedding_space_id does not match canonical identity")
    _require(embedding.get("query_api") == "embed_query", "query_api must be embed_query")
    _require(embedding.get("passage_api") == "embed_passage", "passage_api must be embed_passage")

    topology = manifest.get("topology", {})
    _require(topology.get("claimed") == topology.get("implemented"),
             "claimed and implemented production topology must match")
    _require(isinstance(topology.get("configuration"), dict), "topology configuration is required")

    budget = manifest.get("budget", {})
    _require(budget.get("primary_resource") in {
        "resident_memory_bytes", "on_disk_bytes", "query_latency_ns",
        "search_effort", "build_time_ns", "cpu_seconds",
    }, "invalid primary constrained resource")
    _require(isinstance(budget.get("tolerance"), (int, float)) and 0 <= budget["tolerance"] <= 0.05,
             "budget tolerance must be between 0 and 5%")
    _require(isinstance(budget.get("secondary_resources"), list), "secondary_resources must be an array")
    _require(isinstance(budget.get("indivisible_allocation_unit"), (int, float))
             and budget["indivisible_allocation_unit"] >= 0,
             "indivisible_allocation_unit is required")

    _validate_memory(manifest)
    _require(bool(str(manifest.get("evaluator_version", "")).strip()), "immutable evaluator_version is required")
    _require(manifest.get("artifact_retention_class") in {"candidate", "accepted", "publication"},
             "invalid artifact retention class")
    selection = manifest.get("selection", {})
    _require(isinstance(selection.get("candidate_family"), str), "selection candidate_family is required")
    _require(selection.get("confirmation_error_control") in {"family-wise", "fdr", "single-candidate"},
             "confirmation error control is required")

    if prior is not None:
        prior_embedding = prior.get("embedding_space", {})
        same_corpus = [x.get("sha256") for x in prior.get("datasets", [])] == [
            x.get("sha256") for x in datasets
        ]
        identity_changed = prior_embedding.get("embedding_space_id") != expected_identity
        if same_corpus and identity_changed:
            _require(manifest["revision"] > prior.get("revision", 0),
                     "embedding identity change requires a new experimental revision")
            _require(manifest["phase"] == "confirmation",
                     "embedding identity change requires a new confirmation run")
            old_seeds = set(prior.get("exploration_seeds", [])) | set(prior.get("confirmation_seeds", []))
            _require(old_seeds.isdisjoint(confirmation),
                     "embedding identity change requires fresh confirmation seeds")


def _paired_interval(differences: list[float]) -> tuple[float, float, float, float]:
    _require(len(differences) >= 5, "at least five paired observations are required")
    mean = statistics.fmean(differences)
    sd = statistics.stdev(differences)
    critical = T_CRITICAL_95.get(len(differences) - 1, 1.96)
    margin = critical * sd / math.sqrt(len(differences))
    return mean, sd, mean - margin, mean + margin


def evaluate(manifest: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    validate_manifest(manifest)
    _schema(results, schema_validate.RESULTS_SCHEMA)
    _require(manifest.get("phase") == "confirmation", "only confirmation manifests are promotable")
    _require(results.get("schema_version") == 1, "results schema_version must be 1")
    _require(results.get("manifest_revision") == manifest["revision"], "result revision mismatch")
    _require(results.get("candidate_commit") == manifest["commit"], "result commit mismatch")
    manifest_hash = sha256_bytes(canonical_json(manifest))
    _require(results.get("manifest_sha256") == manifest_hash, "result manifest hash mismatch")
    _require(results.get("embedding_space_id") == manifest["embedding_space"]["embedding_space_id"],
             "result embedding identity mismatch")
    _require(results.get("phase") == "confirmation", "only confirmation results are promotable")

    runs = results.get("runs")
    _require(isinstance(runs, list), "runs must be an array")
    expected_seeds = manifest["confirmation_seeds"]
    actual_seeds = [run.get("seed") for run in runs]
    _require(actual_seeds == expected_seeds, "runs must exactly match predetermined confirmation seeds in order")

    metric = manifest["decision_rule"]["primary_metric"]
    resource = manifest["budget"]["primary_resource"]
    tolerance = manifest["budget"]["tolerance"]
    unit = manifest["budget"]["indivisible_allocation_unit"]
    differences: list[float] = []
    budget_deltas: list[float] = []
    for run in runs:
        control = run.get("control", {})
        treatment = run.get("treatment", {})
        for arm_name, arm in (("control", control), ("treatment", treatment)):
            _require(metric in arm.get("metrics", {}), f"{arm_name} missing primary metric {metric}")
            _require(resource in arm.get("resources", {}), f"{arm_name} missing primary resource {resource}")
            _require(not (MEMORY_COMPONENTS - set(arm.get("memory", {}))),
                     f"{arm_name} missing full memory breakdown")
        control_budget = float(control["resources"][resource])
        treatment_budget = float(treatment["resources"][resource])
        allowed = max(abs(control_budget) * tolerance, float(unit))
        delta = abs(treatment_budget - control_budget)
        _require(delta <= allowed,
                 f"seed {run['seed']} exceeds primary-resource budget: delta={delta}, allowed={allowed}")
        budget_deltas.append(delta)
        raw = float(treatment["metrics"][metric]) - float(control["metrics"][metric])
        differences.append(raw if manifest["decision_rule"]["expected_direction"] == "greater" else -raw)

        selection = run.get("selection")
        if selection is not None:
            _require(selection.get("selected_count") == selection.get("configured_count"),
                     "selection count changed, possibly due to percentile ties")
            _require(bool(selection.get("deterministic_tie_breaker")),
                     "selection requires a deterministic tie breaker")

    mean, sd, lower, upper = _paired_interval(differences)
    meaningful = float(manifest["decision_rule"]["minimum_meaningful_effect"])
    if lower >= meaningful:
        outcome = "pass"
    elif upper <= 0:
        outcome = "fail"
    else:
        outcome = "inconclusive"

    summary = {
        "schema_version": 1,
        "outcome": outcome,
        "primary_metric": metric,
        "paired_effect_mean": mean,
        "paired_effect_standard_deviation": sd,
        "confidence_interval_95": [lower, upper],
        "minimum_meaningful_effect": meaningful,
        "worst_seed_effect": min(differences),
        "maximum_primary_budget_delta": max(budget_deltas, default=0),
        "seed_count": len(differences),
        "manifest_sha256": manifest_hash,
        "results_sha256": sha256_bytes(canonical_json(results)),
        "embedding_space_id": manifest["embedding_space"]["embedding_space_id"],
    }
    declared = results.get("declared_summary")
    if declared is not None:
        _require(declared == summary, "headline/declared summary does not match raw results")
    return summary


def validate_override(value: dict[str, Any], base_sha: str, head_sha: str,
                      failed_checks: list[str], now: dt.datetime | None = None) -> None:
    now = now or dt.datetime.now(dt.timezone.utc)
    _schema(value, schema_validate.OVERRIDE_SCHEMA)
    _require(value.get("schema_version") == 1, "override schema_version must be 1")
    _require(value.get("scope") == "red-base-only", "override scope must be red-base-only")
    _require(value.get("approval_source") == "github-environment:research-gate-override",
             "override must come from protected environment approval")
    _require(value.get("base_sha") == base_sha and value.get("head_sha") == head_sha,
             "override is not scoped to exact base/head SHAs")
    _require(sorted(value.get("failed_checks", [])) == sorted(failed_checks),
             "override failed-check set changed")
    _require(failed_checks and all(check.startswith("base/") for check in failed_checks),
             "override may authorize red-base checks only")
    _require(value.get("approver_permission") in {"maintain", "admin"}, "approver lacks maintain/admin")
    _require(value.get("approver_in_research_gate_codeowners") is True,
             "approver is not in research-gate CODEOWNERS group")
    approved = dt.datetime.fromisoformat(value["approved_at"].replace("Z", "+00:00"))
    expires = dt.datetime.fromisoformat(value["expires_at"].replace("Z", "+00:00"))
    _require(expires > now, "override has expired")
    _require(expires - approved <= dt.timedelta(hours=72), "override validity exceeds 72 hours")
    _require(bool(str(value.get("rationale", "")).strip()), "override rationale is required")


def validate_base_gate(value: dict[str, Any], base_sha: str, head_sha: str) -> None:
    _schema(value, schema_validate.BASE_GATE_SCHEMA)
    _require(value.get("schema_version") == 1, "base gate schema_version must be 1")
    _require(value.get("base_sha") == base_sha and value.get("head_sha") == head_sha,
             "base gate is not scoped to exact base/head SHAs")
    state = value.get("state")
    _require(state in {"green", "authorized-red"}, "invalid base gate state")
    failed = value.get("failed_checks")
    _require(isinstance(failed, list), "base gate failed_checks must be an array")
    if state == "green":
        _require(not failed and value.get("override_sha256") is None,
                 "green base gate cannot carry failures or an override")
    else:
        _require(failed and all(check.startswith("base/") for check in failed),
                 "authorized-red gate can contain base failures only")
        _sha(value.get("override_sha256"), "base gate override_sha256")


def _same_number(reported: Any, evaluated: float, name: str) -> None:
    _require(isinstance(reported, (int, float)) and not isinstance(reported, bool),
             f"report {name} must be a number")
    _require(math.isclose(float(reported), evaluated, rel_tol=1e-9, abs_tol=1e-12),
             f"report {name} ({reported}) does not match the trusted evaluation ({evaluated})")


def validate_report(report: dict[str, Any], manifest: dict[str, Any],
                    evaluation: dict[str, Any]) -> None:
    """Bind a candidate-authored report to the trusted evaluation of hashed raw results.

    ADR-282 acceptance criterion 13: headline values are verified against the
    hashed raw artifacts rather than accepted as transcribed.
    """
    _schema(report, schema_validate.REPORT_SCHEMA)
    _require(report["candidate_commit"] == manifest["commit"], "report commit mismatch")
    _require(report["manifest_revision"] == manifest["revision"], "report manifest revision mismatch")
    _require(report["claim"] == manifest["claim"], "report claim differs from the preregistered claim")
    _require(report["manifest_sha256"] == evaluation["manifest_sha256"], "report manifest hash mismatch")
    _require(report["results_sha256"] == evaluation["results_sha256"],
             "report is not derived from the evaluated raw results")
    _require(report["primary_metric"] == evaluation["primary_metric"], "report primary metric mismatch")
    _require(report["outcome"] == evaluation["outcome"],
             f"report claims outcome {report['outcome']} but the evaluator found {evaluation['outcome']}")

    headline = report["headline"]
    _require(headline["seed_count"] == evaluation["seed_count"], "report seed count mismatch")
    for field in (
        "paired_effect_mean",
        "paired_effect_standard_deviation",
        "minimum_meaningful_effect",
        "worst_seed_effect",
        "maximum_primary_budget_delta",
    ):
        _same_number(headline[field], float(evaluation[field]), f"headline.{field}")
    for index, bound in enumerate(("lower", "upper")):
        _same_number(headline["confidence_interval_95"][index],
                     float(evaluation["confidence_interval_95"][index]),
                     f"headline.confidence_interval_95 {bound}")

    effects = report["per_seed_effects"]
    _require(len(effects) == evaluation["seed_count"],
             "report must include one effect per confirmation seed")
    _same_number(min(effects), float(evaluation["worst_seed_effect"]), "per_seed_effects minimum")


def _present_paths(root_path: Path) -> set[str]:
    """Every file and symlink under root, relative and POSIX-normalized.

    Symlinked directories are recorded but never descended into, so a candidate
    cannot hide payloads behind a link or escape the evidence root.
    """
    present: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
        base = Path(dirpath)
        for name in filenames:
            present.add((base / name).relative_to(root_path).as_posix())
        for name in list(dirnames):
            entry = base / name
            if entry.is_symlink():
                present.add(entry.relative_to(root_path).as_posix())
                dirnames.remove(name)
    return present


def validate_artifact_index(index: dict[str, Any], root: str | Path,
                            allow_unindexed: frozenset[str] = UNINDEXED_BY_DESIGN) -> None:
    _schema(index, schema_validate.ARTIFACT_INDEX_SCHEMA)
    _require(index.get("schema_version") == 1, "artifact index schema_version must be 1")
    _require(index.get("retention_class") in {"candidate", "accepted", "publication"},
             "invalid retention class")
    minimum = {"candidate": 365, "accepted": 2555, "publication": -1}[index["retention_class"]]
    _require(index.get("retention_days") == minimum, "retention period does not meet immutable policy")
    root_path = Path(root).resolve()
    artifacts = index.get("artifacts", [])
    _require(isinstance(artifacts, list) and artifacts, "artifact index cannot be empty")
    indexed: set[str] = set()
    for item in artifacts:
        rel = Path(item["path"])
        _require(not rel.is_absolute() and ".." not in rel.parts, "artifact path escapes root")
        target = (root_path / rel).resolve()
        _require(target.is_relative_to(root_path), "artifact path escapes root")
        _require(target.is_file() and not target.is_symlink(), f"artifact missing or symlink: {rel}")
        _require(target.stat().st_size == item["size_bytes"], f"artifact size changed: {rel}")
        _require(sha256_file(target) == item["sha256"], f"artifact hash changed: {rel}")
        indexed.add(rel.as_posix())

    # The index must account for the whole evidence tree. Anything else would be
    # uploaded and attested without a recorded digest.
    unindexed = sorted(_present_paths(root_path) - indexed - set(allow_unindexed))
    _require(not unindexed,
             "evidence root contains file(s) missing from the artifact index: "
             f"{unindexed}; every retained artifact must carry a recorded digest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate_parser = sub.add_parser("validate-manifest")
    validate_parser.add_argument("manifest")
    validate_parser.add_argument("--prior-manifest")
    validate_parser.add_argument("--expect-sha")
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("manifest")
    eval_parser.add_argument("results")
    eval_parser.add_argument("--output", required=True)
    report_parser = sub.add_parser("validate-report")
    report_parser.add_argument("report")
    report_parser.add_argument("--manifest", required=True)
    report_parser.add_argument("--evaluation", required=True)
    index_parser = sub.add_parser("validate-index")
    index_parser.add_argument("index")
    index_parser.add_argument("--root", required=True)
    id_parser = sub.add_parser("embedding-space-id")
    id_parser.add_argument("identity")
    override_parser = sub.add_parser("validate-override")
    override_parser.add_argument("override")
    override_parser.add_argument("--base-sha", required=True)
    override_parser.add_argument("--head-sha", required=True)
    override_parser.add_argument("--failed-checks", required=True)
    base_gate_parser = sub.add_parser("validate-base-gate")
    base_gate_parser.add_argument("base_gate")
    base_gate_parser.add_argument("--base-sha", required=True)
    base_gate_parser.add_argument("--head-sha", required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate-manifest":
            manifest = load_json(args.manifest)
            prior = load_json(args.prior_manifest) if args.prior_manifest else None
            validate_manifest(manifest, prior)
            if args.expect_sha:
                _require(manifest["commit"] == args.expect_sha, "manifest commit differs from candidate SHA")
        elif args.command == "evaluate":
            summary = evaluate(load_json(args.manifest), load_json(args.results))
            Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _require(summary["outcome"] == "pass", f"confirmation outcome is {summary['outcome']}")
        elif args.command == "validate-report":
            validate_report(load_json(args.report), load_json(args.manifest), load_json(args.evaluation))
        elif args.command == "validate-index":
            validate_artifact_index(load_json(args.index), args.root)
        elif args.command == "embedding-space-id":
            print(embedding_space_id(load_json(args.identity)))
        elif args.command == "validate-override":
            failed_checks = json.loads(Path(args.failed_checks).read_text(encoding="utf-8"))
            _require(isinstance(failed_checks, list), "failed checks file must contain an array")
            validate_override(load_json(args.override), args.base_sha, args.head_sha, failed_checks)
        elif args.command == "validate-base-gate":
            validate_base_gate(load_json(args.base_gate), args.base_sha, args.head_sha)
        return 0
    except (GateError, KeyError, TypeError, ValueError) as error:
        print(f"research gate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
