#!/usr/bin/env python3
"""Offline JSON Schema validation for the ADR-282 research gate.

The gate consumes and emits documents that are hashed into an attested artifact
index, so their shape is a security boundary rather than a convenience. Every
schema under ``schemas/`` is loaded from disk and registered under its own
``$id``; the registry refuses to resolve anything it was not given, so a
candidate cannot redirect validation at a network-hosted schema.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource

MANIFEST_SCHEMA = "research-manifest-v1.json"
RESULTS_SCHEMA = "research-results-v1.json"
REPORT_SCHEMA = "research-report-v1.json"
ARTIFACT_INDEX_SCHEMA = "research-artifact-index-v1.json"
OVERRIDE_SCHEMA = "research-override-v1.json"
BASE_GATE_SCHEMA = "research-base-gate-v1.json"
ATTESTATION_SUBJECT_SCHEMA = "research-attestation-subject-v1.json"
EMBEDDING_IDENTITY_SCHEMA = "embedding-space-identity-v1.json"


class SchemaValidationError(ValueError):
    """Raised when a document violates its declared JSON Schema."""


def schema_dir() -> Path:
    override = os.environ.get("RESEARCH_GATE_SCHEMA_DIR")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2] / "schemas"


def _deny_remote(uri: str) -> Resource:
    raise NoSuchResource(ref=uri)


@lru_cache(maxsize=None)
def _load_schema(name: str) -> dict[str, Any]:
    path = schema_dir() / name
    if not path.is_file():
        raise SchemaValidationError(f"schema {name} is missing from {schema_dir()}")
    contents = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contents, dict):
        raise SchemaValidationError(f"schema {name} must be a JSON object")
    return contents


@lru_cache(maxsize=1)
def _registry() -> Registry:
    resources = []
    for path in sorted(schema_dir().glob("*.json")):
        contents = _load_schema(path.name)
        resource = Resource.from_contents(contents)
        identifier = contents.get("$id")
        resources.append((identifier or path.resolve().as_uri(), resource))
    if not resources:
        raise SchemaValidationError(f"no schemas found in {schema_dir()}")
    return Registry(retrieve=_deny_remote).with_resources(resources)


@lru_cache(maxsize=None)
def _validator(name: str) -> Draft202012Validator:
    schema = _load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        registry=_registry(),
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def validate_document(document: Any, schema_name: str) -> None:
    """Validate ``document`` against ``schema_name`` or raise SchemaValidationError."""
    errors = sorted(_validator(schema_name).iter_errors(document), key=lambda item: list(item.absolute_path))
    if not errors:
        return
    first = errors[0]
    location = "/".join(str(part) for part in first.absolute_path) or "<document root>"
    detail = f"{schema_name}: {location}: {first.message}"
    if len(errors) > 1:
        detail += f" (and {len(errors) - 1} further schema violation(s))"
    raise SchemaValidationError(detail)


def known_schemas() -> list[str]:
    return sorted(path.name for path in schema_dir().glob("*.json"))
