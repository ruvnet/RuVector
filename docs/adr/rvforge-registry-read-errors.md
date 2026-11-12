# Preserve registry read failures

Status: proposed

## Context

Registry readers return empty history or an absent tree head for every filesystem error. If an existing index, log, or witness chain cannot be read, callers can treat an established registry as a new one and lose its lineage. Malformed tree-head JSON is also reported as absent.

## Decision

Only ENOENT means an absent file. Other read failures raise the stable registry error and stop the operation. Apply the same rule to the build receipt reader because verification must not mistake inaccessible evidence for an empty chain. Malformed tree heads fail explicitly; no automatic reconstruction is attempted.

## Validation and consequences

Three regression tests fail on unchanged upstream using real directories at expected file paths. Existing missing-file tests remain successful. Tests cover the release append caller, log and witness readers, tree-head corruption, and build receipt reads. Full TypeScript package tests, build, and typecheck are the local gates. Filesystem errors remain actionable instead of becoming synthetic first releases or empty verification histories.
