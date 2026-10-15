# Enforce numeric vector and configured dimension contracts in RvLite CLI

Status: proposed for review

## Problem and evidence

The real JavaScript CLI accepts wrong-dimensional, string, null, empty, nonfinite or object coordinates and persists them as vectors. Search then silently computes invalid distances, including null JSON scores. Against upstream07933730 ten producer cases fail with one valid configured/unconfigured control passing. Initialization separately accepts zero/negative, fractional, partial-text and infinite dimensions; five additional cases fail against the vector-guard candidate while eleven controls pass.

## Decision

Validate nonempty finite numeric arrays before any insert or query arithmetic. When a database has configured dimensions, enforce that exact length; preserve flexible dimensions for the existing unconfigured-file workflow. Preflight all batch vectors before insertion. Require a positive safe-integer initialization dimension before writing a database. Do not silently coerce coordinates or truncate configuration inputs.

## Validation and limits

Sixteen actual Node24 CLI subprocess tests, syntax/diff checks and independent exact-commit review are required. This changes the JavaScript CLI and preserves ordinary finite vectors including zero vectors; it is not a WASM SDK build or a migration repairing older malformed databases. Validation is O(d) per vector, alongside existing O(d) norm computation, and O(n*d) for batch preflight. No promise is made about concurrent writers or general JSON state schemas.

## Changelog

Reject malformed vector coordinates, configured dimension mismatches and invalid initialization dimensions before persistence or query arithmetic.
