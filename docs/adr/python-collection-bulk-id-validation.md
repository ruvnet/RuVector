# Validate and normalize caller IDs before Python collection bulk build

Status: proposed for review

## Problem and evidence

The Collection.from_vectors(ids=...) contract takes integer IDs, but fractional and boolean IDs are accepted. HNSW then exposes non-integer string IDs that fail integer conversion at query time, while RaBitQ silently coerces fractions to unsigned integers. Four actual constructor producer tests reproduce missing rejection. Valid NumPy integer IDs search correctly, but both backends fail to save because next_id remains a NumPy scalar unsupported by the JSON encoder; two further producer cases reproduce this independently.

## Decision

Require non-boolean Python or NumPy integer IDs before native construction, then normalize accepted values to Python int. Preserve duplicate detection and the existing backend-specific range checks. Do not reinterpret strings, floats or booleans as IDs. This adds O(n) validation and normalization alongside the existing ID materialization.

## Validation and limits

Six actual native-collection producer tests cover invalid input and valid NumPy-integer persistence, with full Python suite validation and independent exact-commit review required. This is Python input-type and scalar normalization, not a change to native ID allocation or storage formats. The separately published PR1156 repairs RaBitQ external IDs across native save/load; that issue is not duplicated here. NumPy-ID save/load tests verify persisted count and vectors on both backends and stable IDs on HNSW; combined validation can additionally use the corrected RaBitQ artifact.

## Changelog

Reject fractional/boolean bulk IDs and persist valid NumPy integer IDs without JSON scalar errors.
