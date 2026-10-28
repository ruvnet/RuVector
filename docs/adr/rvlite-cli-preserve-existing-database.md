# Preserve existing databases on RvLite CLI load errors

Status: proposed for review

## Problem

loadState catches every read/parse failure and continues with a fresh in-memory database. A valid insert against truncated JSON then overwrites the existing bytes, and read-only stats/export claim an empty database. The actual Node CLI reproduces five failing producer scenarios and three passing controls on upstream07933730.

## Decision

Read directly and treat only ENOENT as a missing database. Propagate other filesystem errors and JSON/shape errors to the existing command failure handlers. Existing corrupt files remain available for deliberate recovery. No new init/repair command, schema migration or persistence format is introduced; explicit init continues its existing behavior.

## Validation and limits

Real subprocess tests use locked commander12.1.0/chalk5.6.2/ora8.2.0 dependencies and Node24.18.0. Regressions cover corrupt insert/delete/stats/export, actual directory reads, and permission-denied reads. Missing-path creation and valid append/stats are controls. The tests exercise the CLI's current JavaScript persistence implementation, not WASM/SDK RVF storage. No concurrent-writer or crash-atomicity claim. Fixed tests and independent exact-commit review must pass before publication.
