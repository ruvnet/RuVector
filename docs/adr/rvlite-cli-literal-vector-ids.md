# Preserve literal vector IDs in RvLite CLI persistence

Status: proposed for review

## Problem and evidence

CLI custom IDs are stored in a plain object. Assigning the literal ID __proto__ invokes an inherited setter, so insert succeeds but JSON persistence contains no record. Reads/deletes of absent constructor, toString and __proto__ identifiers also mistake inherited prototype properties for vectors. Twelve actual Node CLI subprocess tests reproduce ten failures and two passing controls on upstream07933730.

## Decision

Use defineProperty to create enumerable, writable, configurable own vector keys, including __proto__. Require own properties for point lookup and deletion, including the REPL's corresponding lookup. Keep the JSON format, arbitrary existing string IDs and last-write replacement behavior; do not ban these names or change ID generation.

## Validation and limits

Tests round-trip reserved/ordinary IDs through real insert/get/delete commands and saved JSON, reject nonexistent inherited IDs without changing the file, cover batch insertion and preserve last-write replacement. Node24.18.0 and locked commander/chalk/ora dependencies execute the current JavaScript CLI. This does not claim WASM/SDK RVF changes, concurrent-writer guarantees or general prototype hardening of graph/triple tables. Fixed tests, syntax/diff checks and independent exact-commit review are required before publication.
