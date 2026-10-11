# Preserve custom and automatic vector identity in RvLite CLI persistence

Status: proposed for review

## Problem and evidence

CLI custom IDs are stored in a plain object. Assigning the literal ID __proto__ invokes an inherited setter, so insert succeeds but JSON persistence contains no record. Reads/deletes of absent constructor, toString and __proto__ identifiers also mistake inherited prototype properties for vectors. Twelve actual Node CLI subprocess tests reproduce ten failures and two passing controls on upstream07933730.

The automatic counter also starts at one without checking caller-supplied vec_1, so the next generated ID overwrites that record. Against the literal-key-only candidate e93e1033, three automatic allocation cases fail while thirteen controls pass. This related identity repair is grouped into the same PR rather than published as an additional quota item.

## Decision

Use defineProperty to create enumerable, writable, configurable own vector keys, including __proto__. Require own properties for point lookup and deletion, including the REPL's corresponding lookup. Keep the JSON format, arbitrary existing string IDs and last-write replacement behavior; do not ban these names or prohibit explicit replacement.

Skip occupied own IDs during automatic allocation and reject invalid/exhausted counters before changing persisted data. Preserve available gaps and explicit caller replacement. Exhausted automatic allocation does not prevent caller-supplied IDs. Runtime cost is proportional to the occupied sequential prefix traversed, with O(1) extra memory for individual allocation and bounded safe-integer allocation. Batch insertion first reserves all explicit IDs in O(n) time and memory, preventing an earlier automatic row from claiming a later explicit ID. Explicit duplicate IDs retain last-write semantics. The forward-batch regression fails against 5ca54203 while eighteen controls pass.

## Validation and limits

Tests round-trip reserved/ordinary IDs through real insert/get/delete commands and saved JSON, reject nonexistent inherited IDs without changing the file, cover batch insertion and preserve last-write replacement. Node24.18.0 and locked commander/chalk/ora dependencies execute the current JavaScript CLI. This does not claim WASM/SDK RVF changes, concurrent-writer guarantees or general prototype hardening of graph/triple tables. The final nineteen tests additionally cover mixed batches, generated-looking caller IDs, gaps, ordinary monotonic allocation and exhausted-counter refusal. Fixed tests, syntax/diff checks and independent exact-commit review are required before publication.
