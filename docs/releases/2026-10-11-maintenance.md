# RuVector maintenance candidate — 11 October 2026

**Status: NOT READY. Unpublished draft.** Candidate CI is pending. No npm or
crates.io version has been published by this maintenance work. The attached
archive is source code, not a prebuilt runtime or a production-approved model.

## What changes in simple terms

- Saving and reopening an INT8 TypeSafe decision system now keeps its answers
  stable. Previously, processing different groups of examples together changed
  the underlying numbers. Native inference now processes each text independently.
  Its model identity includes `#ort-singleton-v1` so frozen-head consumers can
  distinguish the new feature policy from old batched embeddings. Runtime build
  changes still require separate qualification.
- The language-model server opens its port before telling clients it is ready.
  A busy port produces a clear failure instead of a misleading success message.
- Ubuntu CI uses a hash-verified ONNX runtime compatible with that runner.
- The graph rebuild test checks whether reads and writes can really proceed
  during rebuilding. An unrelated scheduling delay no longer fails a 200ms gate.
- Nine compatible npm dependency updates clear the production audit findings.

## Evidence and limits

Implementation: https://github.com/ruvnet/RuVector/pull/1180
Tracking: #1177 (saved-decision parity), #1178 (CI/server/graph), #1179 (dependencies).

The INT8 regression fails on the old code and passes after the fix. The server
suite passes 7 tests. Graph checks pass 10 tests; deliberately holding the write
lock during rebuilding causes the new regression to fail. An existing ignored
performance benchmark is separate from these correctness tests. ONNX Runtime
1.28 initializes successfully in Ubuntu 22.04. All nine updated npm archives
match their lockfile digests, and the production audit reports zero findings.

The JavaScript suite passes 330 tests, with one WASM test skipped because its
module is not built. This is not a full no-skip release qualification. Existing
workspace peer/platform installation conflicts also remain; `npm ci` across
all workspaces is not claimed to pass.

RVF/HTTP benchmark receipts and runtime comparisons are published at
https://gist.github.com/ruvnet/eefbcb089426cc4627bf2d4747ae4187.
This benchmark uses TypeSafe sentence embeddings, not a KGE text-to-graph
adapter. JEV comparisons replay recorded answers. Fast local loopback timings
and historical remote timings have different network overhead. An RVF restores
examples and refits heads; it does not contain a frozen-head native loader.

## Validate locally

```sh
git fetch origin fix/ruvector-status-20261011
git checkout --detach origin/fix/ruvector-status-20261011
cargo test -p ruvllm-cli --test serve_strict
cargo test -p mcp-brain-server --lib graph::
(cd npm && npm audit --omit=dev --audit-level=critical)
node npm/packages/typesafe/scripts/fetch-models.mjs --model bge-small-en-v1.5-int8
cargo test -p ruvector-embed-core --features native --test native_models \
  int8_embeddings_are_independent_of_batch_composition -- --exact --ignored
```

Ubuntu 22.04 users should use the pinned shared ORT setup in `.github/workflows/ci.yml`.
See `npm/packages/typesafe/bench/README.md` and the gist for the full RVF benchmark
build and reproduction steps. Compare hashes from `SOURCE_SHA` and `SHA256SUMS`
before inspecting the attached source archive.

## Before publication

Require successful exact-candidate CI and published-artifact checks; do not
interpret this draft, a queued job, or a skipped test as release approval.
Recheck calibration and accuracy for any model promoted to production. The
singleton inference policy changes previous batched training outputs, so repeat
the benchmarks when upgrading. Keep the last known-good package available until
a separately versioned candidate clears its applicable release gates.
