# RuGraph JavaScript and TypeScript SDK

**Connect knowledge. Reason with evidence. Trace every decision.**

[RuGraph language guide](https://github.com/ruvnet/RuVector/blob/feat/graph-language-v1/crates/ruvector-language/README.md)

JavaScript and TypeScript process bindings for the RuGraph programming language, built on RuVector. This package does not download, install, or execute a model automatically. Install a trusted `rvg` native binary for KGE predictions, or use `rvg-core` for graph and evidence programs.

From the repository root:

```sh
cargo build --manifest-path crates/ruvector-language/Cargo.toml --release
npm install ./npm/packages/graph-language
```

Node.js 20 or newer is required. Set `RUVECTOR_LANGUAGE_BIN` to the absolute binary path, or pass it explicitly:

```js
import { readFile } from 'node:fs/promises';
import { RuGraph, RuGraphError } from '@ruvector/graph-language';

const language = new RuGraph({
  binary: '/absolute/path/to/rvg-core',
  timeoutMs: 30000,
});
const source = await readFile('crates/ruvector-language/examples/incident.rvg', 'utf8');
const program = await language.compile(source);
const snapshot = JSON.parse(await readFile('crates/ruvector-language/examples/snapshot.json', 'utf8'));
const context = JSON.parse(await readFile('crates/ruvector-language/examples/context.json', 'utf8'));
const request = { program, snapshot, context };
const receipt = await language.run(request);
await language.replay(request, receipt);
```

The package includes TypeScript declarations for programs, snapshots, contexts, observations, predictions, decisions, and receipts. `check(source)` validates without executing. `compile(source)` accepts RuGraph text or strict program JSON. `run(request, { modelsPath })` and `replay(request, receipt, { modelsPath })` can load an explicitly supplied native model artifact through `rvg`.

## Process and data limits

The executable and model paths are trusted host configuration. Commands use argument arrays with `shell: false`. Programs cannot choose an executable or add subprocess arguments. Replay writes its receipt into a private temporary directory and removes it after completion.

Input is limited to 16 MiB per call. Combined stdout and stderr are drained with a shared 16 MiB limit; exceeding it terminates the child. The default timeout is 30 seconds and the maximum configurable timeout is 300 seconds. Runtime budgets are separate deterministic work limits. The SDK is not an operating system sandbox for a malicious executable.

All integer values must be within `Number.MIN_SAFE_INTEGER` through `Number.MAX_SAFE_INTEGER`, inclusive. In particular, Rust `u64` timestamps can exceed this JavaScript range. The SDK rejects unsafe integers on input and on parsed output with `UNSAFE_INTEGER` rather than returning rounded values. Use the Python SDK or Rust API when full `u64` precision is required. `BigInt`, circular JSON, nonfinite numbers, malformed runtime output, and invalid UTF8 are rejected.

Failures use `RuGraphError` with `code`, `message`, and optional `position`. Input encoding failures may throw synchronously; process failures reject the returned promise. Wrap the complete call in `try` and `await`:

```js
try {
  const result = await language.run(request);
  console.log(result.receipt_hash);
} catch (error) {
  if (!(error instanceof RuGraphError)) throw error;
  console.error(error.code, error.message, error.position);
}
```

Typical process errors are `TIMEOUT`, `OUTPUT_LIMIT`, `EXECUTABLE_ERROR`, and `INVALID_OUTPUT`. Runtime error codes, such as `UNAUTHORIZED` and `BUDGET_EXCEEDED`, are preserved. Receipt hashes provide integrity only; `replay` checks execution against the supplied inputs and model state.

## Verification

```sh
RUVECTOR_LANGUAGE_BIN="$PWD/crates/ruvector-language/target/release/rvg-core" \
  node --test npm/packages/graph-language/test/*.test.mjs
```

The tests execute the real binary and separately exercise oversized output, process timeouts, malformed responses, and unsafe integer handling.

## Counterfactual audit

`audit(request, options)` recomputes a decision program after each independent
observation, source, or edge removal. Options contain `interventions` (1 to 64
objects with `kind` and `id`) and `max_execution_work` (at most one billion).
The reserved total is the per-run budget times the number of trials plus baseline.
It rejects prediction steps, missing targets and duplicate interventions. Status
changes are in `trials[].changes`; null means a missing decision domain entry,
not `unknown`. `changed_decisions` also counts witness changes. Repeat the audit
and compare the complete report for reproducibility. Options use a private
file that is removed on success or failure. Existing process bounds apply.

This method requires the runtime built from the same revision as this SDK.
Audit sensitivity does not authenticate sources or establish real world causality.
