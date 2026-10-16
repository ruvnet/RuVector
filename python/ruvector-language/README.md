# RuGraph Python SDK

**Connect knowledge. Reason with evidence. Trace every decision.**

[RuGraph language guide](https://github.com/ruvnet/RuVector/blob/feat/graph-language-v1/crates/ruvector-language/README.md)

Python process bindings for RuGraph programs, evidence decisions, prediction artifacts, and replay receipts. Python 3.10 or newer is required. The package has no runtime dependencies and does not install a runtime binary or model automatically.

From the repository root:

```sh
cargo build --manifest-path crates/ruvector-language/Cargo.toml --release
python -m pip install ./python/ruvector-language
```

Use `rvg-core` for graph and evidence operations, or the native `rvg` binary for RuVector KGE predictions. The binary path can be passed directly or configured with `RUVECTOR_LANGUAGE_BIN`.

```python
import json
from pathlib import Path
from ruvector_language import RuGraph, RuGraphError

examples = Path('crates/ruvector-language/examples')
language = RuGraph(binary='/absolute/path/to/rvg-core', timeout=30)
request = {
    'program': language.compile((examples / 'incident.rvg').read_text()),
    'snapshot': json.loads((examples / 'snapshot.json').read_text()),
    'context': json.loads((examples / 'context.json').read_text()),
}
receipt = language.run(request)
assert language.replay(request, receipt) == receipt
```

`check(source)` validates without execution. `compile(source)` accepts RuGraph text or strict program JSON. `run(request, models_path=...)` and `replay(request, receipt, models_path=...)` pass an explicitly supplied artifact to the native runtime. The core runtime does not load KGE models.

## Process and data limits

The executable and model paths are trusted host configuration. The SDK uses argument arrays with `shell=False`; program text is never a shell command. Input is limited to 16 MiB and passed through a temporary file. Replay stores its receipt with owner-only permissions in a private temporary directory.

Stdout and stderr are drained concurrently with a shared 16 MiB buffer limit. A process that exceeds this limit is terminated while running. The default timeout is 30 seconds, configurable up to 300 seconds. On POSIX systems termination also applies to the child process group. On other platforms it terminates the direct child. These bounds do not make a malicious executable safe; use a trusted runtime binary.

Python integers preserve the full Rust `u64` timestamp range. Real runtime tests exercise exact replay near `2**64 - 1`. Nonfinite input or output, circular input, invalid UTF8, duplicate output JSON keys, and malformed responses are rejected.

Failures raise `RuGraphError`, preserving runtime `code`, `message` through `str(error)`, and optional `position`:

```python
try:
    receipt = language.run(request)
except RuGraphError as error:
    print(error.code, str(error), error.position)
```

Typical process errors include `TIMEOUT`, `OUTPUT_LIMIT`, `EXECUTABLE_ERROR`, and `INVALID_OUTPUT`. Runtime work budgets remain separate from process timeouts. Receipt checksums provide integrity, not authenticity or proof that observations are true; replay reexecutes with the supplied snapshot, clock, permissions, and models.

## Verification

```sh
RUVECTOR_LANGUAGE_BIN="$PWD/crates/ruvector-language/target/release/rvg-core" \
  PYTHONPATH=python/ruvector-language \
  python -m unittest discover -s python/ruvector-language/tests -v
```

The tests execute the real runtime and include expiry, retraction, capability denial, exact `u64` values, process timeouts, oversized output, and malformed child responses.

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
