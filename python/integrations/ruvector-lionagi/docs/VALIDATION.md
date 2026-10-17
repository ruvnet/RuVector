# Validation evidence

Measured locally on 2026-10-11 on Linux, Python 3.12.14, LionAGI 0.35.3,
RuVector 0.1.2, NumPy 2.5.4 and pytest 9.1.1. The Rust extension came from the
published RuVector distribution. No replacement vector engine was used.

## Acceptance suite

```sh
python -m pytest tests --cov=ruvector_lionagi --cov-report=term-missing -q
```

Result: **41 passed**, 2.86 seconds for this run. Statement coverage: **92%**
combined; store/tool module 93%, memory adapter 87%. Coverage measures executed
statements, not completeness of security assurance.

Tests exercise:

* Real LionAGI `Branch.act` invoking the registered search `Tool` without model
  provider credentials; results retain the untrusted-data marker.
* Native cosine search, IDs, insertion, deletion, async methods and concurrent
  threaded mutation.
* LionAGI `MemoryStore` protocol, detached memory roundtrips, semantic search,
  tag union matching and conjunctive metadata filters.
* Invalid vector dimensions, zero norm, nonfinite values and booleans; invalid
  result limits; invalid batches produce no partial insertion.
* Capacity, UTF-8 byte, nesting, aggregate payload, vector allocation and tool
  response budgets, including reclamation of payload quota after deletion.
* Metadata cycles, unsupported JSON values, oversized integers, nonstring keys,
  mutation isolation and input array aliasing.
* Cancellation during embedding causes no insertion. Snapshotting happens
  before user supplied embedding code yields.
* Explicitly injected native mutation failure makes the store unavailable;
  explicitly injected deletion rebuild failure preserves existing data.
  These two fault tests replace an index solely to exercise failure handling.

## Security checks

The captured `pip-audit` JSON reported 86 installed distributions with no known
vulnerabilities returned by its advisory database. This includes development
utilities as well as runtime dependencies. The newly developed adapter itself
is validated by its tests and review, not established as safe by this lookup.

The captured Ruflo deep/all scan of this integration reported zero findings
across critical, high, medium and low severities. This is a scanner result, not
a proof that the component has no vulnerabilities.

## Scope and remaining gates

This evidence covers local Linux only. Windows, macOS, additional Python
versions and clean published-wheel installations require their own successful
CI receipts. This document does not claim those jobs completed or that this
package is published. There was no live provider invocation, external model
quality evaluation, adversarial LLM prompt-injection trial or production load
soak. Consult the benchmark report separately for measured performance.

## Distribution acceptance

An isolated PEP 517 build produced the wheel from the source distribution. Both
passed `twine check --strict`. The resulting wheel replaced the editable install;
all 41 tests and the offline quickstart then passed against the installed wheel.
This is local wheel validation, not proof of a completed PyPI release.

The dependency audit snapshot is recorded in [dependency-audit.json](dependency-audit.json).
