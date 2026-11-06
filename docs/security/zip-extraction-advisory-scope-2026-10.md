# ZIP extraction advisory scope

**Status:** temporary exception for review; not a dependency remediation  
**Reviewed:** 2026-10-18  
**Re-review by:** 2026-11-01

## Decision

The current RuVector Cargo.lock contains three ZIP releases with nine RustSec advisory instances:

| ZIP | Locked parent | Advisory IDs |
|---|---|---|
| 1.1.4 | candle-core 0.8.4 | RUSTSEC-2026-0367, RUSTSEC-2026-0368, RUSTSEC-2026-0369 |
| 2.4.2 | ndarray-npy 0.9.1 | RUSTSEC-2026-0367, RUSTSEC-2026-0368, RUSTSEC-2026-0369 |
| 7.2.0 | candle-core 0.9.2 | RUSTSEC-2026-0367, RUSTSEC-2026-0368, RUSTSEC-2026-0369 |

The affected parent crates use ZIP archive reading/writing for NPY, pickle, and NPZ data. Source review found no call to ZIP extraction APIs (`.extract(...)` or `ZipStreamReader`) in these locked parent versions. The vulnerable behavior is not reachable through these current dependency call paths.

This exception leaves the vulnerable ZIP releases in the lockfile. It is **not** a fix to ZIP and is not, by itself, release approval. CI keeps both RustSec and cargo-deny checks enabled, and runs `scripts/check-zip-extraction-scope.mjs` before cargo-audit. The guard verifies the exact ZIP versions and reverse dependencies in Cargo.lock, downloads each crate archive from crates.io, validates SHA-256 against the lockfile, and scans the parent source. Any graph/checksum drift, failed download, or extraction API match fails the job.

## Evidence

The checksums verified against Cargo.lock are:

| Crate archive | SHA-256 |
|---|---|
| zip 1.1.4 | `9cc23c04387f4da0374be4533ad1208cbb091d5c11d070dfef13676ad6497164` |
| zip 2.4.2 | `fabe6324e908f85a1c52063ce7aa26b68dcb7eb6dbc83a2d148403c9bc3eba50` |
| zip 7.2.0 | `c42e33efc22a0650c311c2ef19115ce232583abbe80850bc8b66509ebef02de0` |

Upstream parent-crate work is tracked in [candle-rs/candle#4041](https://github.com/huggingface/candle/pull/4041). ndarray-npy 0.10.0 still constrains ZIP below 9, so its path also remains blocked pending an upstream compatible release. RustSec advisories: [0367](https://rustsec.org/advisories/RUSTSEC-2026-0367), [0368](https://rustsec.org/advisories/RUSTSEC-2026-0368), [0369](https://rustsec.org/advisories/RUSTSEC-2026-0369).

## Exit criteria

Remove all three advisory ignores and this temporary guard when supported upstream parent releases resolve to ZIP >=9. Re-run cargo-audit, cargo-deny, relevant NPY/NPZ/pickle tests, and the full release security workflow. Remove the exception immediately if the dependency graph changes or any affected extraction API becomes reachable.
