# Shared e-means release

The shared `ruvector-e-means` crate and `@ruvector/e-means` package are released
before their dependent consumer integrations. This leaf change contains no new
consumer dependency on an unpublished registry package. RuVector PR #1489 and
ruv-FANN PR #227 integrate the public dependencies afterward.

The release branch runs `e-means-preflight.yml` at an immutable source commit.
Its 15 mandatory commands execute the shared native tests and strict Clippy,
verify a Cargo archive, audit that archive's dependency lock, check Rust 1.81,
execute the packaged Rust tests, and build WASM from the extracted archive.
The WASM build explicitly retains the workspace release optimization settings.
The copied npm package then runs its dependency audit, real WASM/Worker tests,
strict declarations, installed consumers with and without the optional SDK,
and real Chromium numerical checks before packing. A second job installs the
sealed archive offline and exercises the portable APIs on Node 20.0.0.

The source-bound candidate records actual command exit codes, owned logs and
digests. The verifier requires the complete ordered command set and derives
test counts again from the logs. Zero tests, ignored/filtered Rust tests,
skipped/todo Node tests, missing browser or installed-consumer checks, altered
payloads and missing harness files reject the candidate.

After that unchanged commit reaches main, `e-means-protected-release.yml`
selects the successful release-branch run itself. It requires current main,
zero labeled release blockers, both successful jobs, a unique source-bound
artifact and its GitHub ZIP digest. Both public registries are checked for
conflicting immutable bytes before either upload. The controller uploads only
the sealed crate and npm archives, then compares public downloads. Publication
credentials are limited to that upload step.

The final consumer step uses a virgin Cargo home with an exact registry pin,
checks the official Cargo registry source and checksum, and executes numerical
fit/predict/PQ/IVF assertions. It installs the public npm package separately,
compares installed files with the public archive, then runs the transferred
test harness, strict declarations and actual Chromium. Partial publication is
not a completed release; retained receipts and command logs identify the state.

After public consumers pass, the protected job creates immutable tag
`e-means-v0.1.0` and a draft GitHub release. It attaches the canonical archives,
original candidate ZIP, checksums, source-bound receipts and actual command logs.
Every asset is downloaded and compared before the draft becomes public and
again afterward. A conflicting tag or asset rejects promotion; an identical
partial draft may resume without overwriting bytes.

This is scoped package qualification. The existing broad-monorepo advisory
failure is tracked separately in RuVector #1487. A successful leaf audit does
not claim that unrelated workspace dependencies or hosted deployments passed.
Historical benchmark results retain their original provenance; this release
does not relabel them as measurements of the sealed public artifacts.
