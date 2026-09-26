# Finite benchmark gate reproduction

This is a host evidence validation fix. No MCU, sensor, model or energy gain is
claimed. The independent test file was frozen before candidate implementation.
The parent produced five false acceptance outcomes among seven named malformed
fixtures. The candidate rejects all seven and preserves 96 finite results.

From the repository root, reproduce the parent comparison and fixed evaluator:

```sh
python3 examples/esp32-decision/tests/evidence/finite-harness/reproduce.py /tmp/finite-reproduced.json
python3 -m unittest discover -s examples/esp32-decision/tests -p test_finite_benchmark.py -v
```

`reproduce.py` combines the original two inline evaluation invocations. It was
subsequently executed by the coordinator with the same seven cases, seed84491
and 96 finite comparisons. Its docstring distinguishes the reconstructed
wrapper from the original invocations. It needs the parent Git object locally.

For integrated acceptance, use GCC, Cargo and CMake on PATH:

```sh
cd examples/esp32-decision
mkdir -p build-finite
python3 tools/lab.py > build-finite/lab.log 2>&1
```

This execution environment requires `ASAN_OPTIONS=detect_leaks=0` under ptrace;
address and undefined behavior checks remain enabled. Use leak checks normally
where supported. The lab result here is57 Rust tests,36 Python tests and11 exact
native pairs. Target binaries were not rebuilt because target sources and
compiler flags are unchanged.

Real pinned harness calls can replay the archived evaluator receipt against
these exact sources. Keep the archived receipts separate from new outputs:

```sh
cp tests/evidence/finite-experiment.json build-finite/experiment.json
cp tests/evidence/finite-evaluator.json build-finite/evaluator.json
git clone https://github.com/ruvnet/metaharness.git build-finite/metaharness
git -C build-finite/metaharness checkout d5833dc6512ac1adeeef91a331c29055cd8a4dbb
git clone https://github.com/ruvnet/autogenous.git build-finite/autogenous
git -C build-finite/autogenous checkout 905aa6cbe213392f8b3cab5d4f17bc3a48e0a509
node tests/evidence/finite-harness/retention-gate.mjs . build-finite/metaharness build-finite/metaharness-replay.json
python3 tests/evidence/finite-harness/mutation_record.py . build-finite/metaharness-replay.json build-finite/mutation-replay.json
cargo run --locked --manifest-path tests/evidence/finite-harness/authority-adapter/Cargo.toml -- build-finite/mutation-replay.json 1790391984
```

The timestamp is the original recorded execution instant, only for historical
replay. It does not authorize a mutation after expiry03:47:32UTC on2026-09-26.
Autogenous ApplicationCode requires governed authority and automatic promotion
remains false. The real typed admission check rejects authority expansion,
insufficient authority, missing rollback, invariant regression and expiry.
No FitnessVector scores or signed production genome are invented.

MetaHarness uses actual `decidePromotion` source, pinned by SHA256. Its score
is malformed fixture rejection, not speed or unseen quality. It also verifies
regression and replay vetoes. Bootstrap language over these public fixtures
cannot establish generalization or a timing confidence interval.

The recorded invocations preserve original paths. The packaged Cargo manifest
uses a relative dependency and its own workspace; the coordinator compiled
and executed that packaged version successfully. Checksums retain both forms.

Rollback source:6fdb5f12ac7fe16752eeb60405e52567081706db. Restore only rig.py,
preserve rejection evidence, and require new validation before any promotion.
Physical execution labels and correctness attestations still require trusted
external provenance; this change does not authenticate those attestations.
