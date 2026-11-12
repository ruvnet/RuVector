# Workspace capacity experiment

Parent: `5ab62b70fddce9b3436b69f69bdc4e22e43b0227` in PR 1018.
Only storage capacities change. The kernel, model and evaluator hashes in
`workspace-experiment.json` remain fixed. Existing evaluation rows are
regression data, not a new generalization test.

Use GCC, Cargo, CMake, Node with `node:module.stripTypeScriptTypes`, and the
ESP-IDF 5.4.4 environment. Run from `examples/esp32-decision`:

```sh
mkdir -p build-workspace/frozen-model
cp tests/evidence/workspace-experiment.json build-workspace/experiment.json
python3 tools/lab.py > build-workspace/lab.log 2>&1
python3 -m unittest discover -s tests -p test_workspace.py -v > build-workspace/host-tests.log 2>&1
cp build-sensor/selected/model.h build-workspace/frozen-model/model.h

for chip in esp32s3 esp32c6; do
  for variant in generic compact; do
    sized=OFF
    if test "$variant" = compact; then sized=ON; fi
    idf.py -B "build-workspace/$chip-$variant" -DIDF_TARGET="$chip" \
      -DSDKCONFIG="$PWD/build-workspace/sdkconfig.$chip.$variant" \
      '-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.production' \
      -DRD_MODEL_DIR="$PWD/build-workspace/frozen-model" \
      -DRD_MODEL_SIZED_WORKSPACE="$sized" build merge-bin \
      > "build-workspace/build-$chip-$variant.log" 2>&1 || exit 1
  done
done
python3 tools/workspace_check.py --generic build-workspace/esp32c6-generic \
  --compact build-workspace/esp32c6-compact --output build-workspace/memory-esp32c6.json
python3 tools/workspace_check.py --generic build-workspace/esp32s3-generic \
  --compact build-workspace/esp32s3-compact --output build-workspace/memory-esp32s3.json \
  --qemu /path/to/qemu-system-xtensa --vectors build-sensor/validation.json
```

This runtime required `ASAN_OPTIONS=detect_leaks=0` because LeakSanitizer does
not work under ptrace. Address and undefined behavior checks remained enabled.
Use the normal leak check where the environment supports it. QEMU here was
Espressif 9.2.2; its system libraries must be available to the executable.

Invoke the actual pinned MetaHarness source after producing those receipts:

```sh
git clone https://github.com/ruvnet/metaharness.git build-workspace/metaharness
git -C build-workspace/metaharness checkout d5833dc6512ac1adeeef91a331c29055cd8a4dbb
node tools/workspace_gate.mjs . build-workspace/metaharness build-workspace/metaharness-gate.json
```

The adapter pins the promotion and statistics source hashes, checks compiled
images and frozen inputs, and requires host and S3 emulator replay. Its score
is the fraction of workspace and context symbol bytes removed. The returned
upstream statistical wording is not a timing confidence interval. Five
negative probes rejected missing replay, failed replay, missing emulator
evidence, a changed image and a changed evaluator. Missing evidence cannot
be interpreted as successful acceptance.

Autogenous authority invocation is reproducible with its actual crate:

```sh
git clone https://github.com/ruvnet/autogenous.git build-workspace/autogenous
git -C build-workspace/autogenous checkout 905aa6cbe213392f8b3cab5d4f17bc3a48e0a509
mkdir -p build-workspace/authority-adapter/src
cp tools/workspace_authority.rs build-workspace/authority-adapter/src/main.rs
cat > build-workspace/authority-adapter/Cargo.toml <<'TOML'
[package]
name = "esp32-authority-gate"
version = "0.1.0"
edition = "2021"
[workspace]
[dependencies]
agl-types = { path = "../autogenous/crates/agl-types" }
TOML
cargo generate-lockfile --manifest-path build-workspace/authority-adapter/Cargo.toml
cargo run --locked --manifest-path build-workspace/authority-adapter/Cargo.toml
cargo test --locked --manifest-path build-workspace/autogenous/Cargo.toml -p promotion -p agl-types
```

The authority result must deny automatic application code deployment. The 19
upstream tests cover host gate and rollback behavior, not physical firmware
rollback. No firmware fitness probabilities or physical latency are invented.

Acceptance: 1,704 linked static bytes recovered on each target, exact replay,
stable warmed S3 heap, successful software suite and both builds. Delivery is
PR review only. Roll back with `RD_MODEL_SIZED_WORKSPACE=OFF` and the same
model. Physical C6/S3 performance and deployment qualification remain pending.
