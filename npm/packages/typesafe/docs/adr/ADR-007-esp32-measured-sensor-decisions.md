# ADR 007: Measured sensor decisions on ESP32

Status: Implemented in the ESP32 decision example; physical acceptance pending.
Date: 2026-09-26
Scope: ESP32 S3 and C6, ESP-IDF 5.4.4, fixed numeric decision heads.

## Problem

Synthetic parity proves that firmware reproduces a reference head. It does not
prove sensor accuracy, silicon latency or energy consumption. Host and virtual
instruction clocks cannot establish a physical speedup. The previous S3
rounding experiment also showed that a faster host kernel can cost more target
instructions. Claims must identify the executed hardware, workload and image.

## Decisions and invariants

1. Preserve the default synthetic model as a regression fixture. Add a separate
   real occupancy model, trained through the actual RuVector Rust engine.
   Firmware accepts both embedded features with `infer` and raw measurements
   with `sensor` when the compiled model declares a preprocessing pipeline.
2. Use UCI Occupancy Detection, DOI 10.24432/C5X01N, by Luis Candanedo (2016),
   CC BY 4.0. Pin the archive digest. Keep the supplied training and test files
   separate. Reserve the last training day for validation. Fit means, standard
   deviations, model parameters and prototypes only on the training portion.
   Dates and row IDs are never input features. Test labels never select a model,
   threshold, precision or preprocessing parameter. This is one office and
   does not establish generalization to different buildings or sensor devices.
3. Input units and order are Celsius, relative humidity percent, lux, CO2 ppm
   and humidity ratio kg/kg. Apply binary32 standardization, then the kernel's
   existing normalization. Include this schema and preprocessing in the model
   digest. Invalid or nonfinite sensor input fails closed.
4. Select precision on validation with fixed gates: decision and acceptance
   agreement >=99%, maximum absolute probability error <=0.025. Keep INT16
   when INT8 fails. Report test accuracy, confusion matrix, coverage, accepted
   accuracy, Brier score and confidence calibration diagnostics separately
   from reference parity. No quantized calibration claim is inherited.
5. `profile N` reports bounded samples, cycles, timer overhead, percentiles,
   frequency, core and warmed heap. `energy N` brackets a bounded inference
   batch with an optional GPIO marker. GPIO is disabled by default. Capture
   and sensor driver timing remain external until a real driver is attached.
6. Paired measurements use the same instrumentation, model and target build
   settings around both kernels. Pin the baseline Git revision. Alternate
   image order across rounds. Verify model and kernel digests before recording
   results. Never infer physical execution merely from an ESP32 target name.
   Disable floating point contraction in both kernels with `-ffp-contract=off`.
   S3 replay exposed probability differences of roughly 6e-8 with the default
   contraction policy even when x86 output was bit-identical. The strict
   comparison rejected that build; controlled contraction restores equality.
7. Record preprocessing, inference and host round trip separately. Protocol
   timing includes transport and formatting; kernel timing excludes them.
   Cycle counters require fixed CPU frequency and task affinity. Preserve raw
   overhead rather than subtracting it from each small measurement.
8. Integrate timestamped voltage/current traces over GPIO windows. Record gross
   energy, optional idle-adjusted energy, decision count, sample resolution,
   trace hash, image hash and measurement scope. Reject malformed traces,
   incomplete windows and mismatched run counts. Simulated traces verify the
   analysis only; they never become physical power evidence.
9. Retain an optimization only with >=10% paired latency or energy improvement,
   a confidence bound that excludes regression, unchanged exact kernel output
   for arithmetic-only changes, passing quantization gates, and stable warmed
   heap. Correctness failure vetoes a timing win. Physical release requires
   both chips and the intended sensor pipeline; CI cannot waive that gate.

## Delivery inputs, outputs and assumptions

| Work | Inputs | Outputs | Assumption |
| --- | --- | --- | --- |
| Sensor training | Pinned public archive | Model headers, split provenance, accuracy report | Temporal transfer within one office |
| Firmware profiling | Frozen model and bounded requests | Stage times, percentiles, cycles and metadata | Fixed frequency and CPU affinity |
| Board comparison | Two images, manifest, serial port | Paired samples and retain/reject decision | Actual hardware declared by operator |
| Energy analysis | Marker trace and batch result | Joules per decision and trace provenance | Instrument units and scope are correct |

## Alternatives

Using an x86 speedup as an MCU claim is rejected. Promoting INT8 solely because
it is smaller is rejected. Training on randomly shuffled time windows is
rejected because adjacent samples leak conditions across the evaluation split.
S3 DSP assembly remains an optional candidate: integration must preserve wide
accumulation, rounding and model semantics and pass these same gates first.

## Reproduction and rollback

Commands and schema are in the ESP32 section of the typesafe README. Evidence
is under `examples/esp32-decision/tests/evidence/`. Build the original kernel
with the same new profiling harness for a valid paired baseline. The previous
known image and model stay addressable by SHA256; reflash them to roll back.
No firmware, sensor or model update is downloaded automatically by the device.
The board driver overrides `rd_sensor_read` and `rd_sensor_name`; `sample`
measures acquisition separately. Unconfigured and incomplete captures reject
the request. Test capture adapters are explicitly identified as fixtures.

## Follow-up: production memory and coverage experiments

Percentile buffers are configurable from zero to 2,048 samples, at eight bytes
per sample. `sdkconfig.production` sets zero, removing 16,384 static bytes.
Inference, self-tests, `bench` and the GPIO `energy` command remain available;
`profile` explicitly returns `profile_disabled`. Metadata reports actual
capacity and storage. Paired performance builds must have matching capacities.
This is a verified memory reduction, not a physical speed or energy claim.
The production checker warms every replay input and both measured command
paths before checking a second identical pass for stable heap. Newlib's float
formatter can allocate caches when it first encounters a new numeric magnitude;
the initial heap growth is retained in evidence rather than called a leak or
hidden in the steady state check.

The bounded coverage lab reserves the last training day (1,440 readings) for
abstention selection and refits all preprocessing and prototypes on the
remaining 6,129 rows. The original 574-row validation day is a fixed veto.
Three predeclared abstention settings and both precisions use calibration only.
Select the least relaxed setting that meets empirical quality and coverage
requirements. Freeze this choice before final evaluation; never fall back to
another arm after failure. Test data are already seen regression data, not a
new generalization test. Minute readings are correlated, so support counts and
empirical accuracy are not statistical coverage guarantees.

Require >=95% overall accuracy, >=98% accepted accuracy, >=50% coverage,
and per truth class >=95% accepted accuracy, >=20% coverage and >=16 accepted
examples. Both classes must pass; a detector cannot succeed by accepting only
empty rooms. Quantization retains the existing parity/error gates.

The first frozen candidate passed calibration (99.65% accuracy, 71.94%
coverage) but failed the next validation day (90.59% accuracy, zero coverage).
It was rejected. The default model remains unchanged. A separate room/device
dataset is required before any deployment claim. The lab removes stale
candidate headers on rejection and records the rejection for future work.

## Follow up: compile workspace capacity from the immutable model

Firmware builds now default to `RD_MODEL_SIZED_WORKSPACE=ON`. Read literal
`RD_MODEL_DIMS` and `RD_MODEL_CLASSES` exporter metadata and propagate bounded
`RD_MAX_DIMS` / `RD_MAX_CLASSES` definitions through the component's PUBLIC
interface. All consumers must share these values: workspace, result and
context layouts are part of the ABI. Models still support at most 768 features
and 16 classes. Generic C callers retain those original defaults. Legacy
headers without class metadata retain 16 class slots. The model header is a
CMake configure dependency, so replacing it refreshes the capacities.

This changes storage layout only. Model digest, weights, preprocessing,
thresholds and kernel arithmetic are unchanged. A model exceeding compiled
capacity fails initialization before accessing its rows. Invalid or duplicate
capacity metadata fails configuration. Rollback is an explicit
`-DRD_MODEL_SIZED_WORKSPACE=OFF` rebuild with the same frozen model.

For the five feature, two class INT8 occupancy model, workspace drops from
1,732 to 40 bytes. Context drops from 32 to 20 bytes on both MCUs. Linked
static symbols therefore recover 1,704 bytes per target in addition to the
earlier optional profiling savings. With profiling disabled in both variants,
application flash drops 1,008 bytes on S3 and 608 bytes on C6. Golden result
storage also shrinks; this is not a change to model parameter bytes.

Acceptance uses independent generic and compact executables because their
public struct layouts differ. Require exact semantic replies for every head,
INT8/INT16 fixtures, all three kernel profiles, and 574 occupancy regression
rows; verify both target builds, every component's compile definitions and
linked symbols. `tools/workspace_check.py` records images, sections, compiler
propagation and optional S3 emulator replay with warmed heap checks. Static
memory recovery is directly measurable without physical latency evidence.
No new accuracy, calibration, speed or energy claim follows from this change.

The immediate campaign also invoked the real MetaHarness `decidePromotion`
entrypoint at commit `d5833dc6512ac1adeeef91a331c29055cd8a4dbb` through
`tools/workspace_gate.mjs`. The explicit score is the fraction of linked
workspace and context bytes removed, not accuracy or speed. The adapter checks
frozen inputs, compiled artifact hashes, replay and lab receipts before calling
the source gate, and rejects missing evidence. It sets `hiddenTestPassed=false`:
existing regression data is not unseen evaluation. The upstream bootstrap
wording does not turn deterministic linked sizes into a timing confidence
interval. Source hashes pin the invoked functions.

Autogenous at `905aa6cbe213392f8b3cab5d4f17bc3a48e0a509` passed 19 upstream
host tests. Its actual `MutationScope::ApplicationCode.auto_promotable()`
entrypoint returned false, preserving PR review as the delivery boundary.
Firmware safety probabilities and real latency are unavailable, so its fitness
hard gates were not populated with invented values. Neither harness result
permits an automatic merge, flash or deployment. The standalone `ruvnet/rsi`
repository remained unresolved; no standalone RSI invocation is claimed.

## Sources

* https://archive.ics.uci.edu/dataset/357/occupancy+detection
* https://doi.org/10.24432/C5X01N
* https://docs.espressif.com/projects/esp-idf/en/v5.4/esp32s3/api-guides/performance/speed.html
* https://docs.espressif.com/projects/esp-idf/en/v5.4/esp32s3/api-guides/current-consumption-measurement-modules.html

## Acceptance

Run the software lab, cross compile both targets, replay the final S3 image in
QEMU, and run the paired rig on each physical chip. Require unchanged replay
decisions, the fixed probability gate, stable warmed heap, and a measured
latency or energy gain meeting the retention rule. Missing hardware leaves
physical acceptance explicitly unverified.
