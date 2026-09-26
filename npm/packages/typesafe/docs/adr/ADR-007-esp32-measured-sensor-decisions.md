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

## Follow up: finite benchmark evidence before retention

The parent timing/energy retention function accepted infinite timing inputs,
booleans and overflowing derived statistics. An eleven pair fixture with an
infinite baseline produced an infinite speedup and `retain=true`. JSON numeric
syntax such as `1e309` can decode to infinity without a JSON parsing error.
This is an evidence validation defect, not an observed device performance gain.

Require built in integer or float measurements that are positive and finite;
booleans are not measurements. Reject integers outside the finite float range,
nonfinite or underflowed ratios, and nonfinite median/bootstrap results before
returning a retention decision. Preserve integer operands until division to
avoid introducing an extra rounding step. The same function protects energy
comparison. Pair count 11, required speedup 1.10, lower bound above 1.0, seed719,
physical provenance and correctness vetoes remain unchanged.

Treat the retention function as the product under test. Its independent
acceptance evaluator was frozen before implementation, and all earlier model,
kernel and acceptance fixture hashes were preserved. Seven named malformed
cases now reject; five formerly produced false acceptance. Nine new test
methods cover both arms, energy CLI overflow and fixed threshold behavior.
An independent 96 case comparison preserved every valid finite parent result.
The integrated lab passed 57 Rust tests, 36 Python tests, address/undefined
checks and eleven exact native pairs. No target code or compiler flag changed;
MCU rebuilds and physical measurements were not repeated for this host fix.

Pinned MetaHarness `decidePromotion` evaluated observed malformed fixture
rejection and verified regression vetoes. Autogenous `Mutation::admissible`
checked the reversible, governed PR proposal, including authority, rollback,
invariant and expiry negatives. Automatic application code promotion remains
false. Public deterministic fixtures establish neither a new model quality
claim nor a statistical device speedup. Sources and receipts are stored under
`tests/evidence/finite-*`. Restore `tools/rig.py` from parent
`6fdb5f12ac7fe16752eeb60405e52567081706db` to undo this change, while retaining
the regression evidence. Reproduction: run `python3 tools/lab.py` from the
firmware directory and the recorded harness commands in the evidence folder.

## Follow up: bind energy evidence to exact rig rounds

The energy comparator previously checked the rig report digest and model but
did not prove that both arms came from the same named rig round. Reports from
different rounds could be paired, a round could be reused with distinct trace
hashes, and the report's target, kernel, image, decision count or scope could
disagree with that round. All seven synthetic tamper cases reached retention.
This was an offline evidence integrity defect, not a measured energy gain.

Require exactly one baseline and candidate report for every rig round, in the
recorded order. Each report must match the rig execution class, physical flag,
model, arm target, kernel digest, complete image descriptor, energy batch run
count and `kernel_batch` scope. Existing trace uniqueness, fixed workload and
statistical retention gates remain additional checks. The rig report digest
still binds every report to one correctness run. These checks detect stale or
cross-paired evidence; they do not authenticate an actor able to rewrite the
rig report, energy reports and traces together.

An independent evaluator was frozen against parent
`21fd8753ea1d54473ef797e9c6ebaaa62c7749bf`. The parent rejected zero of seven
named tamper cases; the candidate rejects all seven and preserves the valid
eleven-pair result. Fresh integrated acceptance passed 57 Rust tests, 42 Python
tests, address/undefined checks and eleven exact native pairs. Firmware, model,
quantization, compiler flags and target code are unchanged, so this experiment
does not claim a new MCU build, silicon timing or joules per decision.

Pinned MetaHarness `decidePromotion` at
`d5833dc6512ac1adeeef91a331c29055cd8a4dbb` evaluated the observed fraction of
tamper cases rejected and vetoed injected replay, regression and safety
failures. Its deterministic case scores are not a physical timing confidence
interval. Pinned Autogenous at
`905aa6cbe213392f8b3cab5d4f17bc3a48e0a509` admitted the typed, reversible
application-code mutation, rejected six authority, lineage, invariant, expiry
and rollback violations, and confirmed that application code is not
auto-promotable. Candidate production fitness was not invented and deployment
remains false. The standalone `ruvnet/rsi` repository remains the previously
recorded 404 blocker; no fresh standalone invocation is claimed.

Receipts and exact source pins are under `tests/evidence/provenance-*`.
Reproduce with `python3 -m unittest tests.test_energy_provenance -v`, then
`python3 tools/lab.py`, the recorded MetaHarness adapter invocation and the
Autogenous adapter command. Roll back the comparator to parent `21fd8753` while
retaining the evaluator if any provenance or regression gate fails.

## Follow up: enforce one transport query per firmware command

The Python transport previously appended a newline without rejecting a query
that already contained CR or LF. A caller could therefore write two firmware
commands with one `query` call. The first reply was returned immediately and
the second remained queued, so the following query silently received stale
output. This could associate a benchmark step with the wrong firmware reply.
It is a host protocol integrity defect, not a firmware execution or performance
defect.

Require every transport query to be one line. Reject CR or LF with
`ValueError` before selecting or writing to either the subprocess or serial
stream. Preserve spaces and tabs because valid firmware numeric commands may
use them. Empty and otherwise invalid single-line commands remain firmware
protocol concerns and still receive exactly one response.

The independent evaluator was frozen against parent
`b65e0c7cad47e5ad97ad64a55091797694967cff`. The three inputs containing LF
wrote split commands on the parent and left a stale reply. Bare CR was ignored
by the firmware parser, silently joining `alpha` and `beta` into
`alphabeta`. The candidate rejects all four forbidden framing inputs before
a write, leaves zero stale followups or command normalization and preserves the
named single-line commands. Fresh integrated acceptance passed
57 Rust tests, 47 Python tests and eleven exact native pairs. Firmware, model,
quantization, compiler flags and target sources are unchanged, so target builds
and physical measurements were not repeated and no MCU speed, energy or memory
claim is made.

Pinned MetaHarness `decidePromotion` at
`d5833dc6512ac1adeeef91a331c29055cd8a4dbb` evaluated the observed fraction
of four visible framing cases rejected. The adapter hard-vetoes any write,
stale reply, valid-command regression, missing hash or lab failure and records
`hiddenTestPassed=false`. Pinned Autogenous at
`905aa6cbe213392f8b3cab5d4f17bc3a48e0a509` admitted the governed,
reversible mutation, exercised six lineage, authority, rollback, invariant and
expiry negatives, and confirmed that application code is not auto-promotable.
No deployment fitness was invented. The previously recorded standalone RSI
blocker is unchanged and was not queried again.

Receipts are under `tests/evidence/transport-*`. Reproduce from the firmware
directory with `python3 -m unittest tests.test_transport_boundary -v` and
`python3 tools/lab.py`, then run the recorded MetaHarness and Autogenous
commands. Roll back `tools/transport.py` to parent `b65e0c7` while retaining
the evaluator if any framing or regression gate fails.

## Follow up: serialize host transport transactions

The host `Device` previously allowed two threads to write commands and wait for
responses concurrently. The firmware protocol emits ordered JSON lines without
request identifiers. If one caller paused after its write, another caller could
consume the first reply and the first caller could then consume the second. A
decision or benchmark result could therefore be attached to the wrong request
even though both JSON responses were individually valid.

Treat write, flush and response as one transaction protected by a per-device
lock. Validate the single-line framing contract before acquiring the lock.
Apply the same transaction boundary to subprocess and UART transports. This
does not add parallel firmware execution: the line protocol and firmware
command loop were already sequential.

The independent evaluator was frozen against parent
`47b042815052be2263498d0ca9ae2ae1f31cd7e1`. A controlled serial adapter
paused the first caller after writing and reproduced exchanged replies in all
eight parent pairs. The candidate correctly associates all eight pairs and
preserves four sequential commands plus all four CR/LF framing rejections.
Fresh integrated acceptance passed 57 Rust tests, 53 Python tests and eleven
exact native pairs. Firmware, model, quantization, compiler flags and target
sources are unchanged, so target builds and physical measurements were not
repeated. No latency, energy, memory or deployment claim follows.

Pinned MetaHarness `decidePromotion` at
`d5833dc6512ac1adeeef91a331c29055cd8a4dbb` evaluated the visible fraction
of eight correctly associated pairs and hard-vetoed any misassociation,
sequential or framing regression, missing hash or lab failure. It records
`hiddenTestPassed=false`; deterministic scheduling is not a physical timing
confidence interval. Pinned Autogenous at
`905aa6cbe213392f8b3cab5d4f17bc3a48e0a509` admitted the governed,
reversible mutation, rejected six lineage, authority, rollback, invariant and
expiry violations, and confirmed that application code is not auto-promotable.
No deployment fitness was invented. The previously recorded standalone RSI
blocker is unchanged and was not queried again.

Receipts are under `tests/evidence/concurrency-*`. Reproduce from the firmware
directory with `python3 -m unittest tests.test_transport_concurrency -v` and
`python3 tools/lab.py`, then run the recorded MetaHarness and Autogenous
commands. Roll back `tools/transport.py` to parent `47b0428` while retaining
the evaluator if any association or regression gate fails. A timed-out request,
device restart or concurrent `close` remains a separate recovery problem.

## Follow up: C6 physical rig, all ESP32 targets and serial OTA

*Physical paired rig.* The occupancy pair was built in `espressif/idf:v5.4.4`
from parent `19376a6` plus this change. Both arms share the app sources and
differ only in the kernel. The rig ran on one ESP32-C6 (CP210x UART0, 160 MHz,
fixed affinity) for 11 alternating rounds: 128 held-out test rows and 2,048
profiled runs per arm. Every round replayed identical decisions, both arms
matched the reference on all 128 rows, and warmed heap was stable. The mean
kernel time fell from 76.69 µs to 72.63 µs (12,084 to 11,471 mean cycles). The
median paired speedup was 1.056, with a bootstrap lower bound of 1.055. That
is below the 1.10 retention rule, so `retain=false`: the C6 improvement is
real but does not meet the gate. Replay stages show that UART text parsing
(about 119 µs) costs more than inference (about 77 µs). Parsing, not the
kernel, is the next C6 latency target. S3 physical rounds, energy traces and
a real capture driver remain unmeasured. The evidence is in
`tests/evidence/rig-esp32c6-physical*.json`. The manifest's image paths were
made relative so the Windows host could replay it; the image digests are
unchanged.

*All ESP32 targets.* Adding per-target defaults lets the unchanged firmware
build for ESP32, S2, S3, C2, C3, C6, H2 and P4. Each target uses its supported
CPU clock, and the C2 uses 2 MB flash. CI now builds the default image for
every target. Sensor pairs and production checks remain S3/C6 only, because
only those two targets have measured baselines. All 16 images (full and
production for each target) built in the pinned container; their digests are
in `tests/evidence/release-builds.json`. Only the C6 ran on silicon. The other
targets are cross-compiled only: no latency, memory or parity result is
claimed for them.

*Serial OTA.* The default partition table now holds two 960 KiB app slots and
enables bootloader rollback. The app also adds an `ota status` command and an
`ota begin <bytes> <sha256>` command. After `ota begin`, the host sends the
raw image in 1 KiB acknowledged blocks. The firmware then checks the file
digest, and `esp_ota_end` checks the image checksum, appended hash and chip
id. Only then does the new slot become bootable. A new slot boots as
PENDING_VERIFY. It is committed only if the model self-test passes; otherwise
the board rolls back to the previous slot and reboots. OTA uses the existing
UART, so it also works on H2 and P4, which have no Wi-Fi. It grants no
authority beyond the ROM serial bootloader on the same port. It adds no
network listener and no image signing. The core protocol is unchanged.
`app.c` gains only a weak `rd_app_extension` hook that host builds leave as
`unknown_command`.

On the physical C6, with pinned images:
- A 179,728-byte update transferred in about 21.5 s and was committed.
- An image whose golden self-test was altered booted and failed its
  self-test. The board rolled back to the previous valid slot on its own.
- These inputs were all rejected without changing the running slot:
  - wrong digest
  - an ESP32-C3 image (`ESP_ERR_OTA_VALIDATE_FAILED`)
  - a transfer cut off after 20 KiB (`ota_timeout`)
  - an oversize image
  - malformed and zero arguments
- The production profile keeps its measured 16,384-byte heap gain (440,308
  vs. 456,692 bytes free) and the same 174.5 µs default-model bench.

`tests/test_ota_tool.py` checks the host protocol against a simulated board.
In the pinned container, the software lab, 56 Python tests, 11 exact native
pairs, benchmark verification and the committed model header all pass.
Evidence: `tests/evidence/ota-esp32c6-physical.json`. Roll back to parent
`19376a6`.

## Follow up: exact hex-float input removes the C6 parse bottleneck

The physical C6 rig showed decimal parsing was more expensive than inference.
The chip has no FPU, so newlib `strtof` cost about 22 µs per value, while
inference took about 77 µs. `inferx` and `sensorx` now accept the same
values as 8-hex-digit IEEE-754 binary32 bit patterns, separated by spaces or
tabs. The MCU performs no decimal conversion, and the kernel receives exactly
the host's float. The accepted syntax is exactly eight hex digits per value.
Wrong length, non-hex characters, a `0x` prefix, NaN, infinities and a wrong
value count all return the existing error replies. The decimal `infer` and
`sensor` commands, the kernel and the reply format are unchanged.

`tests/test_hex_input.py` sends 401 random binary32 rows both as `%.9g`
decimal text, which round-trips exactly through `strtof`, and as hex. It
requires identical replies apart from timing. It also requires 11 malformed or
non-finite inputs to be rejected. On the physical C6, the pinned v5.4.4 image
with the UCI occupancy model replayed 1,000 held-out rows, interleaving both
commands row by row:
- 1,000 of 1,000 replies were identical.
- Both paths matched the reference equally: decision agreement 1.0 and
  acceptance agreement 0.999.
- Mean parse time fell from 110.3 µs to 8.0 µs, and the maximum from 575 µs
  to 24 µs.
- Mean on-chip request time (parse, preprocessing and inference) fell from
  197.6 µs to 95.0 µs, 2.08 times faster.
- The host round-trip median fell from 80.2 ms to 69.7 ms.

The round trip is now dominated by 115,200-baud UART transfer and JSON reply
formatting rather than MCU computation, so those are the next targets. The
evidence is in `tests/evidence/hexinput-esp32c6-physical.json`. Hosts that
need maximum request throughput should send `inferx` or `sensorx`. Decimal
input remains for people typing commands by hand.

## Follow up: USB-Serial/JTAG console for boards without a UART bridge

Many S3, C3, C6, H2 and P4 boards have only the chip's built-in
USB-Serial/JTAG port. Configuring with `-DRD_USB_SERIAL_JTAG=ON` adds the
driver to the component list and applies `sdkconfig.usbjtag`. The command
loop then installs the USB-Serial/JTAG driver, routes stdio through it, and
reads commands from it instead of UART0. The protocol, OTA, model and kernel
are unchanged. The option is opt-in because requiring the driver
unconditionally cost UART builds 986 bytes of static DIRAM and about 2 KB of
free heap on the C6. Kconfig values are not visible when IDF expands
component requirements, so the driver is linked after registration. UART
builds are byte-for-byte the previous configuration: the C6 has 76,791 DIRAM
bytes, and the physical C6 reports 440,308 bytes of free heap, a 174.6 µs
bench and a passing self-test.

All five USB-capable targets build with the option in the pinned container,
and CI builds them. **Hardware status:** none of these USB-console images has
run on silicon. The attached C6 is connected only through its CP210x UART
bridge, so this is build-only evidence until a board's native USB port is
connected.

## Follow up: remove idle latency from the command loop

With hex input, the C6 computed a request in about 95 µs, but each host round
trip still took about 70 ms. The loop called `uart_read_bytes` for a full
128-byte buffer with a 50 ms timeout, so every shorter command waited out the
timeout. It then called `vTaskDelay(1)`, adding one FreeRTOS tick. The loop now
blocks for the first byte and drains the bytes already buffered without
waiting. It still wakes every 50 ms, so OTA idle timeouts keep working. The
USB-Serial/JTAG path uses the same pattern.

On the physical C6, the pinned occupancy image replayed 1,000 `sensorx` rows
on the old and new loops:
- 1,000 of 1,000 replies were identical.
- The host round-trip median fell from 70.0 ms to 28.7 ms, 2.44 times faster.
- The p99 fell from 70.6 ms to 29.1 ms.
- On-chip parse and inference time were unchanged.

On the new loop, OTA to the production image committed in 21.5 s, and a
broken self-test image rolled back. Free heap held at 439,984 bytes across
2,000 warmed inferences. The remaining round trip mostly comes from the
115,200-baud link, which spends about 29 ms moving a request and its reply.
The evidence is in `tests/evidence/uartloop-esp32c6-physical.json`.

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
