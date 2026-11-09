// Node ESM smoke test for @ruvector/emergent-time (--target web build).
//
// The `web` target normally fetches the .wasm by URL; in Node we read the bytes
// off disk and hand them to `initSync({ module })`, which accepts raw bytes (it
// constructs the WebAssembly.Module internally). This proves the shipped `web`
// build loads and runs end-to-end under Node without a separate `nodejs` target.
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import init, {
  initSync,
  AgenticClock,
  StateDelta,
  WindowedDeltaClock,
  PageHinkleyDetector,
  LearnedWeights,
  TickClassJs,
  AgentHealthJs,
  version,
} from '../../../npm/packages/emergent-time/pkg/emergent_time_wasm.js';

const wasmPath = fileURLToPath(new URL('../../../npm/packages/emergent-time/pkg/emergent_time_wasm_bg.wasm', import.meta.url));
const bytes = await readFile(wasmPath);
initSync({ module: bytes });

const className = (obj, val) =>
  Object.keys(obj).find((k) => obj[k] === val) ?? String(val);

console.log(`emergent-time-wasm version: ${version()}`);

// --- AgenticClock: feed a healthy-then-thrash synthetic trace ----------------
const clock = new AgenticClock();
clock.setWindow(6);

// 12 healthy steps: belief/plan converge a little each step, progress rises,
// contradiction stays low.
console.log('\n--- healthy phase ---');
for (let i = 0; i < 12; i++) {
  // StateDelta(belief, memory, retrieval, goal, contradiction, plan,
  //            contradictionLevel, progress)
  const d = new StateDelta(0.08, 0.04, 0.05, 0.02, 0.0, 0.07, 0.05, 0.04);
  const tick = clock.tick(d);
  if (i === 11) {
    console.log(
      `  step ${i}: Δτ=${tick.deltaTime.toFixed(4)} class=${className(TickClassJs, tick.class)} ` +
        `ATI=${clock.ati.toFixed(3)} health=${className(AgentHealthJs, clock.health)}`,
    );
    console.log(`    reason: ${tick.reason}`);
  }
}

// 8 thrash steps: plan oscillates hard, retrieval destabilizes, contradiction
// climbs, progress stalls.
console.log('\n--- thrash phase ---');
let firstCollapseStep = -1;
for (let i = 0; i < 8; i++) {
  const cl = Math.min(0.1 + 0.12 * i, 0.95); // contradiction level climbing
  const d = new StateDelta(0.35, 0.1, 0.4, 0.25, 0.3, 0.8, cl, 0.0);
  const tick = clock.tick(d);
  const cls = className(TickClassJs, tick.class);
  if (firstCollapseStep < 0 && tick.class === TickClassJs.Collapse) firstCollapseStep = i;
  console.log(
    `  step ${i}: Δτ=${tick.deltaTime.toFixed(3)} class=${cls} ` +
      `ATI=${clock.ati.toFixed(3)} health=${className(AgentHealthJs, clock.health)} ` +
      `domreason="${tick.reason}"`,
  );
}

console.log('\n--- cumulative ---');
console.log(`  cumulativeTime=${clock.cumulativeTime.toFixed(3)}`);
console.log(`  cumulativeProgress=${clock.cumulativeProgress.toFixed(3)}`);
console.log(`  final health=${className(AgentHealthJs, clock.health)}`);

// --- Change-point detectors on a synthetic scalar stream ---------------------
// 20 stationary samples (~1.0 ± 0.05) then a sustained jump to ~3.0.
console.log('\n--- change-point detectors ---');
// std_floor ~ the stationary noise scale (0.05) so the near-constant phase does
// not trip a spurious infinite z-score — the fair-baseline discipline from the
// Rust crate's docs.
const wd = new WindowedDeltaClock(8, 4.0, 0.05);
const ph = new PageHinkleyDetector(0.1, 1.0);
const rng = (() => {
  let s = 42 >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s / 0xffffffff) * 2 - 1;
  };
})();
const stream = [];
for (let i = 0; i < 20; i++) stream.push(1.0 + 0.05 * rng());
const jumpAt = stream.length;
for (let i = 0; i < 20; i++) stream.push(3.0 + 0.05 * rng());
for (const x of stream) {
  wd.push(x);
  ph.push(x);
}
console.log(`  stream: 20 samples ~1.0, jump to ~3.0 at index ${jumpAt}`);
console.log(`  WindowedDeltaClock: alarmed=${wd.alarmed} alarmIndex=${wd.alarmIndex}`);
console.log(`  PageHinkleyDetector: alarmed=${ph.alarmed} alarmIndex=${ph.alarmIndex}`);

// --- Assertions: fail loudly if the wasm misbehaves --------------------------
let ok = true;
function check(cond, msg) {
  if (!cond) {
    ok = false;
    console.error(`  FAIL: ${msg}`);
  }
}
check(version().length > 0, 'version() returns a non-empty string');
check(clock.cumulativeTime > 0, 'cumulative agentic time advanced');
check(
  clock.health === AgentHealthJs.Collapsing ||
    clock.health === AgentHealthJs.NeedsHumanReview,
  'final health is a high-contradiction state',
);
check(wd.alarmed && wd.alarmIndex >= jumpAt, 'windowed detector fires at/after the jump');
check(ph.alarmed && ph.alarmIndex >= jumpAt, 'page-hinkley detector fires at/after the jump');

// --- Regressions (0.1.1) -----------------------------------------------------
// Shrinking the window must drop old samples (0.1.0 kept the old length).
const shrink = new AgenticClock();
shrink.setNoiseFloor(0);
for (let i = 0; i < 8; i++) shrink.tick(new StateDelta(0.1, 0, 0, 0, 0, 0, 0, -1));
shrink.setWindow(2);
for (let i = 0; i < 2; i++) shrink.tick(new StateDelta(0.1, 0, 0, 0, 0, 0, 0, 1));
check(Math.abs(shrink.ati - 10) < 1e-9, `setWindow shrink: ATI over last 2 ticks (got ${shrink.ati})`);
// One NaN sample must not poison the cumulative state.
const nanClock = new AgenticClock();
nanClock.tick(new StateDelta(NaN, Infinity, 0, NaN, 0, 0, NaN, NaN));
nanClock.tick(new StateDelta(0.1, 0, 0, 0, 0, 0, 0, 0.5));
check(Number.isFinite(nanClock.cumulativeTime) && Number.isFinite(nanClock.ati), 'NaN delta is ignored');
// A zero / non-finite std must be rejected rather than yield a NaN probability.
for (const bad of [0, NaN, -1]) {
  let threw = false;
  try {
    LearnedWeights.fromParams(2, [1, 1], 0, [0, 0], [1, bad]);
  } catch {
    threw = true;
  }
  check(threw, `fromParams rejects std=${bad}`);
}
const lw = LearnedWeights.fromParams(2, [1, -1], 0, [0, 0], [1, 1]);
const p = lw.predict([0.3, 0.1]);
check(p > 0 && p < 1, 'predict returns a probability');

// Avoid `init` being flagged unused (it is the default async entry point).
void init;

console.log(`\nSMOKE TEST: ${ok ? 'PASS' : 'FAIL'}`);
process.exit(ok ? 0 : 1);
