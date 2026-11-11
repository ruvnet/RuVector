// TypeScript usage example — compiled with `tsc --noEmit` to verify the shipped
// .d.ts is valid and the public API type-checks. This is documentation that the
// compiler enforces, mirrored in the README quickstart.
import {
  AgenticClock,
  StateDelta,
  WindowedDeltaClock,
  PageHinkleyDetector,
  LearnedWeights,
  TickClassJs,
  AgentHealthJs,
  fullFeatureDim,
  version,
} from '../../../npm/packages/emergent-time/pkg/emergent_time_wasm.js';

export function demo(): void {
  const v: string = version();

  const clock = new AgenticClock();
  clock.setWindow(8);
  clock.setNoiseFloor(1e-3);
  clock.setThresholds(1e-3, 0.5, 0.1, 0.5, 0.8);

  // StateDelta(belief, memory, retrieval, goal, contradiction, plan,
  //            contradictionLevel, progress)
  const delta = new StateDelta(0.3, 0.1, 0.4, 0.2, 0.3, 0.8, 0.6, 0.0);
  const tick = clock.tick(delta);

  const dt: number = tick.deltaTime;
  const cls: TickClassJs = tick.class;
  const reason: string = tick.reason;
  const ati: number = clock.ati;
  const health: AgentHealthJs = clock.health;
  const cumTime: number = clock.cumulativeTime;

  if (cls === TickClassJs.Collapse && health === AgentHealthJs.NeedsHumanReview) {
    // escalate
  }
  void [v, dt, reason, ati, cumTime];

  // Detectors.
  const wd = new WindowedDeltaClock(8, 4.0, 1.0);
  const z: number = wd.push(2.5);
  const wdAlarmed: boolean = wd.alarmed;
  const wdIdx: bigint = wd.alarmIndex;

  const ph = new PageHinkleyDetector(0.1, 1.0);
  const stat: number = ph.push(2.5);
  const phAlarmed: boolean = ph.alarmed;
  void [z, wdAlarmed, wdIdx, stat, phAlarmed];

  // Learned weights (inference of an offline-trained model).
  const dim: number = fullFeatureDim();
  const lw = LearnedWeights.fromParams(
    dim,
    new Float64Array(dim).fill(0.1),
    0.0,
    new Float64Array(dim).fill(0.0),
    new Float64Array(dim).fill(1.0),
  );
  const p: number = lw.predict(new Float64Array(dim).fill(0.5));
  const w: Float64Array = lw.clockWeights();
  void [p, w];
}
