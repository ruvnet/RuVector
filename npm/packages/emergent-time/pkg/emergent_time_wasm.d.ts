/* tslint:disable */
/* eslint-disable */

/**
 * The 7-state agent health verdict (mirrors the Rust `AgentHealth`).
 */
export enum AgentHealthJs {
    /**
     * Progress is keeping pace with internal change.
     */
    Healthy = 0,
    /**
     * Moving, but inefficiently (low progress per unit change).
     */
    Drifting = 1,
    /**
     * Neither changing nor progressing.
     */
    Stuck = 2,
    /**
     * Lots of internal churn, no progress — replan.
     */
    NeedsReplan = 3,
    /**
     * Losing ground (progress going backwards).
     */
    Contradicting = 4,
    /**
     * Contradiction is high and rising.
     */
    Collapsing = 5,
    /**
     * Contradiction is critical — escalate to a human.
     */
    NeedsHumanReview = 6,
}

/**
 * A stateful agentic-time clock. Construct it (optionally with custom channel
 * weights / health thresholds), feed transitions via [`AgenticClock::tick`], and
 * read back cumulative agentic time, the ATI, and the health classification.
 *
 * The clock keeps a small amount of running state: cumulative agentic time,
 * cumulative progress, and a rolling window of the last `window` ticks /
 * progress used for the ATI and health classification.
 */
export class AgenticClock {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Construct a clock with the default channel weights (contradiction 1.5,
     * belief / goal / plan 1.0, memory / retrieval 0.5), default health
     * thresholds, a noise floor of `1e-3`, and a window of 8 ticks.
     */
    constructor();
    /**
     * Reset all running state (cumulative time / progress, window) to zero,
     * keeping the configured weights, thresholds, noise floor, and window length.
     */
    reset(): void;
    /**
     * Override the noise floor (jitter suppression). Ticks whose raw channel sum
     * is below this floor report `deltaTime == 0`.
     */
    setNoiseFloor(floor: number): void;
    /**
     * Override the health-classifier thresholds. Order:
     * `idle, healthyAti, driftingAti, collapse, humanReview`.
     */
    setThresholds(idle: number, healthy_ati: number, drifting_ati: number, collapse: number, human_review: number): void;
    /**
     * Override the rolling window length used for the ATI and health (default 8).
     */
    setWindow(window: number): void;
    /**
     * Feed one transition's channel deltas and return the explainable [`Tick`].
     * Advances the clock's cumulative agentic time and progress, and updates the
     * rolling window used by [`AgenticClock::ati`] and
     * [`AgenticClock::health`].
     */
    tick(delta: StateDelta): Tick;
    /**
     * Construct a clock with custom channel weights. Order:
     * `belief, memory, retrieval, goalGraph, contradiction, plan`.
     */
    static withWeights(belief: number, memory: number, retrieval: number, goal_graph: number, contradiction: number, plan: number): AgenticClock;
    /**
     * The Agentic Time Index over the current window: progress per unit of
     * structural change. High ATI ⇒ learning and moving; near-zero ⇒ spinning;
     * `Infinity` ⇒ progressing with no internal change.
     */
    readonly ati: number;
    /**
     * Cumulative progress accrued across all ticks so far.
     */
    readonly cumulativeProgress: number;
    /**
     * Cumulative agentic time accrued across all ticks so far.
     */
    readonly cumulativeTime: number;
    /**
     * The current health verdict, classified over the rolling window of agentic
     * time, progress, and the latest contradiction level.
     */
    readonly health: AgentHealthJs;
}

/**
 * A fitted logistic-regression scorer over the channel-movement features (the
 * crate's `LearnedWeights`). Reconstruct it from persisted parameters and score
 * raw feature vectors to get a failure-approach probability in `[0, 1]`.
 *
 * The training harness lives in the Rust crate; this binding exposes only the
 * cheap inference path (`predict`) so a model trained offline can run in the
 * browser without bundling the trainer.
 */
export class LearnedWeights {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Non-negative clock weights derived from the learned coefficients (the
     * positive part), suitable for `AgenticClock.withWeights(...)`.
     */
    clockWeights(): Float64Array;
    /**
     * Reconstruct a scorer from persisted parameters. `coef`, `mean`, and `std`
     * must each have length `dim` (the feature count: 6 for the full channel set,
     * 5 for the contradiction-free "honest" set).
     */
    static fromParams(dim: number, coef: Float64Array, bias: number, mean: Float64Array, std: Float64Array): LearnedWeights;
    /**
     * Predicted failure-approach probability in `[0, 1]` for a raw feature vector
     * (the per-channel movements in feature order). `features.length` must equal
     * the model's `dim`.
     */
    predict(features: Float64Array): number;
    /**
     * The model's feature dimensionality.
     */
    readonly dim: number;
}

/**
 * A **Page–Hinkley** adaptive change-point detector: a CUSUM test whose
 * reference is a *running* mean (so a noisy early phase does not permanently
 * raise the bar). Push a scalar each step, get back the current PH statistic and
 * an alarm flag. The adaptive counterpart of [`WindowedDeltaClock`]; both are the
 * fair competitors to the agentic clock.
 */
export class PageHinkleyDetector {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Construct a **downward** (decrease-detecting) Page–Hinkley detector.
     */
    static downward(delta: number, lambda: number): PageHinkleyDetector;
    /**
     * Construct an **upward** (increase-detecting) Page–Hinkley detector with
     * tolerance `delta` (deviations below this are treated as normal jitter) and
     * alarm threshold `lambda` (larger ⇒ fewer false alarms, later detection).
     */
    constructor(delta: number, lambda: number);
    /**
     * Push the next scalar and return the current Page–Hinkley statistic (the
     * rise above the running minimum for the upward form, or the drop below the
     * running maximum for the downward form). Updates [`alarmed`](Self::alarmed)
     * when the statistic exceeds `lambda`.
     */
    push(value: number): number;
    /**
     * Reset the detector's running statistics and alarm latch.
     */
    reset(): void;
    /**
     * The 0-based index at which the detector first fired, or `-1` if it has not.
     */
    readonly alarmIndex: bigint;
    /**
     * Whether the detector has fired (latched true on first alarm).
     */
    readonly alarmed: boolean;
}

/**
 * The six per-transition channel deltas fed to [`AgenticClock::tick`].
 *
 * Each field is the **already-computed scalar movement** of that channel over a
 * transition (e.g. the L2 distance between successive belief embeddings, or the
 * absolute change in a contradiction score). Keeping the JS boundary scalar — six
 * numbers, not six embedding vectors — keeps the wasm tiny and lets the caller
 * pick whatever distance metric and embedding model they like on the JS side.
 *
 * `contradictionLevel` is the *current absolute* contradiction in `[0, 1]` (not a
 * delta); it drives the collapse / human-review health states.
 */
export class StateDelta {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Construct a transition's channel deltas.
     *
     * * `belief`, `memory`, `retrieval`, `plan` — non-negative scalar movements
     *   (typically L2 distance between successive embeddings).
     * * `goal` — absolute change in goal-graph mass (e.g. open-subgoal count).
     * * `contradiction` — absolute change in the contradiction score.
     * * `contradictionLevel` — current absolute contradiction in `[0, 1]`.
     * * `progress` — absolute change in task progress over this transition
     *   (used for the ATI and health classification; pass `0` if unknown).
     *
     * Non-finite inputs (`NaN`, `±Infinity`) are treated as `0` so one bad
     * sample cannot poison the clock's cumulative and windowed state.
     */
    constructor(belief: number, memory: number, retrieval: number, goal: number, contradiction: number, plan: number, contradiction_level: number, progress: number);
}

/**
 * An explainable agentic-time tick: the post-floor internal-time increment, its
 * class, a human-readable reason, and the raw (pre-floor) per-channel weighted
 * contributions. See the Rust crate's `Tick` docs for the post-floor /
 * pre-floor contract: `deltaTime == Σ channels` only when `noiseFloor == 0`.
 */
export class Tick {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Raw (pre-floor) weighted belief contribution.
     */
    readonly belief: number;
    /**
     * The tick class (Idle / Progress / Learning / Contradiction / Collapse).
     */
    readonly class: TickClassJs;
    /**
     * Raw (pre-floor) weighted contradiction contribution.
     */
    readonly contradiction: number;
    /**
     * Post-floor internal-time magnitude: `max(0, Σ channels − noiseFloor)`.
     */
    readonly deltaTime: number;
    /**
     * Raw (pre-floor) weighted goal-graph contribution.
     */
    readonly goalGraph: number;
    /**
     * Raw (pre-floor) weighted memory contribution.
     */
    readonly memory: number;
    /**
     * Raw (pre-floor) weighted plan contribution.
     */
    readonly plan: number;
    /**
     * A human-readable audit string explaining which channel dominated.
     */
    readonly reason: string;
    /**
     * Raw (pre-floor) weighted retrieval contribution.
     */
    readonly retrieval: number;
}

/**
 * One agentic-time class (mirrors the Rust `TickClass`). Exposed as a small
 * enum so the JS side can `switch` on it without string parsing.
 */
export enum TickClassJs {
    /**
     * Below the noise floor — no meaningful change.
     */
    Idle = 0,
    /**
     * Belief / plan / goal moved forward.
     */
    Progress = 1,
    /**
     * New information arrived (retrieval / memory moved).
     */
    Learning = 2,
    /**
     * Contradiction rose.
     */
    Contradiction = 3,
    /**
     * Contradiction is high — failure regime.
     */
    Collapse = 4,
}

/**
 * A **windowed z-score** change-point detector (rolling `mean + kσ`): push a
 * scalar each step, get back the z-score and an alarm flag. This is the *fair
 * baseline* the agentic clock is honestly compared against — a cheap one-signal
 * detector a practitioner would actually deploy.
 */
export class WindowedDeltaClock {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Construct a detector with a trailing `window`, a `kSigma` alarm multiplier
     * (e.g. 4.0), and a `stdFloor` variance floor that prevents a near-constant
     * stream from producing a spurious infinite z-score.
     */
    constructor(window: number, k_sigma: number, std_floor: number);
    /**
     * Push the next scalar observable and return its rolling z-score (deviation
     * from the trailing-window mean over the floored window std). Updates
     * [`alarmed`](Self::alarmed) when the z-score exceeds `kSigma`.
     */
    push(value: number): number;
    /**
     * Reset the detector's history and alarm latch.
     */
    reset(): void;
    /**
     * The 0-based index at which the detector first fired, or `-1` if it has not.
     */
    readonly alarmIndex: bigint;
    /**
     * Whether the detector has fired (latched true on first alarm).
     */
    readonly alarmed: boolean;
}

/**
 * The number of channel features for the full set (6) — for sizing
 * [`LearnedWeights`] parameter arrays.
 */
export function fullFeatureDim(): number;

/**
 * The number of channel features for the contradiction-free "honest" set (5).
 */
export function honestFeatureDim(): number;

/**
 * Optional: route Rust panics to the JS console with a readable message.
 * Call once after instantiation. No-op cost if never called.
 */
export function setPanicHook(): void;

/**
 * The package version (compile-time constant from Cargo).
 */
export function version(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_agenticclock_free: (a: number, b: number) => void;
    readonly __wbg_learnedweights_free: (a: number, b: number) => void;
    readonly __wbg_pagehinkleydetector_free: (a: number, b: number) => void;
    readonly __wbg_statedelta_free: (a: number, b: number) => void;
    readonly __wbg_tick_free: (a: number, b: number) => void;
    readonly __wbg_windoweddeltaclock_free: (a: number, b: number) => void;
    readonly agenticclock_ati: (a: number) => number;
    readonly agenticclock_cumulativeProgress: (a: number) => number;
    readonly agenticclock_cumulativeTime: (a: number) => number;
    readonly agenticclock_health: (a: number) => number;
    readonly agenticclock_new: () => number;
    readonly agenticclock_reset: (a: number) => void;
    readonly agenticclock_setNoiseFloor: (a: number, b: number) => void;
    readonly agenticclock_setThresholds: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly agenticclock_setWindow: (a: number, b: number) => void;
    readonly agenticclock_tick: (a: number, b: number) => number;
    readonly agenticclock_withWeights: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly learnedweights_clockWeights: (a: number, b: number) => void;
    readonly learnedweights_dim: (a: number) => number;
    readonly learnedweights_fromParams: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly learnedweights_predict: (a: number, b: number, c: number, d: number) => void;
    readonly pagehinkleydetector_alarmIndex: (a: number) => bigint;
    readonly pagehinkleydetector_alarmed: (a: number) => number;
    readonly pagehinkleydetector_downward: (a: number, b: number) => number;
    readonly pagehinkleydetector_new: (a: number, b: number) => number;
    readonly pagehinkleydetector_push: (a: number, b: number) => number;
    readonly pagehinkleydetector_reset: (a: number) => void;
    readonly statedelta_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
    readonly tick_belief: (a: number) => number;
    readonly tick_class: (a: number) => number;
    readonly tick_contradiction: (a: number) => number;
    readonly tick_deltaTime: (a: number) => number;
    readonly tick_goalGraph: (a: number) => number;
    readonly tick_memory: (a: number) => number;
    readonly tick_plan: (a: number) => number;
    readonly tick_reason: (a: number, b: number) => void;
    readonly tick_retrieval: (a: number) => number;
    readonly version: (a: number) => void;
    readonly windoweddeltaclock_alarmIndex: (a: number) => bigint;
    readonly windoweddeltaclock_alarmed: (a: number) => number;
    readonly windoweddeltaclock_new: (a: number, b: number, c: number) => number;
    readonly windoweddeltaclock_push: (a: number, b: number) => number;
    readonly windoweddeltaclock_reset: (a: number) => void;
    readonly setPanicHook: () => void;
    readonly fullFeatureDim: () => number;
    readonly honestFeatureDim: () => number;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export2: (a: number, b: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
