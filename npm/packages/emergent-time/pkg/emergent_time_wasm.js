/* @ts-self-types="./emergent_time_wasm.d.ts" */

/**
 * The 7-state agent health verdict (mirrors the Rust `AgentHealth`).
 * @enum {0 | 1 | 2 | 3 | 4 | 5 | 6}
 */
export const AgentHealthJs = Object.freeze({
    /**
     * Progress is keeping pace with internal change.
     */
    Healthy: 0, "0": "Healthy",
    /**
     * Moving, but inefficiently (low progress per unit change).
     */
    Drifting: 1, "1": "Drifting",
    /**
     * Neither changing nor progressing.
     */
    Stuck: 2, "2": "Stuck",
    /**
     * Lots of internal churn, no progress — replan.
     */
    NeedsReplan: 3, "3": "NeedsReplan",
    /**
     * Losing ground (progress going backwards).
     */
    Contradicting: 4, "4": "Contradicting",
    /**
     * Contradiction is high and rising.
     */
    Collapsing: 5, "5": "Collapsing",
    /**
     * Contradiction is critical — escalate to a human.
     */
    NeedsHumanReview: 6, "6": "NeedsHumanReview",
});

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
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(AgenticClock.prototype);
        obj.__wbg_ptr = ptr;
        AgenticClockFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        AgenticClockFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_agenticclock_free(ptr, 0);
    }
    /**
     * The Agentic Time Index over the current window: progress per unit of
     * structural change. High ATI ⇒ learning and moving; near-zero ⇒ spinning;
     * `Infinity` ⇒ progressing with no internal change.
     * @returns {number}
     */
    get ati() {
        const ret = wasm.agenticclock_ati(this.__wbg_ptr);
        return ret;
    }
    /**
     * Cumulative progress accrued across all ticks so far.
     * @returns {number}
     */
    get cumulativeProgress() {
        const ret = wasm.agenticclock_cumulativeProgress(this.__wbg_ptr);
        return ret;
    }
    /**
     * Cumulative agentic time accrued across all ticks so far.
     * @returns {number}
     */
    get cumulativeTime() {
        const ret = wasm.agenticclock_cumulativeTime(this.__wbg_ptr);
        return ret;
    }
    /**
     * The current health verdict, classified over the rolling window of agentic
     * time, progress, and the latest contradiction level.
     * @returns {AgentHealthJs}
     */
    get health() {
        const ret = wasm.agenticclock_health(this.__wbg_ptr);
        return ret;
    }
    /**
     * Construct a clock with the default channel weights (contradiction 1.5,
     * belief / goal / plan 1.0, memory / retrieval 0.5), default health
     * thresholds, a noise floor of `1e-3`, and a window of 8 ticks.
     */
    constructor() {
        const ret = wasm.agenticclock_new();
        this.__wbg_ptr = ret >>> 0;
        AgenticClockFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Reset all running state (cumulative time / progress, window) to zero,
     * keeping the configured weights, thresholds, noise floor, and window length.
     */
    reset() {
        wasm.agenticclock_reset(this.__wbg_ptr);
    }
    /**
     * Override the noise floor (jitter suppression). Ticks whose raw channel sum
     * is below this floor report `deltaTime == 0`.
     * @param {number} floor
     */
    setNoiseFloor(floor) {
        wasm.agenticclock_setNoiseFloor(this.__wbg_ptr, floor);
    }
    /**
     * Override the health-classifier thresholds. Order:
     * `idle, healthyAti, driftingAti, collapse, humanReview`.
     * @param {number} idle
     * @param {number} healthy_ati
     * @param {number} drifting_ati
     * @param {number} collapse
     * @param {number} human_review
     */
    setThresholds(idle, healthy_ati, drifting_ati, collapse, human_review) {
        wasm.agenticclock_setThresholds(this.__wbg_ptr, idle, healthy_ati, drifting_ati, collapse, human_review);
    }
    /**
     * Override the rolling window length used for the ATI and health (default 8).
     * @param {number} window
     */
    setWindow(window) {
        wasm.agenticclock_setWindow(this.__wbg_ptr, window);
    }
    /**
     * Feed one transition's channel deltas and return the explainable [`Tick`].
     * Advances the clock's cumulative agentic time and progress, and updates the
     * rolling window used by [`AgenticClock::ati`] and
     * [`AgenticClock::health`].
     * @param {StateDelta} delta
     * @returns {Tick}
     */
    tick(delta) {
        _assertClass(delta, StateDelta);
        const ret = wasm.agenticclock_tick(this.__wbg_ptr, delta.__wbg_ptr);
        return Tick.__wrap(ret);
    }
    /**
     * Construct a clock with custom channel weights. Order:
     * `belief, memory, retrieval, goalGraph, contradiction, plan`.
     * @param {number} belief
     * @param {number} memory
     * @param {number} retrieval
     * @param {number} goal_graph
     * @param {number} contradiction
     * @param {number} plan
     * @returns {AgenticClock}
     */
    static withWeights(belief, memory, retrieval, goal_graph, contradiction, plan) {
        const ret = wasm.agenticclock_withWeights(belief, memory, retrieval, goal_graph, contradiction, plan);
        return AgenticClock.__wrap(ret);
    }
}
if (Symbol.dispose) AgenticClock.prototype[Symbol.dispose] = AgenticClock.prototype.free;

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
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(LearnedWeights.prototype);
        obj.__wbg_ptr = ptr;
        LearnedWeightsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        LearnedWeightsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_learnedweights_free(ptr, 0);
    }
    /**
     * Non-negative clock weights derived from the learned coefficients (the
     * positive part), suitable for `AgenticClock.withWeights(...)`.
     * @returns {Float64Array}
     */
    clockWeights() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.learnedweights_clockWeights(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayF64FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export(r0, r1 * 8, 8);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * The model's feature dimensionality.
     * @returns {number}
     */
    get dim() {
        const ret = wasm.learnedweights_dim(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Reconstruct a scorer from persisted parameters. `coef`, `mean`, and `std`
     * must each have length `dim` (the feature count: 6 for the full channel set,
     * 5 for the contradiction-free "honest" set).
     * @param {number} dim
     * @param {Float64Array} coef
     * @param {number} bias
     * @param {Float64Array} mean
     * @param {Float64Array} std
     * @returns {LearnedWeights}
     */
    static fromParams(dim, coef, bias, mean, std) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF64ToWasm0(coef, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArrayF64ToWasm0(mean, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            const ptr2 = passArrayF64ToWasm0(std, wasm.__wbindgen_export2);
            const len2 = WASM_VECTOR_LEN;
            wasm.learnedweights_fromParams(retptr, dim, ptr0, len0, bias, ptr1, len1, ptr2, len2);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return LearnedWeights.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Predicted failure-approach probability in `[0, 1]` for a raw feature vector
     * (the per-channel movements in feature order). `features.length` must equal
     * the model's `dim`.
     * @param {Float64Array} features
     * @returns {number}
     */
    predict(features) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF64ToWasm0(features, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.learnedweights_predict(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getFloat64(retptr + 8 * 0, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            if (r3) {
                throw takeObject(r2);
            }
            return r0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) LearnedWeights.prototype[Symbol.dispose] = LearnedWeights.prototype.free;

/**
 * A **Page–Hinkley** adaptive change-point detector: a CUSUM test whose
 * reference is a *running* mean (so a noisy early phase does not permanently
 * raise the bar). Push a scalar each step, get back the current PH statistic and
 * an alarm flag. The adaptive counterpart of [`WindowedDeltaClock`]; both are the
 * fair competitors to the agentic clock.
 */
export class PageHinkleyDetector {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(PageHinkleyDetector.prototype);
        obj.__wbg_ptr = ptr;
        PageHinkleyDetectorFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PageHinkleyDetectorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_pagehinkleydetector_free(ptr, 0);
    }
    /**
     * The 0-based index at which the detector first fired, or `-1` if it has not.
     * @returns {bigint}
     */
    get alarmIndex() {
        const ret = wasm.pagehinkleydetector_alarmIndex(this.__wbg_ptr);
        return ret;
    }
    /**
     * Whether the detector has fired (latched true on first alarm).
     * @returns {boolean}
     */
    get alarmed() {
        const ret = wasm.pagehinkleydetector_alarmed(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Construct a **downward** (decrease-detecting) Page–Hinkley detector.
     * @param {number} delta
     * @param {number} lambda
     * @returns {PageHinkleyDetector}
     */
    static downward(delta, lambda) {
        const ret = wasm.pagehinkleydetector_downward(delta, lambda);
        return PageHinkleyDetector.__wrap(ret);
    }
    /**
     * Construct an **upward** (increase-detecting) Page–Hinkley detector with
     * tolerance `delta` (deviations below this are treated as normal jitter) and
     * alarm threshold `lambda` (larger ⇒ fewer false alarms, later detection).
     * @param {number} delta
     * @param {number} lambda
     */
    constructor(delta, lambda) {
        const ret = wasm.pagehinkleydetector_new(delta, lambda);
        this.__wbg_ptr = ret >>> 0;
        PageHinkleyDetectorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Push the next scalar and return the current Page–Hinkley statistic (the
     * rise above the running minimum for the upward form, or the drop below the
     * running maximum for the downward form). Updates [`alarmed`](Self::alarmed)
     * when the statistic exceeds `lambda`.
     * @param {number} value
     * @returns {number}
     */
    push(value) {
        const ret = wasm.pagehinkleydetector_push(this.__wbg_ptr, value);
        return ret;
    }
    /**
     * Reset the detector's running statistics and alarm latch.
     */
    reset() {
        wasm.pagehinkleydetector_reset(this.__wbg_ptr);
    }
}
if (Symbol.dispose) PageHinkleyDetector.prototype[Symbol.dispose] = PageHinkleyDetector.prototype.free;

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
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        StateDeltaFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_statedelta_free(ptr, 0);
    }
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
     * @param {number} belief
     * @param {number} memory
     * @param {number} retrieval
     * @param {number} goal
     * @param {number} contradiction
     * @param {number} plan
     * @param {number} contradiction_level
     * @param {number} progress
     */
    constructor(belief, memory, retrieval, goal, contradiction, plan, contradiction_level, progress) {
        const ret = wasm.statedelta_new(belief, memory, retrieval, goal, contradiction, plan, contradiction_level, progress);
        this.__wbg_ptr = ret >>> 0;
        StateDeltaFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) StateDelta.prototype[Symbol.dispose] = StateDelta.prototype.free;

/**
 * An explainable agentic-time tick: the post-floor internal-time increment, its
 * class, a human-readable reason, and the raw (pre-floor) per-channel weighted
 * contributions. See the Rust crate's `Tick` docs for the post-floor /
 * pre-floor contract: `deltaTime == Σ channels` only when `noiseFloor == 0`.
 */
export class Tick {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Tick.prototype);
        obj.__wbg_ptr = ptr;
        TickFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TickFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_tick_free(ptr, 0);
    }
    /**
     * Raw (pre-floor) weighted belief contribution.
     * @returns {number}
     */
    get belief() {
        const ret = wasm.tick_belief(this.__wbg_ptr);
        return ret;
    }
    /**
     * The tick class (Idle / Progress / Learning / Contradiction / Collapse).
     * @returns {TickClassJs}
     */
    get class() {
        const ret = wasm.tick_class(this.__wbg_ptr);
        return ret;
    }
    /**
     * Raw (pre-floor) weighted contradiction contribution.
     * @returns {number}
     */
    get contradiction() {
        const ret = wasm.tick_contradiction(this.__wbg_ptr);
        return ret;
    }
    /**
     * Post-floor internal-time magnitude: `max(0, Σ channels − noiseFloor)`.
     * @returns {number}
     */
    get deltaTime() {
        const ret = wasm.tick_deltaTime(this.__wbg_ptr);
        return ret;
    }
    /**
     * Raw (pre-floor) weighted goal-graph contribution.
     * @returns {number}
     */
    get goalGraph() {
        const ret = wasm.tick_goalGraph(this.__wbg_ptr);
        return ret;
    }
    /**
     * Raw (pre-floor) weighted memory contribution.
     * @returns {number}
     */
    get memory() {
        const ret = wasm.tick_memory(this.__wbg_ptr);
        return ret;
    }
    /**
     * Raw (pre-floor) weighted plan contribution.
     * @returns {number}
     */
    get plan() {
        const ret = wasm.tick_plan(this.__wbg_ptr);
        return ret;
    }
    /**
     * A human-readable audit string explaining which channel dominated.
     * @returns {string}
     */
    get reason() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.tick_reason(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Raw (pre-floor) weighted retrieval contribution.
     * @returns {number}
     */
    get retrieval() {
        const ret = wasm.tick_retrieval(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Tick.prototype[Symbol.dispose] = Tick.prototype.free;

/**
 * One agentic-time class (mirrors the Rust `TickClass`). Exposed as a small
 * enum so the JS side can `switch` on it without string parsing.
 * @enum {0 | 1 | 2 | 3 | 4}
 */
export const TickClassJs = Object.freeze({
    /**
     * Below the noise floor — no meaningful change.
     */
    Idle: 0, "0": "Idle",
    /**
     * Belief / plan / goal moved forward.
     */
    Progress: 1, "1": "Progress",
    /**
     * New information arrived (retrieval / memory moved).
     */
    Learning: 2, "2": "Learning",
    /**
     * Contradiction rose.
     */
    Contradiction: 3, "3": "Contradiction",
    /**
     * Contradiction is high — failure regime.
     */
    Collapse: 4, "4": "Collapse",
});

/**
 * A **windowed z-score** change-point detector (rolling `mean + kσ`): push a
 * scalar each step, get back the z-score and an alarm flag. This is the *fair
 * baseline* the agentic clock is honestly compared against — a cheap one-signal
 * detector a practitioner would actually deploy.
 */
export class WindowedDeltaClock {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WindowedDeltaClockFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_windoweddeltaclock_free(ptr, 0);
    }
    /**
     * The 0-based index at which the detector first fired, or `-1` if it has not.
     * @returns {bigint}
     */
    get alarmIndex() {
        const ret = wasm.windoweddeltaclock_alarmIndex(this.__wbg_ptr);
        return ret;
    }
    /**
     * Whether the detector has fired (latched true on first alarm).
     * @returns {boolean}
     */
    get alarmed() {
        const ret = wasm.windoweddeltaclock_alarmed(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Construct a detector with a trailing `window`, a `kSigma` alarm multiplier
     * (e.g. 4.0), and a `stdFloor` variance floor that prevents a near-constant
     * stream from producing a spurious infinite z-score.
     * @param {number} window
     * @param {number} k_sigma
     * @param {number} std_floor
     */
    constructor(window, k_sigma, std_floor) {
        const ret = wasm.windoweddeltaclock_new(window, k_sigma, std_floor);
        this.__wbg_ptr = ret >>> 0;
        WindowedDeltaClockFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Push the next scalar observable and return its rolling z-score (deviation
     * from the trailing-window mean over the floored window std). Updates
     * [`alarmed`](Self::alarmed) when the z-score exceeds `kSigma`.
     * @param {number} value
     * @returns {number}
     */
    push(value) {
        const ret = wasm.windoweddeltaclock_push(this.__wbg_ptr, value);
        return ret;
    }
    /**
     * Reset the detector's history and alarm latch.
     */
    reset() {
        wasm.windoweddeltaclock_reset(this.__wbg_ptr);
    }
}
if (Symbol.dispose) WindowedDeltaClock.prototype[Symbol.dispose] = WindowedDeltaClock.prototype.free;

/**
 * The number of channel features for the full set (6) — for sizing
 * [`LearnedWeights`] parameter arrays.
 * @returns {number}
 */
export function fullFeatureDim() {
    const ret = wasm.fullFeatureDim();
    return ret >>> 0;
}

/**
 * The number of channel features for the contradiction-free "honest" set (5).
 * @returns {number}
 */
export function honestFeatureDim() {
    const ret = wasm.honestFeatureDim();
    return ret >>> 0;
}

/**
 * Optional: route Rust panics to the JS console with a readable message.
 * Call once after instantiation. No-op cost if never called.
 */
export function setPanicHook() {
    wasm.setPanicHook();
}

/**
 * The package version (compile-time constant from Cargo).
 * @returns {string}
 */
export function version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.version(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
    }
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_960c155d3d49e4c2: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return addHeapObject(ret);
        },
        __wbg___wbindgen_throw_6b64449b9b9ed33c: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
    };
    return {
        __proto__: null,
        "./emergent_time_wasm_bg.js": import0,
    };
}

const AgenticClockFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_agenticclock_free(ptr >>> 0, 1));
const LearnedWeightsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_learnedweights_free(ptr >>> 0, 1));
const PageHinkleyDetectorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_pagehinkleydetector_free(ptr >>> 0, 1));
const StateDeltaFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_statedelta_free(ptr >>> 0, 1));
const TickFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_tick_free(ptr >>> 0, 1));
const WindowedDeltaClockFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_windoweddeltaclock_free(ptr >>> 0, 1));

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function dropObject(idx) {
    if (idx < 1028) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getObject(idx) { return heap[idx]; }

let heap = new Array(1024).fill(undefined);
heap.push(undefined, null, true, false);

let heap_next = heap.length;

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('emergent_time_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
