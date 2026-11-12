/**
 * AI4AI-Bench recursive-improvement adapter — first shippable slice
 * (PIR WP25, ADR-328).
 *
 * Source benchmark, cited in full on every reference (same naming discipline
 * as ADR-324 §5): AI4AI-Bench (arXiv:2608.20318, "AI4AI-Bench: Benchmarking
 * LLM Agents in Algorithmic Design for Recursive Self-Improvement";
 * github.com/Einsia/AI4AI-Bench, Apache-2.0). The benchmark freezes ten
 * research repositories spanning ten learning-algorithm families and asks an
 * agent to rewrite the learning algorithm itself; hyperparameter tuning and
 * data collection do not count as genuine improvement.
 *
 * WHAT THIS SLICE IS: the manifest/identity discipline, executor seam, score
 * ingest, and mutation-lineage chain that let MetaHarness treat AI4AI-Bench
 * (arXiv:2608.20318) tasks as an EXTERNAL published benchmark — exactly the
 * "published-benchmark" evidence kind ADR-324's external-grounding invariant
 * admits. Every candidate mutation this adapter observes carries reproducible
 * provenance (task identity → parent digest → diff digest → seed → executor
 * identity), so a run's full lineage can be replayed and audited.
 *
 * WHAT THIS SLICE IS NOT (honest scope): a real AI4AI-Bench run executes a
 * frozen repository's training pipeline inside the benchmark's own Docker
 * evaluator on datacenter-GPU hardware (the paper's official runs assume one
 * B300-class GPU). None of that happens here. The adapter takes a
 * CONSTRUCTOR-INJECTED executor (the same seam as
 * `RuvectorFlywheelOptions.benchmark` / `BenchmarkRunOptions.commandPrefixArgs`)
 * and this repo's CI exercises it only against a first-party fixture. The
 * command executor below spawns whatever evaluator entrypoint the caller
 * pins — locally that is a fixture; on real hardware it is the AI4AI-Bench
 * (arXiv:2608.20318) evaluator.
 *
 * PREPRINT-REPRODUCTION RULE (binding, unchanged from Waves 1–3): the paper's
 * published numbers — best 0.250, mean 0.166 across evaluated systems — are
 * HYPOTHESES and targets, never this program's acceptance bar. Promotion of
 * any mutation this adapter observes still routes exclusively through
 * statistics.ts `pairedBootstrapDecision` and the vetoes.ts conjunctive veto
 * algebra; an AI4AI score above 0.250 proves nothing by itself. The score
 * scale matters when reading results: 0.1 is the frozen repository's shipped
 * algorithm, so 0.250 closes less than a fifth of the ship-to-optimum
 * distance.
 *
 * FAIL-LOUD METRIC INTEGRITY (Wave-3 lesson, #888): an executor crash or a
 * non-finite / out-of-range score THROWS — it is never coerced into an
 * observation. NaN defeats every `<` comparison silently, so finiteness is
 * rejected at the choke point before any gate can see the value. Genuine-vs-
 * tuning classification is ingested only when the evaluator output carries
 * it; this adapter never fabricates the distinction.
 *
 * NO CACHING: a real AI4AI-Bench (arXiv:2608.20318) evaluation retrains a
 * stochastic pipeline; replaying a cached score as if it were a fresh run
 * would be evidence laundering. Every call to the executor is a real call.
 *
 * FROZEN WEIGHTS ARE STRUCTURAL: this file lives inside the mutation-surface
 * directory scripts/frozen-weights-check.mjs scans. It imports no weight-
 * updating API and names no model weight file. Rewriting a frozen BENCHMARK
 * repository's learning algorithm is the benchmark's task definition executed
 * by an external evaluator on external hardware — it is not a weight update
 * performed by this harness, and no path in this module persists weights.
 *
 * TRUST SCOPE (binding, stated once here and again on `Ai4aiEvidencePolicy`):
 * this module provides WITHIN-PROCESS INTEGRITY — a record cannot misreport
 * what the adapter's own executor produced. It is NOT cross-party
 * authentication: the lineage digest is unkeyed and nothing is signed, so an
 * adversary controlling the process can fabricate a chain that verifies.
 * Signing plus an issued-nonce ledger would close that; both are follow-up.
 *
 * KNOWN LIMITATION (accepted, exact parity with benchmark.ts): the command
 * executor SIGKILLs only the spawned child, not its process group — an
 * evaluator that forks can leave orphans past the timeout. Inherited from
 * benchmark.ts's runner discipline; fixing it belongs to both call sites at
 * once, not to this adapter alone.
 */
import { spawn } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { mkdtemp, open, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pairedBootstrapDecision, type PairedDecision } from "./statistics.js";
import type { PromotionVetoProvider } from "./vetoes.js";

/** Exact citation for every reference (no bare "AI4AI" — naming discipline). */
export const AI4AI_BENCH_CITATION =
  "AI4AI-Bench (arXiv:2608.20318, github.com/Einsia/AI4AI-Bench)" as const;

/** The benchmark's score for the shipped algorithm — the floor, not zero. */
export const AI4AI_SHIPPED_ALGORITHM_SCORE = 0.1;

const HEX64 = /^[0-9a-f]{64}$/;
const SLUG = /^[a-z0-9][a-z0-9._-]{0,127}$/;

// ---------------------------------------------------------------------------
// FROZEN TASK IDENTITY — mirrors benchmark.ts's dataset-identity discipline.
// ---------------------------------------------------------------------------

/**
 * A pinned AI4AI-Bench (arXiv:2608.20318) task. `repoSnapshotSha256` is the
 * mandatory lowercase SHA-256 over the frozen repository snapshot the task
 * ships; `evaluatorSha256` pins the evaluator that scores submissions. A
 * manifest missing either hash is refused outright — the same posture
 * normalizeSuiteItem takes toward dataset identity: no hash, no run.
 */
export interface Ai4aiTaskManifest {
  /** Benchmark repo slug, e.g. "einsia-ai4ai-bench". */
  readonly benchmark: string;
  /** Task id inside the benchmark, e.g. "task-03-policy-gradient". */
  readonly taskId: string;
  /** The learning-algorithm family the frozen repo implements. */
  readonly algorithmFamily: string;
  /** Lowercase SHA-256 over the frozen repository snapshot. */
  readonly repoSnapshotSha256: string;
  /** Lowercase SHA-256 over the evaluator that scores submissions. */
  readonly evaluatorSha256: string;
  readonly citation: typeof AI4AI_BENCH_CITATION;
}

/** Refuse anything that is not a fully hash-pinned task manifest. */
export function assertAi4aiTaskManifest(value: unknown): asserts value is Ai4aiTaskManifest {
  if (!value || typeof value !== "object") throw new Error("ai4ai task manifest must be an object");
  const manifest = value as Partial<Ai4aiTaskManifest>;
  if (!manifest.benchmark || !SLUG.test(manifest.benchmark)) {
    throw new Error("ai4ai task manifest requires a benchmark slug");
  }
  if (!manifest.taskId || !SLUG.test(manifest.taskId)) {
    throw new Error("ai4ai task manifest requires a task id slug");
  }
  if (!manifest.algorithmFamily || typeof manifest.algorithmFamily !== "string") {
    throw new Error("ai4ai task manifest requires an algorithm family");
  }
  if (!manifest.repoSnapshotSha256 || !HEX64.test(manifest.repoSnapshotSha256)) {
    throw new Error("ai4ai task manifest requires a lowercase SHA-256 repo snapshot identity");
  }
  if (!manifest.evaluatorSha256 || !HEX64.test(manifest.evaluatorSha256)) {
    throw new Error("ai4ai task manifest requires a lowercase SHA-256 evaluator identity");
  }
  if (manifest.citation !== AI4AI_BENCH_CITATION) {
    throw new Error(`ai4ai task manifest must carry the exact citation "${AI4AI_BENCH_CITATION}"`);
  }
}

/**
 * Canonical encoding for hashing. Keys sort by CODE UNIT, never by
 * `localeCompare`: locale collation is locale- and ICU-build-dependent, so
 * `{ z, ä, a }` orders as a,z,ä under sv_SE but a,ä,z under en_US — the same
 * payload would digest differently on two machines with different LANG. Code
 * -unit order is what RFC 8785 (JSON Canonicalization Scheme) specifies, and
 * it makes cross-process digests reproducible by construction rather than by
 * accident of which key sets happen to be fixed and ASCII.
 */
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

/** The frozen task identity: SHA-256 over the canonical manifest encoding. */
export function ai4aiTaskIdentity(manifest: Ai4aiTaskManifest): string {
  assertAi4aiTaskManifest(manifest);
  return createHash("sha256")
    .update("ruvector.ai4ai.task-identity.v1\0")
    .update(canonical(manifest))
    .digest("hex");
}

// ---------------------------------------------------------------------------
// MUTATIONS + LINEAGE — reproducible provenance for every candidate.
// ---------------------------------------------------------------------------

/**
 * One candidate learning-algorithm mutation submitted against a task. The
 * mutation itself (the diff) lives with the executor; the adapter records its
 * digest so lineage is reproducible without the harness ever holding the
 * candidate code.
 */
export interface Ai4aiMutation {
  /** Stable id, e.g. "mut-007". */
  readonly id: string;
  /** Lowercase SHA-256 over the candidate diff against the parent state. */
  readonly diffSha256: string;
  /** The run seed the evaluator is asked to use. */
  readonly seed: number;
  /** One-line description of the algorithmic change (self-reported). */
  readonly description: string;
}

export function assertAi4aiMutation(value: unknown): asserts value is Ai4aiMutation {
  if (!value || typeof value !== "object") throw new Error("ai4ai mutation must be an object");
  const mutation = value as Partial<Ai4aiMutation>;
  if (!mutation.id || !SLUG.test(mutation.id)) throw new Error("ai4ai mutation requires an id slug");
  if (!mutation.diffSha256 || !HEX64.test(mutation.diffSha256)) {
    throw new Error("ai4ai mutation requires a lowercase SHA-256 diff digest");
  }
  if (!Number.isSafeInteger(mutation.seed) || (mutation.seed ?? -1) < 0) {
    throw new Error("ai4ai mutation seed must be a non-negative integer");
  }
  if (!mutation.description) throw new Error("ai4ai mutation requires a description");
}

/**
 * How the evaluator classified a submission. "unreported" means the evaluator
 * output carried no classification — the adapter NEVER fabricates one, per
 * the paper's own headline finding that only genuine algorithm changes count.
 */
export type Ai4aiSubmissionKind = "algorithm-change" | "tuning-only" | "unreported";

/** Parsed, validated evaluator output for one mutation. */
export interface Ai4aiEvaluation {
  /** Finite score in [0, 1]; 0.1 is the shipped algorithm. */
  readonly score: number;
  readonly submissionKind: Ai4aiSubmissionKind;
}

/**
 * Parse raw evaluator output. FAIL-LOUD: a missing, non-numeric, non-finite,
 * or out-of-range score THROWS — it never becomes an observation (Wave-3
 * lesson: NaN silently defeats every comparison-based gate downstream).
 */
export function parseAi4aiEvaluatorOutput(value: unknown): Ai4aiEvaluation {
  if (!value || typeof value !== "object") throw new Error("ai4ai evaluator output must be an object");
  const output = value as { score?: unknown; algorithm_changed?: unknown };
  if (typeof output.score !== "number" || !Number.isFinite(output.score)) {
    throw new Error("ai4ai evaluator score must be a finite number (fail-loud: refusing non-finite)");
  }
  if (output.score < 0 || output.score > 1) {
    throw new Error(`ai4ai evaluator score out of range [0, 1]: ${output.score}`);
  }
  let submissionKind: Ai4aiSubmissionKind = "unreported";
  if (output.algorithm_changed === true) submissionKind = "algorithm-change";
  else if (output.algorithm_changed === false) submissionKind = "tuning-only";
  else if (output.algorithm_changed !== undefined) {
    throw new Error("ai4ai evaluator algorithm_changed must be boolean when present");
  }
  return Object.freeze({ score: output.score, submissionKind });
}

/**
 * One link in the lineage chain. `parentDigest` is the previous record's
 * digest (the ROOT BINDING for the first record — see `ai4aiRunRoot`, which
 * mixes a per-run nonce into the task identity so a chain is bound to the run
 * that produced it and cannot be replayed as a fresh one). `digest` commits to
 * everything — task, parent, diff, seed, executor identity, whether the
 * executor was a declared wrapper, and the evaluation — so the chain is
 * tamper-evident and any run can be re-derived and audited.
 */
export interface Ai4aiLineageRecord {
  readonly taskIdentity: string;
  readonly parentDigest: string;
  readonly mutation: Ai4aiMutation;
  readonly executorIdentity: string;
  /**
   * HOW `executorIdentity` was established (see `Ai4aiExecutorClass`). Inside
   * the digest body, exactly like `wrapperIndirection`, so it cannot be
   * flipped after the fact.
   */
  readonly executorClass: Ai4aiExecutorClass;
  /**
   * True only when the caller EXPLICITLY declared that `executorIdentity`
   * hashes a wrapper rather than the evaluator itself.
   */
  readonly wrapperIndirection: boolean;
  readonly evaluation: Ai4aiEvaluation;
  readonly digest: string;
}

/**
 * The root a lineage chains from: the frozen task identity mixed with a
 * per-run nonce.
 *
 * WHAT THE NONCE DELIVERS — and only this: two runs over identical inputs
 * cannot produce byte-identical records, so runs are DISTINGUISHABLE. It is a
 * run distinguisher, NOT anti-replay and NOT anti-forgery. There is no
 * signature over the chain and no external record of issued nonces, so
 * re-presenting a whole `{ runNonce, records }` pair still verifies, and
 * `runAi4aiLineage` accepts a caller-supplied nonce whose FORMAT is validated
 * but whose freshness is not. Chain forgery and replay detection are out of
 * scope for this slice; they need a signing authority and an issued-nonce
 * ledger, neither of which exists here.
 */
export function ai4aiRunRoot(manifest: Ai4aiTaskManifest, runNonce: string): string {
  if (!HEX64.test(runNonce)) throw new Error("ai4ai run nonce must be 64 lowercase hex chars");
  return createHash("sha256")
    .update("ruvector.ai4ai.run-root.v1\0")
    .update(`${ai4aiTaskIdentity(manifest)}\0${runNonce}`)
    .digest("hex");
}

/** Fresh per-run nonce. */
export function newAi4aiRunNonce(): string {
  return randomBytes(32).toString("hex");
}

/** A complete run: the nonce its chain is bound to, plus the records. */
export interface Ai4aiLineage {
  readonly runNonce: string;
  readonly records: readonly Ai4aiLineageRecord[];
}

function lineageDigest(record: Omit<Ai4aiLineageRecord, "digest">): string {
  return createHash("sha256")
    .update("ruvector.ai4ai.lineage.v1\0")
    .update(canonical(record))
    .digest("hex");
}

// ---------------------------------------------------------------------------
// EXECUTOR SEAM — injected, never run in-repo (no GPU, no Docker here).
//
// TRUST IS ESTABLISHED, NEVER ASSUMED. A lineage defaults to "NOT
// evaluator-bound evidence"; only this module's own construction can upgrade
// it. `executorIdentity` alone is a self-reported string, and
// `manifest.evaluatorSha256` is PUBLIC — it sits in the manifest every caller
// already holds — so comparing the two proves nothing about what ran: an
// executor that runs nothing, echoes the pinned hash, and returns a perfect
// score would otherwise mint a clean-verifying lineage. Byte-derived identity
// is therefore attested through a module-private registry (below) that only
// `commandAi4aiExecutor` can write to, and the attestation is keyed on the
// exact result object the adapter itself built, so a self-reported identity
// from an arbitrary injected executor CANNOT be presented as byte-derived.
// An injected executor may still run — it simply cannot claim byte-derived
// identity, and its records say so.
// ---------------------------------------------------------------------------

/**
 * How a record's `executorIdentity` was established:
 *   - `byte-derived-command` — this module hashed the entrypoint's real bytes
 *     and the hash equals the manifest's pinned evaluator. The only class that
 *     is evidence OF THE EVALUATOR.
 *   - `wrapper-waived` — the caller explicitly declared the entrypoint wraps
 *     the evaluator, waiving the binding. Honest, but the identity names the
 *     wrapper, not the pinned evaluator.
 *   - `injected` — a caller-supplied executor self-reported an identity this
 *     module cannot corroborate. Runnable, never evaluator-bound.
 */
export type Ai4aiExecutorClass = "byte-derived-command" | "wrapper-waived" | "injected";

/**
 * What this module attests about a result it built itself: the identity it
 * derived from the entrypoint's real bytes, AND a digest over the evaluator
 * output as it stood when execution finished.
 *
 * The CONTENT digest is the load-bearing half. Attesting only the object
 * answers "did this come from the command executor?" but not "is this score
 * what the evaluator produced" — and `Object.freeze` is SHALLOW, so the
 * parsed `raw` payload the score is read from stays mutable even when the
 * wrapper object does not. Recomputing this digest at read time closes that
 * gap by construction, and keeps closing it across refactors that rebuild the
 * result object, which an enumeration of specific mutation paths would not.
 */
interface Ai4aiAttestation {
  readonly identity: string;
  readonly contentDigest: string;
}

/**
 * Module-private attestation registry: maps the exact result object this
 * module constructed to what it attests about that result. Not exported and
 * not reachable from outside — an arbitrary executor cannot add an entry, and
 * mutating a returned object cannot change the attested values because the
 * adapter reads them from here, never off the object.
 *
 * MODULE-INSTANCE SCOPE (see `MODULE_INSTANCE_MARKER`): the registry belongs
 * to ONE loaded copy of this module. A second instance — a bundler duplicate,
 * a symlinked path, dist-vs-src — has its own registry, so a result produced
 * by instance A is unknown to instance B. This is not a laundering path:
 * cross-instance records degrade to `injected`, never upgrade. It does mean an
 * honest byte-derived run can silently become `injected` and a strict gate
 * then rejects GOOD evidence, so the downgrade is made legible rather than
 * mysterious.
 */
const ATTESTATIONS = new WeakMap<object, Ai4aiAttestation>();

/**
 * Cross-realm-visible marker stamped on every result this module builds.
 * Its only purpose is diagnostic: if a result carries the marker but is
 * absent from OUR registry, the result came from a different loaded instance
 * of this module, and the resulting downgrade to `injected` is explained
 * instead of appearing arbitrary.
 */
const MODULE_INSTANCE_MARKER = Symbol.for("ruvector.ai4ai.module-instance");
const MODULE_INSTANCE_ID = randomBytes(8).toString("hex");

/**
 * Stable digest over evaluator output, using the module's canonical encoder.
 * Exported so the content guarantee is DIRECTLY testable: on every
 * constructible path the deep-freeze below stops a payload edit first, so the
 * comparison in `assertAi4aiContentUnchanged` would otherwise never execute in
 * any test — leaving a guard that could be inverted (`!==` to `===`) with the
 * whole suite still green. A pure hash with no state and no authority is a
 * cheap public surface to pay for keeping that guard exercised.
 */
export function ai4aiContentDigest(raw: unknown): string {
  return createHash("sha256")
    .update("ruvector.ai4ai.evaluator-output.v1\0")
    .update(canonical(raw))
    .digest("hex");
}

/**
 * Refuse evaluator output that no longer matches what was attested at
 * execution time. Pure and exported for the same dormant-guard reason as
 * `ai4aiContentDigest`: this is where a content mismatch is DETECTED, and a
 * detection path that no test can reach is a detection path that can rot.
 */
export function assertAi4aiContentUnchanged(
  raw: unknown,
  attestedDigest: string,
  mutationId: string,
): void {
  if (ai4aiContentDigest(raw) !== attestedDigest) {
    throw new Error(
      `ai4ai evaluator output was modified after execution for mutation ` +
      `${mutationId} (content digest mismatch); a mutated result is not a score`,
    );
  }
}

/**
 * Recursively freeze the evaluator output (defense-in-depth, not the fix).
 * Recursion is NOT gated on the parent's frozen state — an already-frozen
 * parent can still hold unfrozen children, which the guard would have skipped.
 * Unreachable for JSON.parse output, wrong if this helper is ever reused. A
 * `seen` set keeps cycles from recursing forever now that frozen nodes no
 * longer terminate the walk.
 */
function deepFreeze<T>(value: T, seen: WeakSet<object> = new WeakSet()): T {
  if (value && typeof value === "object") {
    if (seen.has(value)) return value;
    seen.add(value);
    if (!Object.isFrozen(value)) Object.freeze(value);
    for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child, seen);
  }
  return value;
}

/**
 * The injected executor: given a pinned task and one mutation, run the real
 * AI4AI-Bench (arXiv:2608.20318) evaluator somewhere it can actually run and
 * return its RAW output plus a stable identity for the executor itself (so
 * lineage records name what produced each score). A crash must propagate —
 * the adapter re-throws with context and never scores a crash as a result.
 *
 * A self-reported `executorIdentity` is recorded but classed `injected`: this
 * module cannot corroborate it, so it is never treated as evidence about the
 * evaluator.
 */
export type Ai4aiExecutor = (
  manifest: Ai4aiTaskManifest,
  mutation: Ai4aiMutation,
) => Promise<{
  readonly raw: unknown;
  readonly executorIdentity: string;
  /**
   * Set true ONLY when `executorIdentity` deliberately hashes a wrapper rather
   * than the pinned evaluator itself. Absent/false means the identity IS
   * claimed to be the evaluator's, and `runAi4aiMutation` then ENFORCES it
   * against `manifest.evaluatorSha256`.
   */
  readonly wrapperIndirection?: boolean;
}>;

export interface Ai4aiCommandExecutorOptions {
  /** Absolute path to the evaluator entrypoint to spawn. */
  readonly executorPath: string;
  /** Trusted argv inserted before the entrypoint (primarily test wrappers). */
  readonly commandPrefixArgs?: readonly string[];
  readonly timeoutMs?: number;
  readonly maxOutputBytes?: number;
  /**
   * Declare that `executorPath` is a WRAPPER around the pinned evaluator, not
   * the evaluator itself. Opting out of the evaluator-identity binding must be
   * explicit and visible: it is recorded on every lineage record
   * (`wrapperIndirection: true`) and committed to by the record digest, so an
   * auditor sees exactly which runs waived the check. Default false.
   */
  readonly wrapperIndirection?: boolean;
}

/**
 * A command-spawning executor with benchmark.ts's isolation discipline: a
 * scrubbed environment (PATH only), a wall timeout, an output ceiling, and a
 * temp workspace removed afterward. The entrypoint receives
 * `--task <manifest.json> --mutation <mutation.json> --output <result.json>`
 * and must write its evaluation JSON to the output path. Executor identity is
 * the SHA-256 of the entrypoint file's bytes, so lineage pins WHAT ran — and
 * the entrypoint is RE-HASHED after exit, failing loudly if its bytes changed
 * between hashing and execution (time-of-check/time-of-use), so the recorded
 * identity always names the code that actually ran.
 */
export function commandAi4aiExecutor(options: Ai4aiCommandExecutorOptions): Ai4aiExecutor {
  return async (manifest, mutation) => {
    const executorPath = resolve(options.executorPath);
    const executorIdentity = createHash("sha256").update(await readFile(executorPath)).digest("hex");
    const directory = await mkdtemp(join(tmpdir(), "ruvector-ai4ai-"));
    try {
      const taskFile = join(directory, "task.json");
      const mutationFile = join(directory, "mutation.json");
      const outputFile = join(directory, "result.json");
      await writeFile(taskFile, `${JSON.stringify(manifest)}\n`, { mode: 0o600 });
      await writeFile(mutationFile, `${JSON.stringify(mutation)}\n`, { mode: 0o600 });
      const prefix = options.commandPrefixArgs ?? [];
      const [command, ...leading] = prefix.length > 0 ? prefix : [executorPath];
      const args = [
        ...leading,
        ...(prefix.length > 0 ? [executorPath] : []),
        "--task", taskFile,
        "--mutation", mutationFile,
        "--output", outputFile,
      ];
      await new Promise<void>((resolveRun, reject) => {
        const child = spawn(command!, args, {
          stdio: ["ignore", "pipe", "pipe"],
          env: { PATH: process.env.PATH ?? "" },
        });
        let stderr = "";
        let outputBytes = 0;
        let limitFailure = "";
        const consume = (chunk: Buffer, capture: boolean) => {
          outputBytes += chunk.length;
          if (capture) stderr = `${stderr}${chunk.toString("utf8")}`.slice(-2_000);
          if (outputBytes > (options.maxOutputBytes ?? 10 * 1024 * 1024)) {
            limitFailure = "ai4ai executor output limit exceeded";
            child.kill("SIGKILL");
          }
        };
        child.stdout.on("data", (chunk: Buffer) => consume(chunk, false));
        child.stderr.on("data", (chunk: Buffer) => consume(chunk, true));
        const timer = setTimeout(() => {
          child.kill("SIGKILL");
          reject(new Error(`ai4ai executor exceeded ${options.timeoutMs ?? 300_000}ms`));
        }, options.timeoutMs ?? 300_000);
        child.once("error", (error) => {
          clearTimeout(timer);
          reject(error);
        });
        child.once("exit", (code) => {
          clearTimeout(timer);
          code === 0 && !limitFailure
            ? resolveRun()
            : reject(new Error(limitFailure || `ai4ai executor exited ${code}: ${stderr}`));
        });
      });
      // TOCTOU: the entrypoint could have been swapped between hashing and
      // spawning. Re-hash and refuse rather than record an identity for code
      // that did not run.
      const afterIdentity = createHash("sha256").update(await readFile(executorPath)).digest("hex");
      if (afterIdentity !== executorIdentity) {
        throw new Error(
          `ai4ai executor bytes changed during the run (${executorIdentity} -> ${afterIdentity})`,
        );
      }
      // The stdout/stderr ceiling does not cover the result file the evaluator
      // writes. Bound it on the SAME open handle we then read from: a
      // stat-then-read pair lets the child (same user, and both calls follow
      // symlinks) swap the file in between and defeat the cap.
      const maxOutputBytes = options.maxOutputBytes ?? 10 * 1024 * 1024;
      const handle = await open(outputFile, "r");
      let raw: unknown;
      try {
        const handleStat = await handle.stat();
        if (handleStat.size > maxOutputBytes) {
          throw new Error(
            `ai4ai executor result exceeds output limit: ${handleStat.size} > ${maxOutputBytes} bytes`,
          );
        }
        raw = JSON.parse(await handle.readFile("utf8")) as unknown;
      } finally {
        await handle.close();
      }
      // Attest the CONTENT as well as the object, and freeze the payload.
      // On every reachable path the deep-freeze is what BLOCKS a
      // post-execution edit of the scored payload (plain Object.freeze is
      // shallow and would leave `raw` writable); the content digest, taken
      // here over the output exactly as the evaluator produced it and
      // recomputed at read, is what CATCHES such an edit if a refactor ever
      // rebuilds the result or drops the freeze.
      deepFreeze(raw);
      const outcome = {
        raw,
        executorIdentity,
        wrapperIndirection: options.wrapperIndirection ?? false,
        [MODULE_INSTANCE_MARKER]: MODULE_INSTANCE_ID,
      };
      ATTESTATIONS.set(outcome, {
        identity: executorIdentity,
        contentDigest: ai4aiContentDigest(raw),
      });
      return Object.freeze(outcome);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  };
}

// ---------------------------------------------------------------------------
// RUNNING — one mutation, or a lineage chain of them.
// ---------------------------------------------------------------------------

/**
 * Evaluate one mutation against a pinned task. Validates the manifest and
 * mutation, invokes the injected executor, and parses its output fail-loud.
 * An executor crash propagates as an error naming the mutation — it is NEVER
 * folded into a score (adapter-crash ≠ result, same discipline as the WP16
 * subprocess binder).
 *
 * EVALUATOR-IDENTITY BINDING, established not assumed: a record is
 * `byte-derived-command` ONLY when this module itself hashed the entrypoint's
 * bytes (attested via the module-private registry) AND that hash equals
 * `manifest.evaluatorSha256` — a mismatch there throws, because the module
 * knows what actually ran. A caller-supplied executor's self-reported identity
 * is recorded and classed `injected`; it is never checked against the public
 * pin, because passing that check proves only that the caller can read the
 * manifest. An explicitly declared wrapper is classed `wrapper-waived`.
 */
export async function runAi4aiMutation(
  manifest: Ai4aiTaskManifest,
  mutation: Ai4aiMutation,
  executor: Ai4aiExecutor,
  parentDigest: string,
): Promise<Ai4aiLineageRecord> {
  assertAi4aiTaskManifest(manifest);
  assertAi4aiMutation(mutation);
  if (!HEX64.test(parentDigest)) {
    throw new Error("ai4ai parent digest must be 64 lowercase hex chars");
  }
  let outcome: Awaited<ReturnType<Ai4aiExecutor>>;
  try {
    outcome = await executor(manifest, mutation);
  } catch (error) {
    throw new Error(
      `ai4ai executor failed for mutation ${mutation.id} (crash is not a score): ${
        error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (!HEX64.test(outcome.executorIdentity)) {
    throw new Error("ai4ai executor must report a lowercase SHA-256 identity");
  }
  const wrapperIndirection = outcome.wrapperIndirection ?? false;
  // The ONLY source of a byte-derived identity is this module's own registry.
  // A self-reported string can never reach this variable.
  const attestation = typeof outcome === "object" && outcome !== null
    ? ATTESTATIONS.get(outcome)
    : undefined;
  if (attestation === undefined && typeof outcome === "object" && outcome !== null &&
      (outcome as Record<symbol, unknown>)[MODULE_INSTANCE_MARKER] !== undefined) {
    // Marked but not registry-identical. The record degrades to `injected`
    // either way (never upgrades); this line only explains WHY, so it must not
    // send someone hunting the wrong problem — and must not let the very
    // component being downgraded write arbitrary `ai4ai:`-prefixed log lines.
    // The marker value is attacker-controlled, so it is JSON-encoded (newlines
    // escaped) and length-capped before it reaches the log.
    const theirs = JSON.stringify(
      String((outcome as Record<symbol, unknown>)[MODULE_INSTANCE_MARKER]).slice(0, 64),
    );
    process.stderr.write(
      theirs === JSON.stringify(MODULE_INSTANCE_ID)
        ? `ai4ai: result carries this instance's marker (${theirs}) but is not the ` +
          `object this module registered — wrapped or copied, not a different ` +
          `instance; classing as injected\n`
        : `ai4ai: result produced by a different loaded instance of ai4aiBench ` +
          `(theirs=${theirs}, ours=${JSON.stringify(MODULE_INSTANCE_ID)}); classing ` +
          `as injected — load one instance to keep byte-derived evidence\n`,
    );
  }
  // Content attestation: the scored payload must be what the evaluator
  // produced. The deep-freeze blocks an edit on every reachable path; this
  // catches one if a refactor ever rebuilds the result or drops the freeze.
  if (attestation !== undefined) {
    assertAi4aiContentUnchanged(outcome.raw, attestation.contentDigest, mutation.id);
  }
  const attestedIdentity = attestation?.identity;
  let executorClass: Ai4aiExecutorClass;
  let executorIdentity: string;
  if (attestedIdentity !== undefined && !wrapperIndirection) {
    // This module hashed real bytes: the pin is now a meaningful binding.
    if (attestedIdentity !== manifest.evaluatorSha256) {
      throw new Error(
        `ai4ai executor identity does not match the manifest's pinned evaluator ` +
        `(${attestedIdentity} != ${manifest.evaluatorSha256}); declare ` +
        `wrapperIndirection explicitly if the entrypoint wraps the evaluator`,
      );
    }
    executorClass = "byte-derived-command";
    executorIdentity = attestedIdentity;
  } else if (wrapperIndirection) {
    executorClass = "wrapper-waived";
    executorIdentity = attestedIdentity ?? outcome.executorIdentity;
  } else {
    // Self-reported and uncorroborated. Recorded honestly, never bound.
    executorClass = "injected";
    executorIdentity = outcome.executorIdentity;
  }
  const evaluation = parseAi4aiEvaluatorOutput(outcome.raw);
  const body = {
    taskIdentity: ai4aiTaskIdentity(manifest),
    parentDigest,
    mutation,
    executorIdentity,
    executorClass,
    wrapperIndirection,
    evaluation,
  };
  return Object.freeze({ ...body, digest: lineageDigest(body) });
}

/**
 * Evaluate a sequence of mutations as one lineage chain: the first mutation's
 * parent is the frozen task identity, each subsequent mutation's parent is
 * the previous record's digest. The whole chain is tamper-evident; a single
 * failing evaluation aborts the chain loudly rather than recording a hole.
 */
export async function runAi4aiLineage(
  manifest: Ai4aiTaskManifest,
  mutations: readonly Ai4aiMutation[],
  executor: Ai4aiExecutor,
  runNonce: string = newAi4aiRunNonce(),
): Promise<Ai4aiLineage> {
  assertAi4aiTaskManifest(manifest);
  if (mutations.length === 0) throw new Error("ai4ai lineage requires at least one mutation");
  const records: Ai4aiLineageRecord[] = [];
  let parentDigest = ai4aiRunRoot(manifest, runNonce);
  for (const mutation of mutations) {
    const record = await runAi4aiMutation(manifest, mutation, executor, parentDigest);
    records.push(record);
    parentDigest = record.digest;
  }
  return Object.freeze({ runNonce, records: Object.freeze(records) });
}

/**
 * What a caller demands of a lineage's evidence class. Structural integrity
 * (digest, chain, task, nonce) is ALWAYS checked; this governs only the
 * evidence-strength question.
 *
 * SCOPE BOUNDARY — read this before building a gate on `requireEvaluatorBound`.
 * What this slice provides is WITHIN-PROCESS INTEGRITY: a record cannot
 * misreport what the adapter's own executor produced. It is NOT cross-party
 * authentication — an adversary who controls the process can still fabricate a
 * chain, because nothing here is signed. `lineageDigest` is an UNKEYED hash, so
 * a wholly hand-constructed chain can claim `byte-derived-command` and verify
 * clean; the attestation registry constrains what THIS module's executor path
 * will report, not what a hostile process can hand you.
 *
 * So `requireEvaluatorBound` defends against honest-but-weak evidence — an
 * injected stub, a wrapper entrypoint, a stale run — and NOT against an
 * adversary. Closing that gap requires signing the chain and an issued-nonce
 * ledger, both scoped as follow-up work, neither present here. A gate that
 * treats strict mode as proof against a malicious producer is over-trusting it.
 */
export interface Ai4aiEvidencePolicy {
  /**
   * Demand that EVERY record be `byte-derived-command`. Off by default: real
   * Docker/GPU runs legitimately use a wrapper entrypoint, and hard-blocking
   * those would break them. A caller that needs evaluator-bound evidence turns
   * this on and receives `ai4ai_evaluator_identity_waived` (wrapper) or
   * `ai4ai_executor_not_byte_bound` (injected) — distinct reasons, so an
   * automated gate can tell a bound run from a waived or uncorroborated one
   * instead of seeing an indistinguishable clean verdict.
   */
  readonly requireEvaluatorBound?: boolean;
}

/**
 * The weakest executor class present — the class of the lineage as a whole.
 * An EMPTY lineage is `injected`, not the strongest class: zero records is
 * zero evidence, and falling through to `byte-derived-command` would invert
 * this function's whole contract on a public surface cli.ts prints.
 */
export function ai4aiEvidenceClass(lineage: Ai4aiLineage): Ai4aiExecutorClass {
  if (lineage.records.length === 0) return "injected";
  if (lineage.records.some((record) => record.executorClass === "injected")) return "injected";
  if (lineage.records.some((record) => record.executorClass === "wrapper-waived")) {
    return "wrapper-waived";
  }
  return "byte-derived-command";
}

/**
 * Re-verify a lineage: every record must re-derive its own digest, name the
 * same frozen task, and chain to its predecessor from the run root (task
 * identity + this run's nonce). Under `policy.requireEvaluatorBound`, records
 * that are not byte-derived also draw a distinct reason. Returns veto-style
 * reasons (empty ⇒ intact). This is what the veto provider below consumes.
 */
export function verifyAi4aiLineage(
  manifest: Ai4aiTaskManifest,
  lineage: Ai4aiLineage,
  policy: Ai4aiEvidencePolicy = {},
): string[] {
  const reasons: string[] = [];
  const { runNonce, records } = lineage;
  if (records.length === 0) {
    reasons.push("ai4ai_lineage_empty");
    return reasons;
  }
  if (!HEX64.test(runNonce)) {
    reasons.push("ai4ai_lineage_run_nonce_invalid");
    return [...new Set(reasons)].sort();
  }
  const taskIdentity = ai4aiTaskIdentity(manifest);
  let expectedParent = ai4aiRunRoot(manifest, runNonce);
  for (const record of records) {
    if (record.taskIdentity !== taskIdentity) reasons.push("ai4ai_lineage_task_mismatch");
    if (record.parentDigest !== expectedParent) reasons.push("ai4ai_lineage_broken_chain");
    const { digest, ...body } = record;
    if (lineageDigest(body) !== digest) reasons.push("ai4ai_lineage_digest_mismatch");
    // A byte-derived record is the only one whose identity this module
    // established; for it, disagreement with the pin is a real mismatch.
    if (record.executorClass === "byte-derived-command" &&
        record.executorIdentity !== manifest.evaluatorSha256) {
      reasons.push("ai4ai_evaluator_identity_mismatch");
    }
    if (policy.requireEvaluatorBound) {
      if (record.executorClass === "wrapper-waived") {
        reasons.push("ai4ai_evaluator_identity_waived");
      } else if (record.executorClass === "injected") {
        reasons.push("ai4ai_executor_not_byte_bound");
      }
    }
    expectedParent = record.digest;
  }
  return [...new Set(reasons)].sort();
}

// ---------------------------------------------------------------------------
// PROMOTION — our own paired statistics, never the paper's number.
// ---------------------------------------------------------------------------

/**
 * Compare baseline vs candidate score samples with the EXISTING paired-
 * bootstrap authority. Finiteness is re-checked at this choke point even
 * though ingest already enforced it (defense-in-depth: a hypothetical bypass
 * still cannot reach the gate with a NaN). The paper's 0.250 best-system
 * score is a TARGET for reporting, never an acceptance bar — only this
 * decision, recomputed on our own paired runs, counts toward promotion.
 */
export function ai4aiPairedDecision(
  baseline: readonly number[],
  candidate: readonly number[],
  minimumEffect = 0.005,
): PairedDecision {
  for (const value of [...baseline, ...candidate]) {
    if (!Number.isFinite(value)) {
      throw new Error("ai4ai paired decision refuses non-finite samples (fail-loud)");
    }
  }
  return pairedBootstrapDecision(baseline, candidate, minimumEffect);
}

/**
 * A vetoes.ts-composable provider that REJECTS promotion when the lineage
 * behind a candidate does not verify. Conjunctive like every provider: it can
 * only object, never rescue. `lineageFor` maps the veto context to the
 * manifest + lineage under evaluation, mirroring the `candidateFor` /
 * `environmentFor` seams of the dream-machine and ADR-324 providers. Pass
 * `{ requireEvaluatorBound: true }` to also reject lineages that are not
 * byte-derived — off by default so legitimate wrapper-entrypoint runs are not
 * blocked, on when a gate needs evaluator-bound evidence specifically.
 */
export function ai4aiLineageVetoProvider(
  lineageFor: (context: unknown) => {
    manifest: Ai4aiTaskManifest;
    lineage: Ai4aiLineage;
  } | Promise<{ manifest: Ai4aiTaskManifest; lineage: Ai4aiLineage }>,
  policy: Ai4aiEvidencePolicy = {},
): PromotionVetoProvider {
  return async (context) => {
    const { manifest, lineage } = await lineageFor(context);
    return verifyAi4aiLineage(manifest, lineage, policy);
  };
}
