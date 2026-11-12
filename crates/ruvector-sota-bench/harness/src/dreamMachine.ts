/**
 * Dream Machine adapter (PIR WP2, ADR-306, issue #838).
 *
 * Adapts the `dream-machine` npm engine (ruvnet/dream-machine, ADR-0001) as an
 * EVALUATION stage feeding this harness's EXISTING promotion path:
 *
 *   dream-machine verdict ──► vetoes.ts PromotionVetoProvider input
 *                        ──► flywheel.ts ruvectorPromotionRule (via vetoReasons)
 *
 * Authority split (the integration contract, no duplicated authority):
 *   - statistics.ts owns the paired-bootstrap significance decision;
 *   - the dream-machine engine owns the verdict vocabulary
 *     (ACCEPT | REJECT | INCONCLUSIVE), the ledger record, and the witness
 *     stamp binding a report to a commit;
 *   - vetoes.ts / flywheel.ts own the promotion RECOMMENDATION;
 *   - a human owns promotion itself.
 *
 * CONSTITUTIONAL BOUNDARY (ADR-0001, adopted verbatim by ADR-306):
 * "Evaluation is not promotion — the machine never merges; a human does."
 * This module can only emit evaluation verdicts and veto reasons. It exports
 * no promote/merge/approve function, and the verdict object is frozen data
 * with no callable members. test/dreamMachine.test.ts asserts both.
 */
import { run, VERSION as DREAM_MACHINE_VERSION, type IO, type RunResult } from "dream-machine";
import type { PairedDecision } from "./statistics.js";
import type { PromotionVetoProvider } from "./vetoes.js";

export { DREAM_MACHINE_VERSION };

/** The dream-machine ledger verdict vocabulary (engine-owned). */
export type DreamVerdict = "ACCEPT" | "REJECT" | "INCONCLUSIVE";

/** Whether the candidate's evaluator actually ran (engine-owned vocabulary). */
export type DreamEvaluated = "yes" | "no" | "blocked";

/** A candidate submitted to the dream-machine evaluation stage. */
export interface DreamCandidate {
  /** One-line description of the finding / mutation under evaluation. */
  finding: string;
  /** Full evaluation report (markdown/JSON) to be witness-stamped. */
  report: string;
  /** Git commit sha (7-40 hex) the evaluation ran against. */
  commit: string;
  /** Paired-bootstrap decision from statistics.ts — the statistical authority. */
  decision: PairedDecision;
  /** Did the evaluator run? Defaults to "yes". */
  evaluated?: DreamEvaluated;
  /** Evaluation date, YYYY-MM-DD. Defaults to today (UTC). */
  date?: string;
}

/**
 * The evaluation verdict. Deliberately pure, frozen data: there is no
 * promote(), no merge(), and no side-effect handle here — a verdict is an
 * INPUT to the promotion rule and to ruvector-proof-gate, never a bypass.
 */
export interface DreamMachineVerdict {
  readonly verdict: DreamVerdict;
  readonly evaluated: DreamEvaluated;
  /** Why the verdict is what it is (human-auditable). */
  readonly reasons: readonly string[];
  /** sha256(sha256(report) + commit) — the engine's witness stamp. */
  readonly witness: string;
  /** The exact LEDGER.md markdown produced by the engine for this evaluation. */
  readonly ledger: string;
  /** Engine version that produced this verdict. */
  readonly engineVersion: string;
}

/** In-memory IO surface for the engine — no real filesystem is touched. */
function memoryIO(seed: Record<string, string>, date: string): IO & { files: Map<string, string> } {
  const files = new Map(Object.entries(seed));
  return {
    files,
    async readFile(path: string) {
      const content = files.get(path);
      if (content === undefined) throw new Error(`ENOENT: ${path}`);
      return content;
    },
    async writeFile(path: string, content: string) {
      files.set(path, content);
    },
    now: () => date,
    env: {},
  };
}

function expectOk(step: string, result: RunResult): void {
  if (result.code !== 0) {
    throw new Error(`dream-machine ${step} failed (exit ${result.code}): ${result.err.trim() || result.out.trim()}`);
  }
}

/**
 * Map the harness's statistical decision into the engine's verdict vocabulary.
 * Pure. Conservative by construction: only a statistically confirmed
 * improvement maps to ACCEPT; "inconclusive" is never rounded up.
 */
export function verdictFromDecision(
  decision: PairedDecision,
  evaluated: DreamEvaluated = "yes",
): DreamVerdict {
  if (evaluated !== "yes") return "INCONCLUSIVE";
  if (decision.outcome === "pass") return "ACCEPT";
  if (decision.outcome === "fail") return "REJECT";
  return "INCONCLUSIVE";
}

const LEDGER_PATH = "docs/dream-cycle/LEDGER.md";
const REPORT_PATH = "evaluation-report.md";

/**
 * Run one candidate through the dream-machine engine: record the evaluation
 * in an engine-verified ledger row and witness-stamp the report against the
 * commit. Returns a frozen verdict — data only, no promotion capability.
 */
export async function evaluateCandidate(candidate: DreamCandidate): Promise<DreamMachineVerdict> {
  const evaluated = candidate.evaluated ?? "yes";
  const date = candidate.date ?? new Date().toISOString().slice(0, 10);
  const verdict = verdictFromDecision(candidate.decision, evaluated);
  const reasons: string[] = [
    `statistics_outcome_${candidate.decision.outcome}`,
    `evaluated_${evaluated}`,
  ];

  const io = memoryIO({ [REPORT_PATH]: candidate.report }, date);

  // 1. Engine computes the witness stamp binding report to commit.
  const stamped = await run(["witness", "stamp", REPORT_PATH, candidate.commit], io);
  expectOk("witness stamp", stamped);
  const witnessMatch = stamped.out.match(/witness\s*:\s*([0-9a-f]{64})/);
  if (!witnessMatch) throw new Error(`could not parse witness from engine output: ${stamped.out}`);
  const witness = witnessMatch[1]!;

  // 2. Engine appends the evaluation row to the ledger.
  const appended = await run([
    "ledger", "append",
    "--path", LEDGER_PATH,
    "--date", date,
    "--deep", "sota-harness",
    "--finding", candidate.finding,
    "--evaluated", evaluated,
    "--verdict", verdict,
    "--effect", `meanDelta=${candidate.decision.meanDelta}`,
    "--witness", witness.slice(0, 8),
  ], io);
  expectOk("ledger append", appended);

  // 3. Engine structurally verifies its own ledger (verdict vocabulary check).
  const verified = await run(["ledger", "verify", "--path", LEDGER_PATH], io);
  expectOk("ledger verify", verified);

  // 4. Engine re-verifies the witness stamp round-trips.
  const witnessed = await run(["witness", "verify", REPORT_PATH, candidate.commit, witness], io);
  expectOk("witness verify", witnessed);

  return Object.freeze({
    verdict,
    evaluated,
    reasons: Object.freeze(reasons),
    witness,
    ledger: io.files.get(LEDGER_PATH) ?? "",
    engineVersion: DREAM_MACHINE_VERSION,
  });
}

/**
 * Map a verdict into vetoes.ts veto reasons. Pure. Conjunctive with all other
 * vetoes: a dream-machine ACCEPT adds no credit and rescues nothing; anything
 * other than ACCEPT blocks promotion ("no mutation reaches the proof gate
 * without passing a verdict").
 */
export function vetoesFromVerdict(verdict: DreamMachineVerdict): string[] {
  if (verdict.verdict === "ACCEPT") return [];
  if (verdict.verdict === "REJECT") return ["dream_machine_reject"];
  return ["dream_machine_inconclusive"];
}

/**
 * A PromotionVetoProvider for flywheel.ts: runs the candidate through the
 * dream-machine evaluation stage and feeds the verdict into the existing
 * promotion rule as veto reasons. This is the ONLY wiring between the engine
 * and promotion, and it can only ever say "no" or "no objection".
 */
export function dreamMachineVetoProvider(
  candidateFor: (context: Parameters<PromotionVetoProvider>[0]) => DreamCandidate | Promise<DreamCandidate>,
): PromotionVetoProvider {
  return async (context) => {
    const candidate = await candidateFor(context);
    const verdict = await evaluateCandidate(candidate);
    return vetoesFromVerdict(verdict);
  };
}
