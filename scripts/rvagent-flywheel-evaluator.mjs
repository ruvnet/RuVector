#!/usr/bin/env node
// rvAgent's Evaluator for @metaharness/flywheel (ADR-278).
//
// The flywheel is deliberately host-agnostic — it knows only candidates,
// scores, gates, receipts, and lineage. Everything rvAgent-specific enters
// through this one seam, which is also the trust boundary: the four Score axes
// are where all host meaning lands, and a dishonest projection defeats every
// downstream guarantee the gate provides.
//
// Usage as a library:
//   import { makeRvagentEvaluator } from './rvagent-flywheel-evaluator.mjs';
//   const evaluator = makeRvagentEvaluator({ runItem });
//
// `runItem(policy, item) -> RunOutcome` is injected so this file stays
// testable without spawning real agent runs.

/** Cost-per-win when a policy won nothing.
 *
 * Must match rvagent_core::policy::COST_PER_WIN_NO_WINS. NOT Infinity: JSON has
 * no infinity, so it serializes to null, and the gate's
 * `candidate.costPerWin > baseline.costPerWin` reads `null > n` as false —
 * meaning a policy that won nothing would silently pass the cost clause.
 */
export const COST_PER_WIN_NO_WINS = Number.MAX_VALUE;

/**
 * Aggregate per-item run outcomes into the flywheel's four Score axes.
 *
 * @param {Array<{itemId?: string, succeeded: boolean, madeChanges: boolean, costUsd: number, regressed?: boolean}>} outcomes
 * @returns {{primary: number, noopRate: number, costPerWin: number, regressed: boolean}}
 */
export function scoreFromOutcomes(outcomes) {
  // Zero runs must never look like a clean sweep to the gate.
  if (!Array.isArray(outcomes) || outcomes.length === 0) {
    return { primary: 0, noopRate: 1, costPerWin: COST_PER_WIN_NO_WINS, regressed: false };
  }

  const total = outcomes.length;
  const wins = outcomes.filter((o) => o.succeeded).length;
  // A run that reports success while committing nothing is still a no-op —
  // that is the whole point of the axis. A policy must not earn promotion by
  // making the agent talk rather than act.
  const noops = outcomes.filter((o) => !o.madeChanges).length;
  const cost = outcomes.reduce((sum, o) => sum + (Number(o.costUsd) || 0), 0);

  return {
    primary: wins / total,
    noopRate: noops / total,
    costPerWin: wins === 0 ? COST_PER_WIN_NO_WINS : cost / wins,
    regressed: outcomes.some((o) => o.regressed === true),
  };
}

/** Levers rvAgent knows how to apply. Must match rvagent_core::policy::KNOWN_LEVERS. */
export const KNOWN_LEVERS = [
  'max_iterations',
  'parallel_tools',
  'max_parallel_tools',
  'loop_repeat_threshold',
  'keep_last_observations',
  'max_tool_result_bytes',
  'system_prompt_suffix',
  'compaction_rubric',
];

/**
 * Reject a policy naming a lever rvAgent does not apply.
 *
 * Throwing is deliberate. A mutation to an unapplied lever produces a run
 * identical to baseline; the optimizer would read that as "no effect" and burn
 * generations proposing more of them. Failing loudly keeps the search honest.
 */
export function assertKnownLevers(policy) {
  const unknown = Object.keys(policy ?? {}).filter((k) => !KNOWN_LEVERS.includes(k));
  if (unknown.length > 0) {
    throw new Error(
      `policy names levers rvAgent does not apply: ${unknown.join(', ')}. ` +
        `Known levers: ${KNOWN_LEVERS.join(', ')}`,
    );
  }
}

/**
 * Build an Evaluator for `runFlywheelGenerations`.
 *
 * @param {{runItem: (policy: object, item: unknown) => Promise<object>}} deps
 * @returns {(policy: object, suite: {id: string, items: unknown[]}) => Promise<object>}
 */
export function makeRvagentEvaluator({ runItem }) {
  if (typeof runItem !== 'function') {
    throw new TypeError('makeRvagentEvaluator requires a runItem function');
  }

  return async function evaluate(policy, suite) {
    assertKnownLevers(policy);

    const items = suite?.items ?? [];
    const outcomes = [];
    for (const item of items) {
      // Sequential on purpose: concurrent runs contend for the same workspace
      // and would make cost and wall-clock unattributable per item.
      outcomes.push(await runItem(policy, item));
    }

    // A dropped item would silently shrink the denominator and inflate every
    // axis. Refuse rather than score a partial suite as if it were complete.
    if (outcomes.length !== items.length) {
      throw new Error(
        `evaluator produced ${outcomes.length} outcomes for ${items.length} items`,
      );
    }

    return scoreFromOutcomes(outcomes);
  };
}
