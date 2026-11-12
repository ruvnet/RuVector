/**
 * In-repo Pareto frontier over the promotion objective vector (ADR-335).
 *
 * `darwin.ts` passes `selection: "pareto"` to `evolve()` from the external
 * `@metaharness/darwin` package, so the frontier that guides Darwin's *search*
 * is owned by a dependency. This module is deliberately separate: it is the
 * frontier that this repository's own *promotion rule* consults, so a
 * dependency bump cannot silently change what we are willing to promote.
 * Both exist on purpose. Neither replaces the other.
 *
 * Objectives are declared with an explicit direction rather than inferred, and
 * every value is checked finite at a choke point before any comparison runs.
 * A NaN silently defeats `<` and `>` in both directions, which would make a
 * dominated candidate look non-dominated; this module refuses such input
 * rather than comparing it.
 */

/** Which way is better for one objective. */
export type ObjectiveDirection = "maximize" | "minimize";

export interface ObjectiveSpec {
  /** Key into the point's `values` record. */
  name: string;
  direction: ObjectiveDirection;
}

export interface ParetoPoint {
  /** Caller-chosen identity, echoed back in frontier membership. */
  id: string;
  values: Record<string, number>;
}

/**
 * The promotion objective vector: quality against the resource bought with it.
 *
 * Deliberately TWO axes, not three. Dominance requires "no worse on every
 * axis", so each axis added is one more chance for the condition to fail:
 * more axes can only ever turn *dominated* into *not dominated*, never the
 * reverse. A Pareto gate therefore gets monotonically weaker as axes are
 * added, and this gate exists to block trades dressed as wins.
 *
 * `p99Us` was on this vector and has been removed, because it is not
 * independent evidence — it is already a factor of `costPerWin`
 * (`memoryMb * p99Us / qps`) and a term inside `primary`'s `darwinScore`.
 * Counting it a third time bought no information and cost real blocking
 * power: with it present, a candidate burning 10x the memory to halve tail
 * latency is *not* dominated and clears the gate; with two axes the incumbent
 * dominates it and the gate fires.
 */
export const DEFAULT_OBJECTIVES: readonly ObjectiveSpec[] = Object.freeze([
  Object.freeze({ name: "primary", direction: "maximize" as const }),
  Object.freeze({ name: "costPerWin", direction: "minimize" as const }),
]);

function assertObjectives(objectives: readonly ObjectiveSpec[]): void {
  if (objectives.length === 0) throw new Error("pareto requires at least one objective");
  const seen = new Set<string>();
  for (const objective of objectives) {
    if (!objective.name) throw new Error("pareto objective requires a name");
    if (objective.direction !== "maximize" && objective.direction !== "minimize") {
      throw new Error(`pareto objective ${objective.name} needs direction maximize or minimize`);
    }
    if (seen.has(objective.name)) throw new Error(`duplicate pareto objective: ${objective.name}`);
    seen.add(objective.name);
  }
}

/**
 * Read one objective from a point, refusing anything that cannot be ordered.
 *
 * This is the choke point: every comparison in this module is fed from here,
 * so a non-finite value can never reach a `<`.
 */
function readObjective(point: ParetoPoint, objective: ObjectiveSpec): number {
  const value = point.values[objective.name];
  if (value === undefined) {
    throw new Error(`pareto point ${point.id} is missing objective ${objective.name}`);
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(
      `pareto point ${point.id} has a non-finite ${objective.name}; refusing to compare`,
    );
  }
  return value;
}

/**
 * Read every objective of every point before any of them are compared.
 *
 * This must happen as a separate pass. The comparison loop below short-circuits
 * on the first axis where `a` is worse, so validating inside that loop would
 * only ever check the axes the loop happens to reach — leaving a non-finite
 * value on a later axis completely unexamined. That is not a hypothetical: it
 * is precisely how a NaN reaches a comparison and reads as "not worse" on
 * every test, which is the failure this module exists to prevent.
 */
function assertComparable(
  points: readonly ParetoPoint[],
  objectives: readonly ObjectiveSpec[],
): void {
  for (const point of points) {
    for (const objective of objectives) readObjective(point, objective);
  }
}

/** True when `a` is at least as good on every objective and strictly better on one. */
export function dominates(
  a: ParetoPoint,
  b: ParetoPoint,
  objectives: readonly ObjectiveSpec[] = DEFAULT_OBJECTIVES,
): boolean {
  assertObjectives(objectives);
  assertComparable([a, b], objectives);
  let strictlyBetterSomewhere = false;
  for (const objective of objectives) {
    const left = readObjective(a, objective);
    const right = readObjective(b, objective);
    const better = objective.direction === "maximize" ? left > right : left < right;
    const worse = objective.direction === "maximize" ? left < right : left > right;
    if (worse) return false;
    if (better) strictlyBetterSomewhere = true;
  }
  return strictlyBetterSomewhere;
}

/**
 * The non-dominated subset, in input order.
 *
 * Points that tie on every objective do not dominate each other, so all of
 * them stay on the frontier. That is the correct reading of Pareto dominance
 * and it matters here: it keeps an equal-scoring candidate eligible rather
 * than letting arrival order decide.
 */
export function frontier(
  points: readonly ParetoPoint[],
  objectives: readonly ObjectiveSpec[] = DEFAULT_OBJECTIVES,
): ParetoPoint[] {
  assertObjectives(objectives);
  assertComparable(points, objectives);
  return points.filter(
    (candidate) => !points.some((other) => other !== candidate && dominates(other, candidate, objectives)),
  );
}

export interface FrontierAdmission {
  /** False when some frontier member dominates the candidate. */
  admitted: boolean;
  /** Ids of the frontier members that dominate it, sorted for stable reasons. */
  dominatedBy: string[];
}

/**
 * Whether a candidate is non-dominated by the current frontier.
 *
 * An empty frontier admits: with nothing to compare against there is no
 * evidence of domination, and the surrounding promotion rule still has to
 * pass every other gate. This function can only ever *block* a promotion —
 * it never supplies a reason to promote.
 */
export function admitToFrontier(
  candidate: ParetoPoint,
  current: readonly ParetoPoint[],
  objectives: readonly ObjectiveSpec[] = DEFAULT_OBJECTIVES,
): FrontierAdmission {
  assertObjectives(objectives);
  assertComparable([candidate, ...current], objectives);
  const dominatedBy = current
    .filter((member) => dominates(member, candidate, objectives))
    .map((member) => member.id)
    .sort();
  return { admitted: dominatedBy.length === 0, dominatedBy };
}
