/**
 * Repository optimization manifest (ADR-335).
 *
 * NVIDIA's Dynamo agent skillpack (ai-dynamo/dynamo PR #13557, 2026-08-21,
 * Apache-2.0) makes an optimizing agent state its objective, build a
 * benchmark, isolate one variable, and justify the spend before consuming GPU
 * time. Most of that already existed here as *code* — the closed lever set in
 * `benchmark.ts`, the single-lever step in `flywheel.ts`, the paired bootstrap
 * in `statistics.ts`, the conjunctive vetoes in `vetoes.ts`. What was missing
 * is the ability for a repository to *declare* those choices in one auditable
 * place instead of having them implied by five modules.
 *
 * This module loads that declaration and derives a per-experiment ADR-282
 * research manifest from it. The manifest is the standing policy; the research
 * manifest is one run under that policy.
 *
 * STATUS: this module is currently INERT. Nothing in `runRuvectorFlywheel`
 * loads a manifest, and `declaredObjectives()` has no caller — the promotion
 * rule uses `DEFAULT_OBJECTIVES` directly. So `promotion.pareto` is a
 * *declaration* that is checked for self-consistency, not policy that is
 * enforced at runtime. Registration in `schema_validate.py` is likewise by
 * directory glob, so naming the schema there is documentation rather than
 * wiring. Treat a passing manifest as "this repository has stated its intent
 * coherently", not as "this repository is being optimized under it".
 *
 * Scope note, deliberately narrow: NVIDIA's skillpack also carries
 * domain-specific optimization *knowledge* (what to try, in what order) and an
 * adversarial-review step performed by a separate agent. This manifest encodes
 * neither. It declares the frame — objective, levers, invariants, budget — and
 * says nothing about strategy. Claiming parity with the skillpack on the
 * strength of this file would be wrong.
 */

import { readFile } from "node:fs/promises";
import { POLICY_LEVER_RANGES, POLICY_LEVERS } from "./benchmark.js";
import { DEFAULT_OBJECTIVES, type ObjectiveSpec } from "./pareto.js";

export interface ManifestObjective {
  name: string;
  direction: "maximize" | "minimize";
  metric: string;
  computed_by: string;
  minimum_meaningful_effect: number;
}

export interface ManifestLever {
  name: string;
  values?: string[];
  bounds?: { minimum: number; maximum: number };
}

export interface OptimizationManifest {
  schema_version: 1;
  repository: string;
  objective: ManifestObjective;
  benchmark: { binary: string; runner_sets: string[]; isolated: true };
  levers: ManifestLever[];
  protected_invariants: { required_veto_providers: string[]; hard_regressions: string[] };
  budget: {
    cost_ceiling_factor: number;
    max_generations: number;
    wall_clock_seconds: number;
    risk_budget_total?: number;
  };
  promotion: {
    comparison: "paired";
    alpha: 0.05;
    outcome_rules: Record<string, string>;
    pareto: { objectives: ObjectiveSpec[]; require_non_dominated: true };
  };
}

const RUNNER_SETS = new Set(["core", "core-lsm", "all"]);

function requireObject(value: unknown, what: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`optimization manifest ${what} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireFinite(value: unknown, what: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`optimization manifest ${what} must be a finite number`);
  }
  return value;
}

function requireStringArray(value: unknown, what: string): string[] {
  if (!Array.isArray(value) || value.length === 0
    || value.some((entry) => typeof entry !== "string" || entry.length === 0)) {
    throw new Error(`optimization manifest ${what} must be a non-empty array of strings`);
  }
  return value as string[];
}

/**
 * Validate a parsed manifest, including the check the JSON Schema cannot make.
 *
 * The schema carries a `lever_name` enum, but an enum is a second copy of the
 * runner's closed lever set and copies drift. The authority is
 * `POLICY_LEVERS`, exported from `benchmark.ts` — the same list
 * `normalizePolicy` enforces — so a manifest can never declare a lever the
 * runner would reject at spawn time.
 */
export function assertOptimizationManifest(value: unknown): asserts value is OptimizationManifest {
  const manifest = requireObject(value, "document");
  if (manifest.schema_version !== 1) {
    throw new Error("optimization manifest must be schema_version 1");
  }
  if (typeof manifest.repository !== "string" || !manifest.repository) {
    throw new Error("optimization manifest requires a repository");
  }

  const objective = requireObject(manifest.objective, "objective");
  if (objective.direction !== "maximize" && objective.direction !== "minimize") {
    throw new Error("optimization manifest objective.direction must be maximize or minimize");
  }
  for (const field of ["name", "metric", "computed_by"]) {
    if (typeof objective[field] !== "string" || !objective[field]) {
      throw new Error(`optimization manifest objective.${field} must be a non-empty string`);
    }
  }
  const effect = requireFinite(objective.minimum_meaningful_effect, "objective.minimum_meaningful_effect");
  if (effect <= 0) {
    throw new Error("optimization manifest objective.minimum_meaningful_effect must exceed zero");
  }

  const benchmark = requireObject(manifest.benchmark, "benchmark");
  if (typeof benchmark.binary !== "string" || !benchmark.binary) {
    throw new Error("optimization manifest benchmark.binary must be a non-empty string");
  }
  if (benchmark.isolated !== true) {
    throw new Error("optimization manifest benchmark.isolated must be true");
  }
  for (const runnerSet of requireStringArray(benchmark.runner_sets, "benchmark.runner_sets")) {
    if (!RUNNER_SETS.has(runnerSet)) {
      throw new Error(`optimization manifest declares unknown runner_set: ${runnerSet}`);
    }
  }

  if (!Array.isArray(manifest.levers) || manifest.levers.length === 0) {
    throw new Error("optimization manifest requires at least one lever");
  }
  const allowed = new Set(POLICY_LEVERS);
  const seen = new Set<string>();
  for (const raw of manifest.levers) {
    const lever = requireObject(raw, "lever");
    const name = lever.name;
    if (typeof name !== "string" || !allowed.has(name)) {
      throw new Error(
        `optimization manifest declares a lever the runner would reject: ${String(name)}`,
      );
    }
    if (seen.has(name)) throw new Error(`optimization manifest repeats lever ${name}`);
    seen.add(name);
    // A declared range or value set must be one the runner can actually
    // effect. Otherwise a manifest can advertise a lever setting that
    // `normalizePolicy` rejects at spawn time -- an authorization that looks
    // real and does nothing.
    const runnerRange = POLICY_LEVER_RANGES[name];
    if (lever.values !== undefined) {
      const values = requireStringArray(lever.values, `lever ${name} values`);
      if (runnerRange) {
        for (const raw of values) {
          const parsed = Number(raw);
          if (!Number.isSafeInteger(parsed)
            || parsed < runnerRange.minimum || parsed > runnerRange.maximum) {
            throw new Error(
              `optimization manifest lever ${name} declares ${raw}, outside the runner's `
              + `[${runnerRange.minimum}, ${runnerRange.maximum}]`,
            );
          }
        }
      } else {
        // The enum lever. `POLICY_LEVER_RANGES` covers only the integer
        // levers, so without this the same defect the range check closes for
        // them stays open here: a declared `runner_set` the runner would
        // refuse at spawn time reads as authorized.
        for (const raw of values) {
          if (!RUNNER_SETS.has(raw)) {
            throw new Error(
              `optimization manifest lever ${name} declares unknown runner_set: ${raw}`,
            );
          }
        }
      }
    }
    if (lever.bounds !== undefined) {
      const bounds = requireObject(lever.bounds, `lever ${name} bounds`);
      const minimum = requireFinite(bounds.minimum, `lever ${name} bounds.minimum`);
      const maximum = requireFinite(bounds.maximum, `lever ${name} bounds.maximum`);
      if (minimum > maximum) {
        throw new Error(`optimization manifest lever ${name} has minimum above maximum`);
      }
      if (!runnerRange) {
        throw new Error(`optimization manifest lever ${name} takes values, not bounds`);
      }
      if (minimum < runnerRange.minimum || maximum > runnerRange.maximum) {
        throw new Error(
          `optimization manifest lever ${name} declares [${minimum}, ${maximum}], outside the `
          + `runner's [${runnerRange.minimum}, ${runnerRange.maximum}]`,
        );
      }
    }
  }

  const invariants = requireObject(manifest.protected_invariants, "protected_invariants");
  requireStringArray(invariants.required_veto_providers, "protected_invariants.required_veto_providers");
  requireStringArray(invariants.hard_regressions, "protected_invariants.hard_regressions");

  const budget = requireObject(manifest.budget, "budget");
  const ceiling = requireFinite(budget.cost_ceiling_factor, "budget.cost_ceiling_factor");
  if (ceiling <= 1) throw new Error("optimization manifest budget.cost_ceiling_factor must exceed 1");
  if (!Number.isSafeInteger(budget.max_generations) || (budget.max_generations as number) < 1) {
    throw new Error("optimization manifest budget.max_generations must be a positive integer");
  }
  if (!Number.isSafeInteger(budget.wall_clock_seconds) || (budget.wall_clock_seconds as number) < 1) {
    throw new Error("optimization manifest budget.wall_clock_seconds must be a positive integer");
  }

  const promotion = requireObject(manifest.promotion, "promotion");
  if (promotion.comparison !== "paired") {
    throw new Error("optimization manifest promotion.comparison must be paired");
  }
  if (promotion.alpha !== 0.05) throw new Error("optimization manifest promotion.alpha must be 0.05");
  const pareto = requireObject(promotion.pareto, "promotion.pareto");
  if (pareto.require_non_dominated !== true) {
    throw new Error("optimization manifest cannot disable the non-dominated requirement");
  }
  if (!Array.isArray(pareto.objectives) || pareto.objectives.length === 0) {
    throw new Error("optimization manifest promotion.pareto requires objectives");
  }
  // Cross-check against the vector the promotion rule actually uses. A
  // manifest declaring `{primary, minimize}` is harmless only while this
  // module is inert; the day it is wired in, an inverted direction is a
  // promotion gate that rewards regressions. Refuse it now, while the cost of
  // being wrong is a failing test rather than a bad promotion.
  const gateDirections = new Map(DEFAULT_OBJECTIVES.map((o) => [o.name, o.direction]));
  for (const raw of pareto.objectives) {
    const objectiveSpec = requireObject(raw, "pareto objective");
    const name = objectiveSpec.name;
    if (typeof name !== "string" || !name) {
      throw new Error("optimization manifest pareto objective requires a name");
    }
    if (objectiveSpec.direction !== "maximize" && objectiveSpec.direction !== "minimize") {
      throw new Error(`optimization manifest pareto objective ${name} needs a direction`);
    }
    const expected = gateDirections.get(name);
    if (expected === undefined) {
      throw new Error(
        `optimization manifest declares pareto objective ${name}, which the promotion gate `
        + `does not evaluate (it uses ${[...gateDirections.keys()].join(", ")})`,
      );
    }
    if (expected !== objectiveSpec.direction) {
      throw new Error(
        `optimization manifest declares ${name} as ${objectiveSpec.direction}, but the promotion `
        + `gate treats it as ${expected}`,
      );
    }
  }
}

/** Read and validate a manifest from disk. */
export async function loadOptimizationManifest(path: string): Promise<OptimizationManifest> {
  const raw = await readFile(path, "utf8");
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`optimization manifest ${path} is not valid JSON: ${(error as Error).message}`);
  }
  assertOptimizationManifest(parsed);
  return parsed;
}

/** The lever names this manifest permits an experiment to vary. */
export function declaredLevers(manifest: OptimizationManifest): string[] {
  return manifest.levers.map((lever) => lever.name);
}

/** The Pareto objectives this manifest promotes against. */
export function declaredObjectives(manifest: OptimizationManifest): ObjectiveSpec[] {
  return manifest.promotion.pareto.objectives.map((objective) => ({
    name: objective.name,
    direction: objective.direction,
  }));
}

export interface DerivationInput {
  manifest: OptimizationManifest;
  /** The single lever this experiment varies. */
  independentVariable: string;
  commit: string;
  revision: number;
  claim: string;
  explorationSeeds: number[];
  confirmationSeeds: number[];
}

/**
 * Derive the ADR-282 research manifest fields this document fixes.
 *
 * Only the fields the standing policy determines are produced here. Anything
 * that is genuinely per-run — datasets, embedding space identity, environment,
 * memory accounting — is left to the caller, because inventing it from a
 * repository-level document would fabricate provenance rather than record it.
 *
 * The single-independent-variable rule is enforced here as well as by
 * `nextParameter`, and the variable must be one the manifest declared.
 */
export function deriveResearchManifestFields(input: DerivationInput): Record<string, unknown> {
  const { manifest, independentVariable } = input;
  if (!declaredLevers(manifest).includes(independentVariable)) {
    throw new Error(
      `independent variable ${independentVariable} is not a lever declared by the optimization manifest`,
    );
  }
  if (!/^[0-9a-f]{40}$/.test(input.commit)) {
    throw new Error("derived research manifest requires a full 40-character commit");
  }
  if (!Number.isSafeInteger(input.revision) || input.revision < 1) {
    throw new Error("derived research manifest requires a positive integer revision");
  }
  if (!input.claim) throw new Error("derived research manifest requires a claim");
  if (input.explorationSeeds.length < 5 || input.confirmationSeeds.length < 5) {
    throw new Error("derived research manifest requires at least five seeds per phase");
  }
  const shared = input.explorationSeeds.filter((seed) => input.confirmationSeeds.includes(seed));
  if (shared.length > 0) {
    throw new Error(`exploration and confirmation seeds must be disjoint; shared: ${shared.join(", ")}`);
  }
  return {
    schema_version: 1,
    commit: input.commit,
    revision: input.revision,
    phase: "confirmation",
    claim: input.claim,
    independent_variable: independentVariable,
    decision_rule: {
      primary_metric: manifest.objective.metric,
      minimum_meaningful_effect: manifest.objective.minimum_meaningful_effect,
      expected_direction: manifest.objective.direction === "maximize" ? "greater" : "less",
      alpha: manifest.promotion.alpha,
      comparison: manifest.promotion.comparison,
      sample_size_or_power_rationale:
        `derived from ${manifest.repository} optimization manifest (ADR-335)`,
      outcome_rules: manifest.promotion.outcome_rules,
    },
    exploration_seeds: [...input.explorationSeeds],
    confirmation_seeds: [...input.confirmationSeeds],
  };
}
