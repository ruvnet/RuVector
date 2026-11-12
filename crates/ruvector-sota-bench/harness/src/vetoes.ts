import { assertNoLiveCredential } from "@metaharness/redblue";
import { detectRewardHack, type DarwinTrajectory } from "@metaharness/weight-eft";
import { workspaceProbeScore } from "@metaharness/workspace-probe";
import type { WorkspaceLensReceipt } from "@metaharness/workspace-lens";
import type { Policy, Suite } from "@metaharness/flywheel";
import type { BenchmarkObservation } from "./benchmark.js";

export interface VetoContext {
  policy: Policy;
  suite: Suite;
  observations: BenchmarkObservation[];
}

export type PromotionVetoProvider = (context: VetoContext) => Promise<string[]> | string[];

/**
 * Compose veto providers conjunctively: the union of every provider's
 * reasons, deduplicated and sorted. Lets additive gates (e.g. the
 * dream-machine evaluation stage and the HarnessRisk (arXiv:2608.17597)
 * lifecycle-security gate, ADR-317) run side by side — any one provider can
 * block promotion; none can rescue it.
 */
export function composeVetoProviders(
  ...providers: readonly PromotionVetoProvider[]
): PromotionVetoProvider {
  return async (context) => {
    const reasons: string[] = [];
    for (const provider of providers) {
      reasons.push(...(await provider(context)));
    }
    return [...new Set(reasons)].sort();
  };
}

export interface CapabilityEvidence {
  workspaceReceipts?: readonly WorkspaceLensReceipt[];
  trajectories?: readonly DarwinTrajectory[];
}

/**
 * Promotion vetoes are conjunctive. Missing optional evidence adds no credit;
 * evidence that is present can only reject, never rescue a weak benchmark.
 */
export function capabilityVetoProvider(
  evidenceFor: (context: VetoContext) => CapabilityEvidence | Promise<CapabilityEvidence>,
): PromotionVetoProvider {
  return async (context) => {
    const reasons: string[] = [];
    try {
      assertNoLiveCredential(JSON.stringify(context.policy));
    } catch {
      reasons.push("redblue_live_credential_detected");
    }
    const evidence = await evidenceFor(context);
    if (evidence.workspaceReceipts) {
      const score = workspaceProbeScore(evidence.workspaceReceipts);
      if (score.criticalRate > 0) reasons.push("workspace_critical_trigger");
      if (score.meanDrift >= 0.25) reasons.push("workspace_drift_threshold_exceeded");
    }
    for (const trajectory of evidence.trajectories ?? []) {
      for (const finding of detectRewardHack(trajectory)) {
        reasons.push(`weight_eft_${finding.kind}`);
      }
    }
    return [...new Set(reasons)].sort();
  };
}
