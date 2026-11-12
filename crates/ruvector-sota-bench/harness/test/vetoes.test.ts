import assert from "node:assert/strict";
import test from "node:test";
import { capabilityVetoProvider } from "../src/vetoes.js";

test("workspace and Weight-EFT evidence are hard promotion vetoes", async () => {
  const provider = capabilityVetoProvider(() => ({
    workspaceReceipts: [{
      promptHash: "p",
      modelId: "m",
      lensId: "l",
      layerRange: [1, 2],
      positions: [1],
      topTokens: [],
      flags: {
        promptInjection: false,
        evalAwareness: true,
        hiddenObjective: false,
        refusalConflict: false,
      },
      triggers: [{ concept: "eval-awareness", layer: 1, position: 1, score: 1, critical: true }],
      workspaceDrift: 0.4,
      entropyTrajectory: [],
      createdAt: "fixture",
    }],
    trajectories: [{
      instance_id: "fixture",
      model: "fixture",
      tier: "cheap",
      resolved: true,
      model_patch: "",
      messages: [{
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "1",
          type: "function",
          function: { name: "read_file", arguments: "{\"path\":\"gold_patch\"}" },
        }],
      }],
    }],
  }));
  const reasons = await provider({
    policy: { ef_search: "100" },
    suite: { id: "fixture", items: [] },
    observations: [],
  });
  assert.ok(reasons.includes("workspace_critical_trigger"));
  assert.ok(reasons.includes("workspace_drift_threshold_exceeded"));
  assert.ok(reasons.includes("weight_eft_gold-read"));
});
