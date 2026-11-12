import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

// Compiled test lives at harness/dist/test/, so the repo root is five up.
const REPO_ROOT = resolve(import.meta.dirname, "../../../../..");
const SCRIPT = resolve(REPO_ROOT, "scripts/frozen-weights-check.mjs");

const run = (args: string[]) => {
  const r = spawnSync(process.execPath, [SCRIPT, ...args], { encoding: "utf8", timeout: 60_000 });
  return { code: r.status ?? 1, stdout: r.stdout ?? "", stderr: r.stderr ?? "" };
};

test("frozen-weights gate: this repo's mutation surfaces are clean (ADR-313 structural check)", () => {
  assert.ok(existsSync(SCRIPT), `expected ${SCRIPT} to exist`);
  const result = run([]);
  assert.equal(result.code, 0, `check failed:\n${result.stderr}`);
  assert.ok(result.stdout.includes("frozen-weights-check: OK"));
  // The harness src dir (this loop's own home) is one of the enforced surfaces.
  assert.ok(result.stdout.includes("mutation surfaces"));
});

test("frozen-weights gate: --self-test passes (positive + negative fixtures)", () => {
  const result = run(["--self-test"]);
  assert.equal(result.code, 0, `self-test failed:\n${result.stdout}\n${result.stderr}`);
  assert.ok(result.stdout.includes("self-test OK"));
  assert.ok(!result.stdout.includes("FAIL"));
});
