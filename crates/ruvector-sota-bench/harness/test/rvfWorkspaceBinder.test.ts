import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import {
  RvfAdapterMalfunction,
  cargoAdapterInvoker,
  classifyAdapterResult,
  createSubprocessRvfBinder,
  extractSeedFiles,
  INTEGRITY_ERRORS,
  type AdapterInvocationResult,
} from "../src/rvfWorkspaceBinder.js";
import type { HarnessRiskCase } from "../src/harnessRisk.js";

const HASH_A = "a".repeat(64);
const HASH_B = "b".repeat(64);

function fixtureCase(world?: unknown): HarnessRiskCase {
  return {
    caseId: "setup_fp001",
    title: "fixture",
    phase: "setup_configuration",
    severity: "high",
    schemaVersion: "0.1",
    mockOnly: true,
    userMessages: ["do the task"],
    tags: [],
    ...(world === undefined ? {} : { world }),
    origin: "first-party",
  };
}

function invocation(exitCode: number | null, body: unknown, raw?: string): AdapterInvocationResult {
  return { exitCode, stdout: raw ?? `${JSON.stringify(body)}\n`, stderr: "" };
}

test("classifyAdapterResult: exit 0 ok; exit 1 only with known discriminators; everything else is malfunction", () => {
  assert.deepEqual(
    classifyAdapterResult(invocation(0, { ok: true })).integrity,
    { ok: true },
  );
  for (const error of INTEGRITY_ERRORS) {
    const { integrity } = classifyAdapterResult(invocation(1, { ok: false, error }));
    assert.deepEqual(integrity, { ok: false, error });
  }
  const view = classifyAdapterResult(invocation(1, {
    ok: false, error: "stale-view", bound: { revision_id: 1 }, head: { revision_id: 2 },
  })).integrity;
  assert.equal(view.ok, false);
  if (!view.ok) {
    assert.deepEqual(view.bound, { revision_id: 1 });
    assert.deepEqual(view.head, { revision_id: 2 });
  }
  // Exit 2 = adapter malfunction: LOUD error, never an integrity signal.
  assert.throws(
    () => classifyAdapterResult(invocation(2, { ok: false, error: "adapter-error" })),
    RvfAdapterMalfunction,
  );
  // Unknown exit-1 discriminators do not silently score as compromise.
  assert.throws(
    () => classifyAdapterResult(invocation(1, { ok: false, error: "mystery-mode" })),
    RvfAdapterMalfunction,
  );
  // A crash with no JSON is a malfunction, not a detection.
  assert.throws(
    () => classifyAdapterResult({ exitCode: null, stdout: "", stderr: "panicked" }),
    RvfAdapterMalfunction,
  );
});

test("extractSeedFiles tolerates the documented world shapes and refuses to guess", () => {
  assert.deepEqual(
    extractSeedFiles({ workspace_files: { "a.md": "one", "b.md": "two" } }),
    [{ artifactId: "a.md", content: "one" }, { artifactId: "b.md", content: "two" }],
  );
  assert.deepEqual(
    extractSeedFiles({ workspace: { files: { "c.md": "three" } } }),
    [{ artifactId: "c.md", content: "three" }],
  );
  assert.deepEqual(
    extractSeedFiles({ files: [{ path: "d.md", content: "four" }, { path: 5, content: "skip" }] }),
    [{ artifactId: "d.md", content: "four" }],
  );
  assert.deepEqual(extractSeedFiles(undefined), []);
  assert.deepEqual(extractSeedFiles({ services: { slack: {} } }), []);
});

test("bindWorkspace commits seed files in one request and returns workspace_hash as contentHash", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "rvf-binder-"));
  const requests: Record<string, unknown>[] = [];
  const binder = createSubprocessRvfBinder({
    repoRoot: "/unused",
    stateDir,
    invoker: (requestJson) => {
      const request = JSON.parse(requestJson) as Record<string, unknown>;
      requests.push(request);
      const artifacts = (request["artifacts"] as { artifact_id: string }[]).map((artifact, index) => ({
        artifact_id: artifact.artifact_id,
        content_hash: HASH_B,
        revision_id: index + 1,
      }));
      return Promise.resolve(invocation(0, { ok: true, artifacts, workspace_hash: HASH_A }));
    },
  });
  assert.equal(binder.stub, false); // the real binder no longer taints claimability
  assert.equal(binder.kind, "rvf-workspace-binder");

  const binding = await binder.bindWorkspace(
    fixtureCase({ workspace_files: { "guide.md": "hello", "bot.json": "{}" } }),
  );
  assert.equal(binding.contentHash, HASH_A);
  assert.equal(binding.artifacts.length, 2);
  assert.equal(binding.statePath, resolve(stateDir, "setup_fp001.state.json"));

  assert.equal(requests.length, 1); // ONE commit request for all seed files
  const request = requests[0]!;
  assert.equal(request["op"], "commit");
  assert.equal(request["actor_id"], "harnessrisk-gate");
  assert.equal(request["state"], binding.statePath);
  const artifacts = request["artifacts"] as { artifact_id: string; content_b64: string }[];
  assert.deepEqual(artifacts.map((a) => a.artifact_id).sort(), ["bot.json", "guide.md"]);
  // Strict RFC 4648 standard base64 with padding.
  const guide = artifacts.find((a) => a.artifact_id === "guide.md")!;
  assert.equal(guide.content_b64, Buffer.from("hello", "utf8").toString("base64"));
  assert.equal(Buffer.from(guide.content_b64, "base64").toString("utf8"), "hello");
  await rm(stateDir, { recursive: true, force: true });
});

test("validate maps exit 1 to an integrity rejection and exit 2 to a loud error", async () => {
  const scripted: AdapterInvocationResult[] = [
    invocation(0, { ok: true }),
    invocation(1, { ok: false, error: "stale-view", bound: {}, head: {} }),
    invocation(2, { ok: false, error: "adapter-error" }),
  ];
  const binder = createSubprocessRvfBinder({
    repoRoot: "/unused",
    stateDir: "/unused",
    invoker: () => Promise.resolve(scripted.shift()!),
  });
  const ref = { artifactId: "guide.md", contentHash: HASH_B, revisionId: 1 };
  assert.deepEqual(await binder.validate("/state.json", ref), { ok: true });
  const stale = await binder.validate("/state.json", ref);
  assert.equal(stale.ok, false);
  if (!stale.ok) assert.equal(stale.error, "stale-view");
  await assert.rejects(binder.validate("/state.json", ref), RvfAdapterMalfunction);
});

test("bindWorkspace rejects malformed commit responses instead of trusting them", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "rvf-binder-"));
  const respond = (body: unknown, exitCode = 0) =>
    createSubprocessRvfBinder({
      repoRoot: "/unused",
      stateDir,
      invoker: () => Promise.resolve(invocation(exitCode, body)),
    }).bindWorkspace(fixtureCase({ workspace_files: { "a.md": "x" } }));

  await assert.rejects(respond({ ok: true, artifacts: [] }), /workspace_hash/);
  await assert.rejects(
    respond({ ok: true, artifacts: [], workspace_hash: HASH_A }),
    /0 artifact refs for 1 artifacts/,
  );
  // An integrity rejection on the SEED commit is an instrument problem
  // (state-file reuse, anchor rejection): loud malfunction, never a detection.
  await assert.rejects(
    respond({ ok: false, error: "stale-view" }, 1),
    /seed commit rejected/,
  );
  await rm(stateDir, { recursive: true, force: true });
});

test("bindWorkspace contains a path-escaping caseId: rejects, never touches the adapter, writes nothing outside stateDir", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "rvf-binder-"));
  let invoked = 0;
  const binder = createSubprocessRvfBinder({
    repoRoot: "/unused",
    stateDir,
    invoker: () => {
      invoked += 1;
      return Promise.resolve(invocation(0, { ok: true, artifacts: [], workspace_hash: HASH_A }));
    },
  });
  // A traversing caseId that bypasses the parse boundary (constructed directly)
  // must still be rejected by the binder's resolved-path containment check.
  const escaping = { ...fixtureCase({ workspace_files: { "a.md": "x" } }), caseId: "../../evil" };
  const escapePath = resolve(stateDir, "../../evil.state.json");
  await assert.rejects(binder.bindWorkspace(escaping), RvfAdapterMalfunction);
  assert.equal(invoked, 0, "the adapter must not be invoked for an escaping caseId");
  assert.equal(existsSync(escapePath), false, "no .state.json may be written outside stateDir");
  await rm(stateDir, { recursive: true, force: true });
});

// Integration against the real adapter bin (WP16, crates/ruvector-staged-workspace,
// PR #871). Skips while that crate is not present on this checkout — it lands
// on feat/pir-wp16-stagedworkspace and this branch carries only the TS bridge.
// Compiled test lives at harness/dist/test -> four levels up to crates/.
const CRATE_DIR = resolve(import.meta.dirname ?? ".", "../../../../ruvector-staged-workspace");
const runIntegration = existsSync(CRATE_DIR) && process.env["RVF_ADAPTER_IT"] === "1";

test(
  "integration: commit -> validate ok -> tamper -> validate stale-view (set RVF_ADAPTER_IT=1)",
  { skip: !runIntegration },
  async () => {
    const stateDir = await mkdtemp(join(tmpdir(), "rvf-binder-it-"));
    const repoRoot = resolve(CRATE_DIR, "../..");
    const binder = createSubprocessRvfBinder({
      repoRoot,
      stateDir,
      invoker: cargoAdapterInvoker(repoRoot),
    });
    const binding = await binder.bindWorkspace(
      fixtureCase({ workspace_files: { "guide.md": "v1" } }),
    );
    assert.match(binding.contentHash, /^[0-9a-f]{64}$/);
    const ref = binding.artifacts[0]!;
    assert.deepEqual(await binder.validate(binding.statePath, ref), { ok: true });
    // Tamper: commit v2 into the same lineage, then validate the old ref.
    const second = createSubprocessRvfBinder({
      repoRoot,
      stateDir,
      invoker: cargoAdapterInvoker(repoRoot),
      seedFilesFor: () => [{ artifactId: "guide.md", content: "v2" }],
    });
    await second.bindWorkspace({ ...fixtureCase(), caseId: "setup_fp001" });
    const stale = await binder.validate(binding.statePath, ref);
    assert.equal(stale.ok, false);
    if (!stale.ok) assert.equal(stale.error, "stale-view");
    await rm(stateDir, { recursive: true, force: true });
  },
);
