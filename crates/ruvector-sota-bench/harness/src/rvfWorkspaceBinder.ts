/**
 * Real WP16 RvfWorkspaceBinder (issue #863, ADR-318) — subprocess bridge to
 * the `staged-workspace-adapter` bin in `crates/ruvector-staged-workspace`
 * (PR #871). Replaces stubRvfBinder() for the treated leg of ruv's Wave-2
 * acceptance test (harnessRiskAcceptance.ts) once both branches merge.
 *
 * FROZEN CONTRACT (source of truth: module docs in
 * crates/ruvector-staged-workspace/src/adapter.rs):
 *   - One JSON request on stdin, one JSON response line on stdout.
 *   - Exit 0 = ok.
 *   - Exit 1 = INTEGRITY REJECTION, error in {"stale-view",
 *     "binding-mismatch", "unknown-artifact", "anchor-rejected"} — the ONLY
 *     exit that may score as a detected compromise (ADR-318 hard rejection).
 *   - Exit 2 (error "adapter-error"), crashes, and unparsable output are
 *     ADAPTER MALFUNCTION — surfaced as a loud harness error, never scored
 *     as compromise. A broken measuring instrument must not manufacture
 *     detections.
 *   - commit: {"op":"commit","actor_id","artifacts":[{artifact_id,
 *     content_b64}],"state"} -> {"ok":true,"artifacts":[{artifact_id,
 *     content_hash,revision_id}],"workspace_hash"} where workspace_hash is
 *     commit-order independent over all current lineage heads — used as the
 *     seam's single contentHash.
 *   - validate: {"op":"validate","artifact_id","content_hash",
 *     "revision_id","state"} -> exit 0 ok / exit 1 with discriminator and
 *     bound/head refs on view errors.
 *   - State file: one WorkspaceSnapshot JSON per case run, created on first
 *     commit; content_b64 is strict RFC 4648 standard base64 with padding.
 */
import { spawn } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import type { HarnessRiskCase } from "./harnessRisk.js";
import type { RvfWorkspaceBinder } from "./harnessRiskAcceptance.js";

/** The adapter's exit-1 integrity-rejection discriminators (frozen). */
export const INTEGRITY_ERRORS = Object.freeze([
  "stale-view",
  "binding-mismatch",
  "unknown-artifact",
  "anchor-rejected",
] as const);

export type IntegrityError = (typeof INTEGRITY_ERRORS)[number];

/** Thrown on exit 2, crashes, or unparsable output — never a compromise. */
export class RvfAdapterMalfunction extends Error {
  constructor(message: string, public readonly exitCode: number | null, public readonly raw: string) {
    super(`staged-workspace-adapter malfunction: ${message}`);
    this.name = "RvfAdapterMalfunction";
  }
}

export interface AdapterArtifactRef {
  artifactId: string;
  contentHash: string;
  revisionId: number;
}

export interface WorkspaceBinding {
  /** workspace_hash over all lineage heads — the seam's contentHash. */
  contentHash: string;
  artifacts: readonly AdapterArtifactRef[];
  /** Absolute path of this case run's state file (for later validate calls). */
  statePath: string;
}

export type ValidationResult =
  | { ok: true }
  | { ok: false; error: IntegrityError; bound?: unknown; head?: unknown };

export interface AdapterInvocationResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
}

/** One adapter round trip: request JSON in, {exitCode, stdout} out. Injectable. */
export type AdapterInvoker = (requestJson: string) => Promise<AdapterInvocationResult>;

/** Default invoker: cargo run -q -p ruvector-staged-workspace --bin staged-workspace-adapter. */
export function cargoAdapterInvoker(repoRoot: string): AdapterInvoker {
  return (requestJson) =>
    new Promise((resolvePromise, rejectPromise) => {
      const child = spawn(
        "cargo",
        ["run", "-q", "-p", "ruvector-staged-workspace", "--bin", "staged-workspace-adapter"],
        { cwd: repoRoot, stdio: ["pipe", "pipe", "pipe"] },
      );
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString("utf8"); });
      child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString("utf8"); });
      child.on("error", rejectPromise);
      child.on("close", (exitCode) => resolvePromise({ exitCode, stdout, stderr }));
      child.stdin.end(requestJson);
    });
}

/** A seed workspace file extracted from a case's world. */
export interface SeedFile {
  artifactId: string;
  content: string;
}

/**
 * Default extraction of seed workspace files from the upstream `world`
 * component. Tolerates the shapes observed in the HarnessRisk
 * (arXiv:2608.17597) dataset docs ("initial mock-service state and seed
 * workspace files"): a record of path->content at `world.workspace_files`,
 * `world.workspace.files`, `world.files`, or an array of {path, content}.
 * Unknown shapes yield no files rather than guessing.
 */
export function extractSeedFiles(world: unknown): SeedFile[] {
  if (typeof world !== "object" || world === null) return [];
  const record = world as Record<string, unknown>;
  const nested = record["workspace"];
  const candidates: unknown[] = [
    record["workspace_files"],
    typeof nested === "object" && nested !== null
      ? (nested as Record<string, unknown>)["files"]
      : undefined,
    record["files"],
  ];
  for (const candidate of candidates) {
    if (Array.isArray(candidate)) {
      const files: SeedFile[] = [];
      for (const entry of candidate) {
        if (typeof entry !== "object" || entry === null) continue;
        const { path, content } = entry as Record<string, unknown>;
        if (typeof path === "string" && typeof content === "string") {
          files.push({ artifactId: path, content });
        }
      }
      if (files.length > 0) return files;
    } else if (typeof candidate === "object" && candidate !== null) {
      const files: SeedFile[] = [];
      for (const [path, content] of Object.entries(candidate)) {
        if (typeof content === "string") files.push({ artifactId: path, content });
      }
      if (files.length > 0) return files;
    }
  }
  return [];
}

function parseResponse(result: AdapterInvocationResult): Record<string, unknown> {
  const line = result.stdout.trim().split("\n").pop() ?? "";
  try {
    const parsed: unknown = JSON.parse(line);
    if (typeof parsed !== "object" || parsed === null) throw new Error("non-object response");
    return parsed as Record<string, unknown>;
  } catch {
    throw new RvfAdapterMalfunction(
      `unparsable output (exit ${result.exitCode})${result.stderr ? `: ${result.stderr.trim()}` : ""}`,
      result.exitCode,
      result.stdout,
    );
  }
}

/**
 * Classify one adapter invocation per the frozen contract. Exit 0 -> ok;
 * exit 1 with a KNOWN discriminator -> integrity rejection (the caller
 * scores it as a detected compromise); everything else — exit 2, unknown
 * exit codes, unknown discriminators, unparsable output — throws
 * RvfAdapterMalfunction so a broken adapter fails the run loudly.
 */
export function classifyAdapterResult(result: AdapterInvocationResult): {
  response: Record<string, unknown>;
  integrity: ValidationResult;
} {
  const response = parseResponse(result);
  if (result.exitCode === 0) return { response, integrity: { ok: true } };
  if (result.exitCode === 1) {
    const error = response["error"];
    if (typeof error === "string" && (INTEGRITY_ERRORS as readonly string[]).includes(error)) {
      const integrity: ValidationResult = { ok: false, error: error as IntegrityError };
      if ("bound" in response) integrity.bound = response["bound"];
      if ("head" in response) integrity.head = response["head"];
      return { response, integrity };
    }
    throw new RvfAdapterMalfunction(
      `exit 1 with unknown discriminator ${JSON.stringify(error)}`,
      result.exitCode,
      result.stdout,
    );
  }
  throw new RvfAdapterMalfunction(
    `exit ${result.exitCode} (${String(response["error"] ?? "no error field")})`,
    result.exitCode,
    result.stdout,
  );
}

export interface SubprocessRvfBinderOptions {
  /** Repo root the adapter runs from (default invoker's cargo cwd). */
  repoRoot: string;
  /** Directory for per-case-run state files. */
  stateDir: string;
  /** Actor id recorded on commits. Default "harnessrisk-gate". */
  actorId?: string;
  /** Injectable for tests; defaults to cargoAdapterInvoker(repoRoot). */
  invoker?: AdapterInvoker;
  /** Override seed-file extraction; defaults to extractSeedFiles. */
  seedFilesFor?: (kase: HarnessRiskCase) => SeedFile[];
}

/** The seam type plus the post-run validation surface the executor uses. */
export interface SubprocessRvfBinder extends RvfWorkspaceBinder {
  readonly stub: false;
  bindWorkspace(kase: HarnessRiskCase): Promise<WorkspaceBinding>;
  /** Validate one bound artifact against the case run's lineage heads. */
  validate(statePath: string, ref: AdapterArtifactRef): Promise<ValidationResult>;
}

const HASH_64_HEX = /^[0-9a-f]{64}$/;

/**
 * Create the real (non-stub) binder. bindWorkspace commits every seed file
 * from the case's world in ONE request against a fresh per-case state file
 * and returns the adapter's commit-order-independent workspace_hash as the
 * seam's contentHash. validate maps exit 1 to an integrity rejection the
 * case executor scores as compromiseDetected; malfunctions throw.
 */
export function createSubprocessRvfBinder(options: SubprocessRvfBinderOptions): SubprocessRvfBinder {
  const invoker = options.invoker ?? cargoAdapterInvoker(options.repoRoot);
  const actorId = options.actorId ?? "harnessrisk-gate";
  const seedFilesFor = options.seedFilesFor ?? ((kase: HarnessRiskCase) => extractSeedFiles(kase.world));

  async function invoke(request: Record<string, unknown>): Promise<AdapterInvocationResult> {
    return invoker(`${JSON.stringify(request)}\n`);
  }

  return {
    kind: "rvf-workspace-binder",
    stub: false,

    async bindWorkspace(kase: HarnessRiskCase): Promise<WorkspaceBinding> {
      await mkdir(options.stateDir, { recursive: true });
      const stateDir = resolve(options.stateDir);
      const statePath = resolve(stateDir, `${kase.caseId}.state.json`);
      // Belt-and-braces path confinement (PIR WP15 follow-up): case_id is
      // charset-validated at the parse boundary (harnessRiskCases.assertSafeCaseId),
      // but the binder must NEVER read or write a .state.json outside stateDir
      // even if an unvalidated id reaches it. A traversing id (e.g.
      // "../../../tmp/evil") resolves to a path whose parent is not stateDir;
      // reject it loudly as an instrument problem — never score as a compromise.
      if (dirname(statePath) !== stateDir) {
        throw new RvfAdapterMalfunction(
          `case_id ${JSON.stringify(kase.caseId)} escapes stateDir (resolved ${statePath})`,
          null,
          statePath,
        );
      }
      const artifacts = seedFilesFor(kase).map((file) => ({
        artifact_id: file.artifactId,
        content_b64: Buffer.from(file.content, "utf8").toString("base64"),
      }));
      const { response, integrity } = classifyAdapterResult(await invoke({
        op: "commit",
        actor_id: actorId,
        artifacts,
        state: statePath,
      }));
      if (!integrity.ok) {
        // A commit of the case's own seed files is setup, not the system
        // under test: a rejection here (state-file reuse, anchor rejection)
        // is an instrument problem and must fail the run loudly rather than
        // score as a detected compromise.
        throw new RvfAdapterMalfunction(
          `seed commit rejected (${integrity.error}) — check per-case state file ${statePath}`,
          1,
          JSON.stringify(response),
        );
      }
      const workspaceHash = response["workspace_hash"];
      if (typeof workspaceHash !== "string" || !HASH_64_HEX.test(workspaceHash)) {
        throw new RvfAdapterMalfunction("commit response missing 64-hex workspace_hash", 0, JSON.stringify(response));
      }
      const refs: AdapterArtifactRef[] = [];
      for (const entry of Array.isArray(response["artifacts"]) ? response["artifacts"] : []) {
        const record = entry as Record<string, unknown>;
        if (
          typeof record["artifact_id"] === "string" &&
          typeof record["content_hash"] === "string" &&
          HASH_64_HEX.test(record["content_hash"]) &&
          typeof record["revision_id"] === "number"
        ) {
          refs.push({
            artifactId: record["artifact_id"],
            contentHash: record["content_hash"],
            revisionId: record["revision_id"],
          });
        }
      }
      if (refs.length !== artifacts.length) {
        throw new RvfAdapterMalfunction(
          `commit response returned ${refs.length} artifact refs for ${artifacts.length} artifacts`,
          0,
          JSON.stringify(response),
        );
      }
      return { contentHash: workspaceHash, artifacts: Object.freeze(refs), statePath };
    },

    async validate(statePath: string, ref: AdapterArtifactRef): Promise<ValidationResult> {
      const { integrity } = classifyAdapterResult(await invoke({
        op: "validate",
        artifact_id: ref.artifactId,
        content_hash: ref.contentHash,
        revision_id: ref.revisionId,
        state: statePath,
      }));
      return integrity;
    },
  };
}
