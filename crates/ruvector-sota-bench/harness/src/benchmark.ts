import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  mkdir, mkdtemp, readFile, rename, rm, stat, writeFile,
} from "node:fs/promises";
import { hostname, platform, release } from "node:os";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { parseBenchReport, type BenchReport } from "./metrics.js";

export type RunnerSet = "core" | "core-lsm" | "all";

export interface BenchmarkPolicy extends Record<string, string> {
  ef_search: string;
  m: string;
  ef_construction: string;
  k: string;
  runner_set: RunnerSet;
}

export interface BenchmarkSuiteItem {
  seed: number;
  dataset: string;
  dataset_sha256: string;
  kind: "smoke" | "synthetic" | "real";
  dataset_file?: string;
}

export interface ResourceObservation {
  processPeakRssBytes: number;
  rawProcessPeakRssBytes: number;
  baselineRssBytes: number;
  sampleIntervalMs: number;
  samples: number;
  wallTimeMs: number;
  cacheHit: boolean;
}

export interface BenchmarkObservation {
  report: BenchReport;
  item: BenchmarkSuiteItem;
  policy: BenchmarkPolicy;
  resources: ResourceObservation;
  cacheKey: string;
}

export interface BenchmarkRunOptions {
  repoRoot: string;
  policy: Record<string, string>;
  item?: BenchmarkSuiteItem;
  timeoutMs?: number;
  binary?: string;
  cacheDir?: string;
  embeddingSpaceId?: string;
  extraArgs?: readonly string[];
  /** Trusted argv inserted before benchmark flags (primarily test wrappers). */
  commandPrefixArgs?: readonly string[];
  useCache?: boolean;
  maxRssBytes?: number;
  maxOutputBytes?: number;
  /** Internal native sweep used by runBenchmarkBatch. */
  efSearchSweep?: readonly string[];
}

const SHA256 = /^[0-9a-f]{64}$/;
const DEFAULT_ITEM: BenchmarkSuiteItem = {
  seed: 11,
  dataset: "smoke-128",
  dataset_sha256: createHash("sha256").update("ruvector:smoke-128:v1").digest("hex"),
  kind: "smoke",
};

function boundedInteger(policy: Record<string, string>, key: string, min: number, max: number): string {
  const raw = policy[key];
  if (!raw || !/^[0-9]{1,6}$/.test(raw)) throw new Error(`${key} must be an integer string`);
  const value = Number(raw);
  if (value < min || value > max) throw new Error(`${key} must be in [${min}, ${max}]`);
  return String(value);
}

export function normalizePolicy(policy: Record<string, string>): BenchmarkPolicy {
  const runnerSet = policy.runner_set ?? "core";
  if (runnerSet !== "core" && runnerSet !== "core-lsm" && runnerSet !== "all") {
    throw new Error("runner_set must be core, core-lsm, or all");
  }
  const allowed = new Set(["ef_search", "m", "ef_construction", "k", "runner_set"]);
  const unknown = Object.keys(policy).filter((key) => !allowed.has(key));
  if (unknown.length) throw new Error(`unsupported policy lever(s): ${unknown.join(", ")}`);
  return {
    ef_search: boundedInteger(policy, "ef_search", 1, 4096),
    m: boundedInteger({ m: policy.m ?? "32" }, "m", 2, 128),
    ef_construction: boundedInteger(
      { ef_construction: policy.ef_construction ?? "200" },
      "ef_construction",
      4,
      4096,
    ),
    k: boundedInteger({ k: policy.k ?? "10" }, "k", 1, 100),
    runner_set: runnerSet,
  };
}

export function normalizeSuiteItem(value: unknown): BenchmarkSuiteItem {
  if (!value || typeof value !== "object") throw new Error("suite item must be an object");
  const item = value as Partial<BenchmarkSuiteItem>;
  if (!Number.isSafeInteger(item.seed) || (item.seed ?? -1) < 0) throw new Error("suite seed must be non-negative");
  if (!item.dataset || !/^[a-z0-9][a-z0-9-]{1,63}$/.test(item.dataset)) {
    throw new Error("invalid suite dataset");
  }
  if (!item.dataset_sha256 || !SHA256.test(item.dataset_sha256)) {
    throw new Error("suite item requires a lowercase SHA-256 dataset identity");
  }
  if (item.kind !== "smoke" && item.kind !== "synthetic" && item.kind !== "real") {
    throw new Error("suite item kind must be smoke, synthetic, or real");
  }
  if (item.kind === "real" && (!item.dataset_file || !item.dataset_file.startsWith("/"))) {
    throw new Error("real suite items require an absolute dataset_file");
  }
  return item as BenchmarkSuiteItem;
}

async function sha256File(file: string): Promise<string> {
  return createHash("sha256").update(await readFile(file)).digest("hex");
}

function gitCommit(repoRoot: string): string {
  const result = spawnSync("git", ["rev-parse", "HEAD"], {
    cwd: repoRoot, encoding: "utf8", env: { PATH: process.env.PATH ?? "" },
  });
  return result.status === 0 ? result.stdout.trim() : "unknown";
}

async function environmentFingerprint(binary: string): Promise<Record<string, unknown>> {
  const binaryStat = await stat(binary);
  return {
    platform: platform(),
    release: release(),
    arch: process.arch,
    node: process.version,
    hostname: hostname(),
    binary_size: binaryStat.size,
    binary_mtime_ms: Math.trunc(binaryStat.mtimeMs),
  };
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

async function readPeakRssBytes(pid: number): Promise<number> {
  try {
    if (platform() === "linux") {
      const status = await readFile(`/proc/${pid}/status`, "utf8");
      const match = /^VmRSS:\s+(\d+)\s+kB$/m.exec(status);
      return match ? Number(match[1]) * 1024 : 0;
    }
    const result = spawnSync("ps", ["-o", "rss=", "-p", String(pid)], {
      encoding: "utf8", env: { PATH: process.env.PATH ?? "" },
    });
    return result.status === 0 ? Number(result.stdout.trim()) * 1024 : 0;
  } catch {
    return 0;
  }
}

function runnerArgs(policy: BenchmarkPolicy, efSearchSweep?: readonly string[]): string[] {
  const args = [
    "--ef-search", efSearchSweep?.join(",") ?? policy.ef_search,
    "--m", policy.m,
    "--ef-construction", policy.ef_construction,
    "--k", policy.k,
  ];
  if (policy.runner_set === "core") {
    args.push("--no-hybrid", "--no-matryoshka", "--no-lsm", "--no-rabitq");
  } else if (policy.runner_set === "core-lsm") {
    args.push("--no-hybrid", "--no-matryoshka", "--no-rabitq");
  }
  return args;
}

async function writeAtomic(file: string, content: string): Promise<void> {
  await mkdir(dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  await writeFile(temporary, content, { mode: 0o600 });
  await rename(temporary, file);
}

export async function runObservedBenchmark(options: BenchmarkRunOptions): Promise<BenchmarkObservation> {
  const policy = normalizePolicy(options.policy);
  const item = normalizeSuiteItem(options.item ?? DEFAULT_ITEM);
  if (item.kind === "real" && await sha256File(item.dataset_file!) !== item.dataset_sha256) {
    throw new Error(`real dataset hash mismatch: ${item.dataset}`);
  }
  const binary = options.binary ?? resolve(options.repoRoot, "target/release/sota-all");
  const fingerprint = await environmentFingerprint(binary);
  const cacheKey = createHash("sha256").update(canonical({
    schema: 1,
    commit: gitCommit(options.repoRoot),
    policy,
    item,
    embedding_space_id: options.embeddingSpaceId ?? "not-applicable",
    environment: fingerprint,
    threads: process.env.RUVECTOR_BENCH_THREADS ?? "1",
    max_rss_bytes: options.maxRssBytes ?? 8 * 1024 * 1024 * 1024,
    extra_args: options.extraArgs ?? [],
    command_prefix_args: options.commandPrefixArgs ?? [],
    ef_search_sweep: options.efSearchSweep ?? [],
  })).digest("hex");
  const cacheDir = resolve(options.cacheDir ?? resolve(options.repoRoot, ".metaharness/cache"));
  const cacheFile = resolve(cacheDir, `${cacheKey}.json`);
  if (options.useCache !== false) {
    try {
      const cached = JSON.parse(await readFile(cacheFile, "utf8")) as BenchmarkObservation;
      cached.report = parseBenchReport(cached.report);
      cached.resources.cacheHit = true;
      return cached;
    } catch {
      // Cache misses and invalid entries both execute a fresh isolated run.
    }
  }

  const directory = await mkdtemp(join(tmpdir(), "ruvector-metaharness-"));
  const reportPath = join(directory, "report.json");
  const args = [
    ...(options.commandPrefixArgs ?? []),
    ...(item.kind === "smoke" ? ["--smoke"] : []),
    ...(item.kind === "real" ? ["--real", "--dataset-cache", dirname(item.dataset_file!)] : []),
    "--seed", String(item.seed),
    "--dataset", item.dataset,
    "--json", reportPath,
    "--measurement-start-delay-ms", "100",
    ...runnerArgs(policy, options.efSearchSweep),
    ...(options.extraArgs ?? []),
  ];
  const started = performance.now();
  let peakRss = 0;
  let baselineRss = Number.POSITIVE_INFINITY;
  let samples = 0;

  try {
    await new Promise<void>((resolveRun, reject) => {
      const child = spawn(binary, args, {
        cwd: options.repoRoot,
        stdio: ["ignore", "pipe", "pipe"],
        env: {
          PATH: process.env.PATH ?? "",
          RAYON_NUM_THREADS: process.env.RUVECTOR_BENCH_THREADS ?? "1",
          OMP_NUM_THREADS: process.env.RUVECTOR_BENCH_THREADS ?? "1",
        },
      });
      let stderr = "";
      let outputBytes = 0;
      let limitFailure = "";
      const consume = (chunk: Buffer, capture: boolean) => {
        outputBytes += chunk.length;
        if (capture) stderr = `${stderr}${chunk.toString("utf8")}`.slice(-2_000);
        if (outputBytes > (options.maxOutputBytes ?? 10 * 1024 * 1024)) {
          limitFailure = "benchmark output limit exceeded";
          child.kill("SIGKILL");
        }
      };
      child.stdout.on("data", (chunk: Buffer) => consume(chunk, false));
      child.stderr.on("data", (chunk: Buffer) => consume(chunk, true));
      const sampler = setInterval(() => {
        void readPeakRssBytes(child.pid ?? 0).then((rss) => {
          peakRss = Math.max(peakRss, rss);
          if (rss > (options.maxRssBytes ?? 8 * 1024 * 1024 * 1024)) {
            limitFailure = "benchmark RSS limit exceeded";
            child.kill("SIGKILL");
          }
          const elapsed = performance.now() - started;
          // The native binary deliberately idles for 100ms. Ignore the
          // pre-exec samples and measure its loaded, pre-allocation baseline.
          if (rss > 0 && elapsed >= 40 && elapsed <= 100) {
            baselineRss = Math.min(baselineRss, rss);
          }
          samples += 1;
        });
      }, 10);
      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error(`benchmark exceeded ${options.timeoutMs ?? 300_000}ms`));
      }, options.timeoutMs ?? 300_000);
      const done = () => {
        clearInterval(sampler);
        clearTimeout(timer);
      };
      child.once("error", (error) => {
        done();
        reject(error);
      });
      child.once("exit", (code) => {
        done();
        code === 0 && !limitFailure
          ? resolveRun()
          : reject(new Error(limitFailure || `benchmark exited ${code}: ${stderr}`));
      });
    });
    const report = parseBenchReport(JSON.parse(await readFile(reportPath, "utf8")));
    const observation: BenchmarkObservation = {
      report,
      item,
      policy,
      resources: {
        processPeakRssBytes: Math.max(0, peakRss - (Number.isFinite(baselineRss) ? baselineRss : 0)),
        rawProcessPeakRssBytes: peakRss,
        baselineRssBytes: Number.isFinite(baselineRss) ? baselineRss : 0,
        sampleIntervalMs: 10,
        samples,
        wallTimeMs: performance.now() - started,
        cacheHit: false,
      },
      cacheKey,
    };
    if (options.useCache !== false) await writeAtomic(cacheFile, `${JSON.stringify(observation)}\n`);
    return observation;
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

export async function runBenchmark(options: BenchmarkRunOptions): Promise<BenchReport> {
  return (await runObservedBenchmark(options)).report;
}

/**
 * Batch policies that differ only in ef_search into one native invocation.
 * This is intended for sweeps outside Flywheel's sequential proposer contract.
 */
export async function runBenchmarkBatch(
  options: Omit<BenchmarkRunOptions, "policy"> & { policies: Record<string, string>[] },
): Promise<Map<string, BenchReport>> {
  if (options.policies.length === 0) return new Map();
  const normalized = options.policies.map(normalizePolicy);
  const first = normalized[0]!;
  for (const policy of normalized) {
    if (policy.m !== first.m || policy.ef_construction !== first.ef_construction ||
        policy.k !== first.k || policy.runner_set !== first.runner_set) {
      throw new Error("batch policies may differ only in ef_search");
    }
  }
  const efValues = [...new Set(normalized.map((policy) => policy.ef_search))];
  const observation = await runObservedBenchmark({
    ...options,
    policy: first,
    efSearchSweep: efValues,
    useCache: false,
  });
  const result = new Map<string, BenchReport>();
  for (const efSearch of efValues) {
    const scores = observation.report.scores.filter((score) => score.params?.ef_search === efSearch);
    result.set(efSearch, { ...observation.report, scores });
  }
  return result;
}
