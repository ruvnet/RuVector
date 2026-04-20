/*
 * Obsidian E2E runner — downloads the real Obsidian AppImage (cached at
 * $HOME/.cache/obsidian-brain-e2e), provisions a disposable vault with the
 * main plugin + harness plugin enabled, spins up the real brain + a mock
 * embedder, launches Obsidian under `xvfb-run` (when $DISPLAY is unset),
 * and waits for the harness to write a JSON report. Quits cleanly.
 *
 * No mocks on the plugin side — the harness runs inside the real Obsidian
 * runtime and exercises the plugin through its actual command/view API.
 */

import { spawn, spawnSync, ChildProcess } from "node:child_process";
import {
	existsSync,
	mkdirSync,
	readFileSync,
	rmSync,
	writeFileSync,
	createWriteStream,
	copyFileSync,
	readdirSync,
	chmodSync,
} from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { pipeline } from "node:stream/promises";
import { createServer, Server } from "node:http";
import { AddressInfo } from "node:net";
import { get as httpsGet } from "node:https";

export interface E2EHarnessOptions {
	obsidianVersion?: string;
	obsidianAppImageUrl?: string;
	pluginDir: string; // absolute path to examples/obsidian-brain
	keepVault?: boolean;
}

export interface E2EReport {
	version: string;
	startedAt: string;
	finishedAt: string;
	checks: Array<{ name: string; ok: boolean; detail?: string }>;
	passed: number;
	failed: number;
}

export interface E2EResult {
	report: E2EReport;
	vaultDir: string;
}

const DEFAULT_VERSION = "1.6.5";
function defaultAppImageUrl(version: string): string {
	return `https://github.com/obsidianmd/obsidian-releases/releases/download/v${version}/Obsidian-${version}.AppImage`;
}

function cacheDir(): string {
	const dir = join(homedir(), ".cache", "obsidian-brain-e2e");
	mkdirSync(dir, { recursive: true });
	return dir;
}

async function downloadIfMissing(url: string, dest: string): Promise<void> {
	if (existsSync(dest)) return;
	const tmpDest = dest + ".part";
	await new Promise<void>((resolvePromise, rejectPromise) => {
		const handle = (incomingUrl: string, redirectsLeft: number): void => {
			httpsGet(incomingUrl, async (res) => {
				if (
					res.statusCode &&
					res.statusCode >= 300 &&
					res.statusCode < 400 &&
					res.headers.location
				) {
					if (redirectsLeft <= 0)
						return rejectPromise(new Error("too many redirects"));
					res.resume();
					return handle(
						new URL(res.headers.location, incomingUrl).toString(),
						redirectsLeft - 1,
					);
				}
				if (res.statusCode !== 200)
					return rejectPromise(new Error(`download ${res.statusCode}`));
				try {
					await pipeline(res, createWriteStream(tmpDest));
					resolvePromise();
				} catch (e) {
					rejectPromise(e as Error);
				}
			}).on("error", (e) => rejectPromise(e));
		};
		handle(url, 8);
	});
	chmodSync(tmpDest, 0o755);
	copyFileSync(tmpDest, dest);
	rmSync(tmpDest);
}

async function ensureObsidianAppImage(opts: E2EHarnessOptions): Promise<string> {
	const version = opts.obsidianVersion ?? DEFAULT_VERSION;
	const url = opts.obsidianAppImageUrl ?? defaultAppImageUrl(version);
	const dest = join(cacheDir(), `Obsidian-${version}.AppImage`);
	await downloadIfMissing(url, dest);
	// Extract once so we don't need FUSE at runtime — works in locked-down
	// environments (CI, containers) and avoids libfuse2 ABI drift. We return
	// the raw electron binary rather than AppRun because AppRun's APPDIR
	// auto-detection misbehaves when the first argument is an absolute path
	// outside the extraction tree.
	const extracted = join(cacheDir(), `Obsidian-${version}-extracted`);
	const electronBin = join(extracted, "squashfs-root", "obsidian");
	if (!existsSync(electronBin)) {
		mkdirSync(extracted, { recursive: true });
		const r = spawnSync(dest, ["--appimage-extract"], {
			cwd: extracted,
			stdio: "pipe",
			encoding: "utf8",
		});
		if (r.status !== 0) {
			throw new Error(
				`AppImage extraction failed (exit ${r.status}): ${r.stderr}\n${r.stdout}`,
			);
		}
	}
	return electronBin;
}

function buildHarness(pluginDir: string): string {
	const harnessDir = join(pluginDir, "tests", "e2e", "harness");
	spawnSync(
		"node",
		[join(pluginDir, "node_modules", ".bin", "esbuild"), "--version"],
		{ stdio: "ignore" },
	);
	const r = spawnSync("node", ["esbuild.config.mjs"], {
		cwd: harnessDir,
		stdio: "pipe",
		encoding: "utf8",
	});
	if (r.status !== 0)
		throw new Error(`harness build failed: ${r.stderr}\n${r.stdout}`);
	return harnessDir;
}

function buildMainPlugin(pluginDir: string): void {
	const r = spawnSync("npm", ["run", "build"], {
		cwd: pluginDir,
		stdio: "pipe",
		encoding: "utf8",
	});
	if (r.status !== 0)
		throw new Error(`main plugin build failed: ${r.stderr}\n${r.stdout}`);
}

function createVault(
	pluginDir: string,
	harnessDir: string,
	reportPath: string,
): string {
	const vault = join(tmpdir(), `obsidian-brain-vault-${Date.now()}`);
	const pluginsDir = join(vault, ".obsidian", "plugins");
	const brainPlugin = join(pluginsDir, "obsidian-brain");
	const harnessPlugin = join(pluginsDir, "obsidian-brain-e2e-harness");
	mkdirSync(brainPlugin, { recursive: true });
	mkdirSync(harnessPlugin, { recursive: true });
	for (const f of ["main.js", "manifest.json", "styles.css", "versions.json"]) {
		if (existsSync(join(pluginDir, f))) {
			copyFileSync(join(pluginDir, f), join(brainPlugin, f));
		}
	}
	for (const f of ["main.js", "manifest.json"]) {
		copyFileSync(join(harnessDir, f), join(harnessPlugin, f));
	}
	// Enable both plugins by default.
	writeFileSync(
		join(vault, ".obsidian", "community-plugins.json"),
		JSON.stringify(["obsidian-brain", "obsidian-brain-e2e-harness"], null, 2),
	);
	writeFileSync(
		join(vault, ".obsidian", "app.json"),
		JSON.stringify(
			{ promptDelete: false, alwaysUpdateLinks: false },
			null,
			2,
		),
	);
	// Seed a welcome note so the vault isn't empty.
	writeFileSync(
		join(vault, "Welcome.md"),
		"---\nbrain-category: seed\n---\nWelcome to the E2E vault.\n",
	);
	// Tell the harness where to write its report.
	process.env.OBSIDIAN_BRAIN_E2E_REPORT = reportPath;
	void pluginDir;
	return vault;
}

async function startEmbedder(): Promise<{ url: string; server: Server }> {
	const server = createServer((req, res) => {
		if (!req.url || req.method !== "POST" || !req.url.startsWith("/embed")) {
			res.writeHead(404).end();
			return;
		}
		const chunks: Buffer[] = [];
		req.on("data", (c) => chunks.push(c));
		req.on("end", () => {
			const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
				texts: string[];
			};
			const DIM = 16;
			const vectors = body.texts.map((t) => {
				const v = new Array<number>(DIM).fill(0);
				for (let i = 0; i < t.length; i++)
					v[i % DIM] += t.charCodeAt(i) / 256;
				const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
				return v.map((x) => x / norm);
			});
			res.writeHead(200, { "content-type": "application/json" });
			res.end(JSON.stringify({ vectors }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const addr = server.address() as AddressInfo;
	return { url: `http://127.0.0.1:${addr.port}`, server };
}

async function pickPort(): Promise<number> {
	const s = createServer();
	await new Promise<void>((r) => s.listen(0, "127.0.0.1", r));
	const port = (s.address() as AddressInfo).port;
	await new Promise<void>((r) => s.close(() => r()));
	return port;
}

async function waitForHealth(url: string, timeoutMs: number): Promise<void> {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		try {
			const r = await fetch(url + "/health");
			if (r.ok) return;
		} catch {
			/* retry */
		}
		await sleep(100);
	}
	throw new Error("brain did not come up");
}

function resolveBrainBinary(pluginDir: string): string {
	const envBin = process.env.RUVBRAIN_BIN;
	if (envBin && existsSync(envBin)) return envBin;
	const fromRepo = resolve(
		pluginDir,
		"..",
		"..",
		"target",
		"release",
		"mcp-brain-server-local",
	);
	if (existsSync(fromRepo)) return fromRepo;
	throw new Error(
		`mcp-brain-server-local not found at ${fromRepo}. ` +
			`Build with: cargo build --release -p mcp-brain-server --features local --bin mcp-brain-server-local`,
	);
}

export async function runObsidianE2E(opts: E2EHarnessOptions): Promise<E2EResult> {
	const pluginDir = resolve(opts.pluginDir);
	buildMainPlugin(pluginDir);
	const harnessDir = buildHarness(pluginDir);

	const reportPath = join(tmpdir(), `obsidian-brain-report-${Date.now()}.json`);
	const vault = createVault(pluginDir, harnessDir, reportPath);

	const embedder = await startEmbedder();
	const brainPort = await pickPort();
	const brainUrl = `http://127.0.0.1:${brainPort}`;
	const brainBin = resolveBrainBinary(pluginDir);
	const brainDataDir = join(vault, ".brain-data");
	mkdirSync(brainDataDir, { recursive: true });

	const brain = spawn(brainBin, [], {
		env: {
			...process.env,
			RUVBRAIN_PORT: String(brainPort),
			RUVBRAIN_DB: join(brainDataDir, "brain.sqlite"),
			RUVBRAIN_BLOBS: join(brainDataDir, "blobs"),
			RUVBRAIN_STORE: "rvf",
			RUVBRAIN_EMBEDDER_URL: embedder.url,
			RUST_LOG: process.env.RUST_LOG ?? "warn",
		},
		stdio: ["ignore", "pipe", "pipe"],
	});
	brain.stderr?.on("data", (d: Buffer) =>
		process.stderr.write(`[brain] ${d.toString()}`),
	);
	await waitForHealth(brainUrl, 20_000);

	// Patch the plugin's data.json so the Brain URL matches our scratch server.
	// Pi.ruv.io integration is only wired when BRAIN_API_KEY is set in env so
	// offline runs remain deterministic.
	const piToken = process.env.BRAIN_API_KEY ?? "";
	writeFileSync(
		join(vault, ".obsidian", "plugins", "obsidian-brain", "data.json"),
		JSON.stringify(
			{
				settings: {
					brainUrl,
					embedderUrl: embedder.url,
					defaultCategory: "e2e",
					autoIndex: false,
					autoIndexDebounceMs: 1000,
					indexMinChars: 5,
					enableAIDefence: true,
					searchLimit: 8,
					relatedLimit: 8,
					bulkSyncBatchSize: 8,
					bulkSyncIncludeFolders: "",
					bulkSyncExcludeFolders: ".obsidian,.trash",
					storeMapping: {},
					dpoDirection: "quality",
					piUrl: process.env.PI_URL ?? "https://pi.ruv.io",
					piToken,
					piPullLimit: 5,
					piPullQuery: "",
				},
				indexState: { pathToHash: {}, hashToId: {}, idToPath: {}, lastSync: 0 },
			},
			null,
			2,
		),
	);

	const appImage = await ensureObsidianAppImage(opts);
	// Isolated HOME so we don't touch the developer's Obsidian vault registry.
	const fakeHome = join(vault, ".scratch-home");
	mkdirSync(join(fakeHome, ".config", "obsidian"), { recursive: true });
	const vaultId = Math.random().toString(16).slice(2, 10) + "e2e";
	writeFileSync(
		join(fakeHome, ".config", "obsidian", "obsidian.json"),
		JSON.stringify(
			{
				vaults: {
					[vaultId]: { path: vault, ts: Date.now(), open: true },
				},
				showInlineTitle: false,
				insider: false,
			},
			null,
			2,
		),
	);

	const needsXvfb = !process.env.DISPLAY;
	const cmd = needsXvfb ? "xvfb-run" : appImage;
	const args = needsXvfb
		? ["-a", "--server-args=-screen 0 1200x900x24", appImage, "--no-sandbox"]
		: ["--no-sandbox"];

	const obsidian: ChildProcess = spawn(cmd, args, {
		stdio: ["ignore", "pipe", "pipe"],
		env: {
			...process.env,
			HOME: fakeHome,
			XDG_CONFIG_HOME: join(fakeHome, ".config"),
			XDG_DATA_HOME: join(fakeHome, ".local", "share"),
			XDG_CACHE_HOME: join(fakeHome, ".cache"),
			OBSIDIAN_BRAIN_E2E_REPORT: reportPath,
		},
	});
	obsidian.stderr?.on("data", (d: Buffer) =>
		process.stderr.write(`[obsidian] ${d.toString()}`),
	);
	obsidian.stdout?.on("data", (d: Buffer) =>
		process.stdout.write(`[obsidian] ${d.toString()}`),
	);

	try {
		const report = await waitForReport(reportPath, 120_000);
		return { report, vaultDir: vault };
	} finally {
		try {
			obsidian.kill("SIGTERM");
		} catch {
			/* ignore */
		}
		try {
			brain.kill("SIGTERM");
		} catch {
			/* ignore */
		}
		await new Promise<void>((r) => embedder.server.close(() => r()));
		if (!opts.keepVault) {
			try {
				rmSync(vault, { recursive: true, force: true });
			} catch {
				/* ignore */
			}
		}
	}
}

async function waitForReport(path: string, timeoutMs: number): Promise<E2EReport> {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		if (existsSync(path)) {
			try {
				const raw = readFileSync(path, "utf8");
				if (raw.trim()) return JSON.parse(raw) as E2EReport;
			} catch {
				/* partial write — retry */
			}
		}
		await sleep(500);
	}
	throw new Error(`no report written to ${path} within ${timeoutMs}ms`);
}

// helper only referenced by tests below
void readdirSync;
