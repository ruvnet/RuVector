/**
 * Download build artifacts and verify them on arrival (requirements §5.8).
 *
 * Every downloaded byte is hashed and compared against the digest the service
 * advertised before the file is kept. A mismatch removes the partial file and
 * fails — a downloaded artifact is either verified or absent, never both.
 */

import { mkdir, rm, writeFile } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';

import { ForgeApiClient, type BuildArtifact, type ForgeApiOptions } from './api';
import { ForgeError, ForgeErrorCode } from './errors';
import { sha256Buffer } from './hash';
import { assertBuildId } from './status';

export interface DownloadOptions {
  outDir?: string;
  /** Only fetch artifacts whose name matches one of these (exact match). */
  only?: readonly string[];
  api?: ForgeApiOptions;
}

export interface DownloadedArtifact {
  name: string;
  path: string;
  sizeBytes: number;
  sha256: string;
  verified: true;
}

export interface DownloadResult {
  buildId: string;
  outDir: string;
  artifacts: DownloadedArtifact[];
  skipped: string[];
}

/**
 * @throws {ForgeError} `FORGE_E_NOT_FOUND` for an unknown build or artifact,
 * `FORGE_E_NETWORK` when the service is unreachable, `FORGE_E_VERIFY_FAILED`
 * when a downloaded digest does not match.
 */
export async function runDownload(id: string, options: DownloadOptions = {}): Promise<DownloadResult> {
  const buildId = assertBuildId(id);
  const client = new ForgeApiClient(options.api);
  const outDir = resolve(options.outDir ?? join('forge-artifacts', buildId));

  const { artifacts } = await client.listArtifacts(buildId);
  const wanted = selectArtifacts(artifacts, options.only);
  if (wanted.length === 0) {
    throw new ForgeError(ForgeErrorCode.NOT_FOUND, `Build ${buildId} has no matching artifacts.`, {
      buildId,
      requested: options.only ?? [],
    });
  }

  await mkdir(outDir, { recursive: true });

  const downloaded: DownloadedArtifact[] = [];
  for (const artifact of wanted) {
    downloaded.push(await downloadOne(client, artifact, outDir, buildId));
  }

  return {
    buildId,
    outDir,
    artifacts: downloaded,
    skipped: artifacts.filter((a) => !wanted.includes(a)).map((a) => a.name),
  };
}

async function downloadOne(
  client: ForgeApiClient,
  artifact: BuildArtifact,
  outDir: string,
  buildId: string,
): Promise<DownloadedArtifact> {
  const body = await client.fetchArtifact(artifact.url);
  const actual = sha256Buffer(body);

  if (actual !== artifact.sha256) {
    throw new ForgeError(
      ForgeErrorCode.VERIFY_FAILED,
      `Downloaded ${artifact.name} does not match the digest the service advertised; the file was discarded.`,
      { buildId, artifact: artifact.name, expected: artifact.sha256, actual },
    );
  }

  // `basename` keeps a service-supplied name from escaping the output
  // directory via `../` or an absolute path.
  const path = join(outDir, basename(artifact.name));
  try {
    await writeFile(path, body);
  } catch (err) {
    await rm(path, { force: true });
    throw new ForgeError(ForgeErrorCode.IO, `Cannot write ${path}: ${(err as Error).message}`, { path });
  }

  return { name: artifact.name, path, sizeBytes: body.byteLength, sha256: actual, verified: true };
}

function selectArtifacts(
  artifacts: readonly BuildArtifact[],
  only?: readonly string[],
): BuildArtifact[] {
  if (!only || only.length === 0) return [...artifacts];
  const wanted = new Set(only);
  return artifacts.filter((a) => wanted.has(a.name));
}
