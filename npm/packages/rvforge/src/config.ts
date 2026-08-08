/**
 * `forge.config.json` — the project file `forge init` scaffolds and the other
 * commands read. It holds app metadata, targets, packaging, runtime, the
 * capability policy, and signing *references*.
 *
 * It never holds credentials. Signing key material lives in an HSM, KMS, or
 * the platform keychain; the config records only which key to ask for.
 */

import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

import { ForgeError, ForgeErrorCode } from './errors';
import { generatePublisherKey, writePublisherKey } from './keys';
import { defaultCapabilityPolicy, normalizeCapabilityPolicy } from './manifest';
import { PROJECT_FILENAME, defaultProject, writeProject, type RvforgeProject } from './project';
import {
  PACKAGING_MODES,
  RUNTIME_PROFILES,
  resolveTargets,
  type ForgeConfig,
  type PackagingMode,
  type RuntimeProfile,
} from './types';

export const CONFIG_FILENAME = 'forge.config.json';

/** A ready-to-edit config with a default-deny policy and no signing refs. */
export function defaultConfig(overrides: Partial<ForgeConfig> = {}): ForgeConfig {
  return {
    app: {
      name: 'My Agent',
      version: '0.1.0',
      publisher: 'Example Publisher',
      identifier: 'com.example.myagent',
      description: 'An RVF agent packaged with RVForge.',
      ...overrides.app,
    },
    rvf: overrides.rvf ?? 'agent.rvf',
    targets: overrides.targets ?? ['linux-x64', 'macos-universal', 'windows-x64'],
    packaging: { mode: 'embedded', ...overrides.packaging },
    runtime: {
      profile: 'wasm',
      rvmVersion: '0.1.0',
      rvmCommit: 'unknown',
      ...overrides.runtime,
    },
    capabilityPolicy: overrides.capabilityPolicy ?? defaultCapabilityPolicy(),
    signing: overrides.signing ?? {},
  };
}

/**
 * Read and validate a config file.
 *
 * @throws {ForgeError} `FORGE_E_NOT_FOUND` if absent, `FORGE_E_CONFIG` if it
 * does not parse or does not validate.
 */
export async function loadConfig(path: string): Promise<ForgeConfig> {
  let raw: string;
  try {
    raw = await readFile(path, 'utf8');
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
      throw new ForgeError(
        ForgeErrorCode.NOT_FOUND,
        `No ${CONFIG_FILENAME} at ${path}. Run "forge init" first.`,
        { path },
      );
    }
    throw new ForgeError(ForgeErrorCode.IO, `Cannot read ${path}.`, { path });
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    throw new ForgeError(
      ForgeErrorCode.CONFIG,
      `${path} is not valid JSON: ${(err as Error).message}`,
      { path },
    );
  }
  return validateConfig(parsed, path);
}

/** Validate an already-parsed config object. */
export function validateConfig(value: unknown, path = CONFIG_FILENAME): ForgeConfig {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ForgeError(ForgeErrorCode.CONFIG, `${path} must contain a JSON object.`, { path });
  }
  const cfg = value as Partial<ForgeConfig>;

  if (!cfg.app || typeof cfg.app !== 'object') {
    throw new ForgeError(ForgeErrorCode.CONFIG, `${path} is missing the "app" section.`, { path });
  }
  if (typeof cfg.rvf !== 'string' || cfg.rvf.trim() === '') {
    throw new ForgeError(ForgeErrorCode.CONFIG, `${path} must set "rvf" to the path of the .rvf file.`, {
      path,
    });
  }

  const mode = cfg.packaging?.mode;
  if (!mode || !PACKAGING_MODES.includes(mode as PackagingMode)) {
    throw new ForgeError(
      ForgeErrorCode.CONFIG,
      `${path} has an invalid packaging.mode; expected one of ${PACKAGING_MODES.join(', ')}.`,
      { path, mode },
    );
  }

  const profile = cfg.runtime?.profile;
  if (!profile || !RUNTIME_PROFILES.includes(profile as RuntimeProfile)) {
    throw new ForgeError(
      ForgeErrorCode.CONFIG,
      `${path} has an invalid runtime.profile; expected one of ${RUNTIME_PROFILES.join(', ')}.`,
      { path, profile },
    );
  }

  return {
    ...cfg,
    app: cfg.app,
    rvf: cfg.rvf,
    targets: resolveTargets(cfg.targets ?? []),
    packaging: { ...cfg.packaging, mode: mode as PackagingMode },
    runtime: {
      profile: profile as RuntimeProfile,
      rvmVersion: cfg.runtime?.rvmVersion ?? '0.0.0',
      rvmCommit: cfg.runtime?.rvmCommit ?? 'unknown',
    },
    capabilityPolicy: normalizeCapabilityPolicy(cfg.capabilityPolicy),
    signing: cfg.signing ?? {},
  } as ForgeConfig;
}

export interface InitResult {
  path: string;
  created: boolean;
  config: ForgeConfig;
  /** Path to the scaffolded `rvforge.json`, when one was written. */
  projectPath?: string;
  project?: RvforgeProject;
  /**
   * Path to a generated key file. The path only — the key itself is written
   * with owner-only permissions and never returned, logged, or transmitted.
   */
  keyFile?: string;
  keyId?: string;
}

export interface InitOptions {
  force?: boolean;
  /** Also scaffold `rvforge.json` (the marketplace project file). */
  project?: boolean;
  /** Generate an Ed25519 publisher key and point the project file at it. */
  keygen?: boolean;
  /** Where to write the key; defaults to `rvforge-publisher.key` beside the project. */
  keyFile?: string;
  /** Where to write `rvforge.json`; defaults to `rvforge.json` beside the config. */
  projectPath?: string;
}

/**
 * Write a starter config, and — when asked — the marketplace project file and
 * a publisher key beside it.
 *
 * `--keygen` implies the project file: a key with nothing to sign is not a
 * useful thing to have generated, and the project file is where its path is
 * recorded.
 *
 * @throws {ForgeError} `FORGE_E_CONFIG` if a file exists and `force` is not
 * set; `FORGE_E_KEY` if a key file exists (never overwritten, even with
 * `--force`).
 */
export async function runInit(path: string, options: InitOptions = {}): Promise<InitResult> {
  const config = defaultConfig();
  const body = `${JSON.stringify(config, null, 2)}\n`;
  try {
    await writeFile(path, body, { encoding: 'utf8', flag: options.force ? 'w' : 'wx' });
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'EEXIST') {
      throw new ForgeError(
        ForgeErrorCode.CONFIG,
        `${path} already exists. Pass --force to overwrite it.`,
        { path },
      );
    }
    throw new ForgeError(ForgeErrorCode.IO, `Cannot write ${path}.`, { path });
  }

  const result: InitResult = { path, created: true, config };
  if (!options.project && !options.keygen) return result;

  const projectPath = resolve(options.projectPath ?? join(dirname(path), PROJECT_FILENAME));
  let keyFile: string | undefined;
  let keyId: string | undefined;

  if (options.keygen) {
    keyFile = resolve(options.keyFile ?? join(dirname(path), 'rvforge-publisher.key'));
    const generated = generatePublisherKey();
    keyId = generated.keyId;
    await writePublisherKey(keyFile, generated);
  }

  const project = defaultProject(
    keyFile ? { publisher: { ...defaultProject().publisher, keyFile } } : {},
  );
  await writeProject(projectPath, project, { force: options.force });

  return { ...result, projectPath, project, ...(keyFile ? { keyFile, keyId } : {}) };
}
