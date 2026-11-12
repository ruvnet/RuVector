import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { ForgeError, ForgeErrorCode } from '../src/errors';
import { main, parseArgs, resolveMode, splitBuildOperands } from '../src/cli';
import { EXIT_CODES } from '../src/errors';
import { PROJECT_FILENAME, writeProject } from '../src/project';
import { buildRvfFixture } from './fixtures/rvf-fixture';
import { fixtureProject } from './fixtures/project-fixture';

describe('parseArgs', () => {
  it('separates the command, positionals, and flags', () => {
    const args = parseArgs(['build', 'agent.rvf', 'linux', '--out', 'dist', '--json']);
    expect(args.command).toBe('build');
    expect(args.positionals).toEqual(['agent.rvf', 'linux']);
    expect(args.flags.get('out')).toBe('dist');
    expect(args.flags.get('json')).toBe(true);
  });

  it('accepts --flag=value form', () => {
    expect(parseArgs(['build', '--out=dist']).flags.get('out')).toBe('dist');
  });

  it('collects repeated --only flags', () => {
    const args = parseArgs(['download', 'b1', '--only', 'a.exe', '--only', 'b.msi']);
    expect(args.repeated.get('only')).toEqual(['a.exe', 'b.msi']);
  });

  it('expands -h and -v', () => {
    expect(parseArgs(['-h']).flags.get('help')).toBe(true);
    expect(parseArgs(['-v']).flags.get('version')).toBe(true);
  });

  it('rejects a value flag with no value', () => {
    expect(() => parseArgs(['build', '--out'])).toThrow(ForgeError);
    expect(() => parseArgs(['build', '--out', '--json'])).toThrow(ForgeError);
  });

  it('rejects a value on a boolean flag', () => {
    try {
      parseArgs(['build', '--json=yes']);
      throw new Error('expected a ForgeError');
    } catch (err) {
      expect((err as ForgeError).code).toBe(ForgeErrorCode.USAGE);
    }
  });

  it('treats everything after -- as positional', () => {
    expect(parseArgs(['verify', '--', '--weird-name.exe']).positionals).toEqual(['--weird-name.exe']);
  });
});

describe('splitBuildOperands', () => {
  it('reads no operands as "use the config"', () => {
    expect(splitBuildOperands([])).toEqual({});
  });

  it('reads a lone .rvf as the payload', () => {
    expect(splitBuildOperands(['agent.rvf'])).toEqual({ rvfPath: 'agent.rvf' });
  });

  it('reads a lone non-.rvf operand as a target, not a path', () => {
    expect(splitBuildOperands(['linux-x64'])).toEqual({ targets: ['linux-x64'] });
  });

  it('reads a path followed by targets', () => {
    expect(splitBuildOperands(['dir/agent.RVF', 'linux', 'windows'])).toEqual({
      rvfPath: 'dir/agent.RVF',
      targets: ['linux', 'windows'],
    });
  });

  it('reads several targets with no path', () => {
    expect(splitBuildOperands(['linux', 'macos'])).toEqual({ targets: ['linux', 'macos'] });
  });
});

describe('resolveMode', () => {
  it('leaves the configured mode in place when --mode is absent', () => {
    expect(resolveMode(undefined)).toBeUndefined();
  });

  it('accepts every packaging mode, case- and space-insensitively', () => {
    expect(resolveMode('embedded')).toBe('embedded');
    expect(resolveMode('  Thin ')).toBe('thin');
    expect(resolveMode('SHARED-READER')).toBe('shared-reader');
  });

  it('rejects an unknown mode as a usage error', () => {
    try {
      resolveMode('capsule');
      throw new Error('expected a ForgeError');
    } catch (err) {
      expect((err as ForgeError).code).toBe(ForgeErrorCode.USAGE);
      expect((err as ForgeError).message).toMatch(/embedded/);
    }
  });

  it('parses --mode as a value flag', () => {
    expect(parseArgs(['build', '--mode', 'thin']).flags.get('mode')).toBe('thin');
    expect(parseArgs(['build', '--mode=embedded']).flags.get('mode')).toBe('embedded');
  });
});

/**
 * End-to-end runs of the publisher verbs through the router, asserting the
 * `--json` envelope and the exit code an unattended caller branches on.
 */
describe('publisher verbs through the router', () => {
  jest.setTimeout(30_000);

  let root: string;
  let rvfPath: string;
  let projectPath: string;
  let keyFile: string;
  let registryDir: string;
  const stdout: string[] = [];
  const stderr: string[] = [];

  const capture = (): void => {
    jest.spyOn(process.stdout, 'write').mockImplementation((chunk: unknown): boolean => {
      stdout.push(String(chunk));
      return true;
    });
    jest.spyOn(process.stderr, 'write').mockImplementation((chunk: unknown): boolean => {
      stderr.push(String(chunk));
      return true;
    });
  };

  const lastEnvelope = <T>(): { ok: boolean; command: string; data?: T; exitCode: number } =>
    JSON.parse(stdout[stdout.length - 1]);

  beforeAll(async () => {
    root = await mkdtemp(join(tmpdir(), 'rvforge-cli-'));
    rvfPath = join(root, 'agent.rvf');
    projectPath = join(root, PROJECT_FILENAME);
    keyFile = join(root, 'publisher.key');
    registryDir = join(root, 'registry');
    await writeFile(rvfPath, buildRvfFixture());
    await writeProject(projectPath, fixtureProject());
    await main([
      'init',
      '--keygen',
      '--quiet',
      '--config',
      join(root, 'forge.config.json'),
      '--key-file',
      keyFile,
      '--project',
      join(root, 'scaffold.json'),
    ]);
  });

  afterAll(async () => {
    await rm(root, { recursive: true, force: true });
  });

  beforeEach(() => {
    stdout.length = 0;
    stderr.length = 0;
    capture();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('packs and reports the draft release in the JSON envelope', async () => {
    const code = await main(['pack', rvfPath, '--project', projectPath, '--json']);
    expect(code).toBe(0);
    const envelope = lastEnvelope<{ release: { releaseId: string }; securityProfile: string }>();
    expect(envelope.ok).toBe(true);
    expect(envelope.command).toBe('pack');
    expect(envelope.data?.securityProfile).toBe('restricted');
    expect(envelope.data?.release.releaseId).toMatch(/^sha256:[0-9a-f]{64}$/);
  });

  it('exits with the license code when the project declares none', async () => {
    const noLicense = join(root, 'no-license.json');
    await writeProject(noLicense, fixtureProject({ license: null }));
    const code = await main(['pack', rvfPath, '--project', noLicense, '--json']);
    expect(code).toBe(EXIT_CODES[ForgeErrorCode.LICENSE]);
    expect(lastEnvelope().ok).toBe(false);
  });

  it('tests and reports skipped categories rather than passes', async () => {
    const code = await main(['test', rvfPath, '--project', projectPath, '--json']);
    expect(code).toBe(0);
    const envelope = lastEnvelope<{ categories: Array<{ id: string; status: string }>; skipped: number }>();
    expect(envelope.data?.skipped).toBeGreaterThanOrEqual(4);
    expect(
      envelope.data?.categories.find((c) => c.id === 'clean-installation')?.status,
    ).toBe('skipped');
  });

  it('exits non-zero when a test category fails', async () => {
    const broken = join(root, 'broken.json');
    await writeProject(broken, fixtureProject({ version: 'latest' }));
    const code = await main(['test', rvfPath, '--project', broken, '--json']);
    expect(code).toBe(EXIT_CODES[ForgeErrorCode.VERIFY_FAILED]);
  });

  it('publishes into the registry named by --registry', async () => {
    const code = await main([
      'publish', rvfPath,
      '--project', projectPath,
      '--registry', registryDir,
      '--key-file', keyFile,
      '--json',
    ]);
    expect(code).toBe(0);
    const envelope = lastEnvelope<{ registryDir: string; releaseId: string; keyId: string }>();
    expect(envelope.data?.registryDir).toBe(registryDir);
    expect(envelope.data?.releaseId).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(envelope.data?.keyId).toMatch(/^ed25519:[0-9a-f]{64}$/);
  });

  it('lists the three publisher verbs in the usage text', async () => {
    const code = await main(['--help', '--json']);
    expect(code).toBe(0);
    const usage = lastEnvelope<{ usage: string }>().data?.usage ?? '';
    expect(usage).toMatch(/rvforge pack <agent\.rvf>/);
    expect(usage).toMatch(/rvforge test <agent\.rvf>/);
    expect(usage).toMatch(/rvforge publish <agent\.rvf>/);
  });
});
