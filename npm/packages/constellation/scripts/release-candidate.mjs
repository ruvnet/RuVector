import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, mkdtemp, rm, access } from 'node:fs/promises';
import { resolve, dirname, join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = resolve(process.argv[2] || await mkdtemp(join(tmpdir(), 'constellation-release-')));
const env = { ...process.env, CI: 'true', FORCE_COLOR: '0' };
delete env.NPM_TOKEN;
delete env.NODE_AUTH_TOKEN;
function run(command, args, cwd = root) {
  const result = spawnSync(command, args, { cwd, env, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
  if (result.error || result.status !== 0) throw new Error(`${command} failed (${result.status}): ${result.error?.message || result.stderr || result.stdout}`);
  return result.stdout;
}
function requireCleanSource() {
  if (run('git', ['status', '--porcelain=v1', '--untracked-files=all']).trim()) throw new Error('Release candidate requires a clean tracked and untracked checkout');
}
requireCleanSource();
await mkdir(output, { recursive: true });
const manifest = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'));
if (manifest.name !== '@ruvector/constellation' || !/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(manifest.version)) throw new Error('Invalid release package identity');
const sha = run('git', ['rev-parse', 'HEAD']).trim();
if (!/^[0-9a-f]{40}$/.test(sha)) throw new Error('Missing exact source SHA');
if (process.env.EXPECTED_SHA && process.env.EXPECTED_SHA !== sha) throw new Error('Source SHA differs from expected_sha');
const tap = run('npm', ['test', '--workspaces=false']);
await writeFile(join(output, 'tests.tap'), tap);
const count = key => {
  const matches = [...tap.matchAll(new RegExp(`^# ${key} (\\d+)\\s*$`, 'gm'))];
  if (matches.length !== 1) throw new Error(`Missing or ambiguous TAP ${key} total`);
  return Number(matches[0][1]);
};
const tests = count('tests');
if (!tests || count('pass') !== tests || count('fail') || count('cancelled') || count('skipped') || count('todo')) throw new Error('Tests must all pass, with nonzero coverage and no skipped/todo tests');
const audit = run('npm', ['audit', '--workspaces=false', '--json']);
await writeFile(join(output, 'audit.json'), audit);
requireCleanSource();
run(process.execPath, ['scripts/prepare-skills.mjs']);
requireCleanSource();
const packs = JSON.parse(run('npm', ['pack', '--ignore-scripts', '--workspaces=false', '--json', '--pack-destination', output]));
if (packs.length !== 1 || !/^[a-zA-Z0-9._-]+\.tgz$/.test(packs[0].filename)) throw new Error('Expected exactly one package tarball');
const pack = packs[0];
if (pack.name !== manifest.name || pack.version !== manifest.version) throw new Error('Packed package differs from source manifest');
const tarball = join(output, pack.filename);
const consumer = await mkdtemp(join(tmpdir(), 'constellation-consumer-'));
try {
  await writeFile(join(consumer, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  run('npm', ['install', '--ignore-scripts', '--workspaces=false', '--no-audit', '--no-fund', tarball], consumer);
  const installed = join(consumer, 'node_modules', '@ruvector', 'constellation');
  const packedManifest = JSON.parse(await readFile(join(installed, 'package.json'), 'utf8'));
  if (packedManifest.name !== manifest.name || packedManifest.version !== manifest.version) throw new Error('Packed consumer identity differs from source');
  const help = run(process.execPath, [join(installed, 'src/cli.mjs'), '--help'], consumer);
  if (!help.trim()) throw new Error('Packed CLI produced empty help');
  run(process.execPath, ['--input-type=module', '-e', "const m=await import('@ruvector/constellation'); if(typeof m.discoverPackages!=='function') throw Error('Missing public API')"], consumer);
  for (const path of ['LICENSE', 'plugin.json', 'skills/ruvector-constellation/SKILL.md']) {
    await access(join(installed, path));
    if (!(await readFile(join(installed, path), 'utf8')).trim()) throw new Error(`Empty packed ${path}`);
  }
} finally {
  await rm(consumer, { recursive: true, force: true });
}
requireCleanSource();
if (run('git', ['rev-parse', 'HEAD']).trim() !== sha) throw new Error('Source HEAD changed during candidate creation');
const bytes = await readFile(tarball);
const seal = { name: pack.name, version: pack.version, filename: pack.filename, sourceSha: sha, sha256: createHash('sha256').update(bytes).digest('hex'), integrity: `sha512-${createHash('sha512').update(bytes).digest('base64')}`, tests, packedConsumer: 'passed' };
await writeFile(join(output, 'seal.json'), JSON.stringify(seal, null, 2) + '\n');
console.log(JSON.stringify(seal, null, 2));
