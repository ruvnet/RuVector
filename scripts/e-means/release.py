#!/usr/bin/env python3
"""Seal, publish and independently exercise the two shared e-means archives."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

import registry

ROOT = Path(__file__).resolve().parents[2]
REPO = 'ruvnet/RuVector'
PREFLIGHT = '.github/workflows/e-means-preflight.yml'
PUBLISHER = 'e-means-protected-release'
VERSION = '0.1.0'
NPM_NAME = '@ruvector/e-means'
CRATE = 'ruvector-e-means'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gh(endpoint):
    return json.loads(subprocess.check_output(['gh', 'api', endpoint], text=True))


def pages(endpoint, field):
    result = []
    for page in range(1, 101):
        rows = gh(endpoint + ('&' if '?' in endpoint else '?') + f'per_page=100&page={page}')[field]
        result.extend(rows)
        if len(rows) < 100:
            return result
    raise ValueError('Pagination exceeded explicit bound')


def identity(source):
    if not re.fullmatch('[a-f0-9]{40}', source):
        raise ValueError('Expected full immutable SHA')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != source:
        raise ValueError('Checkout source mismatch')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT):
        raise ValueError('Source checkout must be clean')


def protected(source):
    identity(source)
    expected = ('true', PUBLISHER, 'refs/heads/main', source)
    actual = tuple(os.environ.get(k) for k in ('GITHUB_ACTIONS', 'GITHUB_WORKFLOW', 'GITHUB_REF', 'GITHUB_SHA'))
    if actual != expected:
        raise ValueError('Publication requires the protected main workflow')
    if gh(f'repos/{REPO}/git/ref/heads/main')['object']['sha'] != source:
        raise ValueError('Main moved since qualification')
    if gh(f'repos/{REPO}/issues?state=open&labels=release-blocker&per_page=1'):
        raise ValueError('Open release blockers')


def execute(args, cwd, base, commands, kind=None, env=None):
    log = base / f'command-{len(commands)}.log'
    with log.open('w') as stream:
        result = subprocess.run(args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, **(env or {})})
    output = log.read_text(errors='replace')
    record = {'args': [str(x) for x in args], 'exitCode': result.returncode,
              'log': log.name, 'logSha256': sha(log)}
    commands.append(record)
    if result.returncode:
        print(output[-12000:], file=sys.stderr)
        raise RuntimeError(f'Command failed: {args}')
    if kind == 'rust':
        summaries = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', output)
        if not summaries or not sum(int(x[0]) for x in summaries) or any(int(x[1]) or int(x[2]) for x in summaries):
            raise ValueError('Rust qualification needs executed tests with no failures or ignored tests')
        record['testsPassed'] = sum(int(x[0]) for x in summaries)
    if kind == 'node':
        passed = re.search(r'^# pass (\d+)$', output, re.M)
        if not passed or int(passed[1]) == 0 or any(not re.search(rf'^# {key} 0$', output, re.M) for key in ('fail', 'cancelled', 'skipped', 'todo')):
            raise ValueError('Node qualification needs nonzero tests and zero failures/skips/todos')
        record['testsPassed'] = int(passed[1])
    if kind == 'browser':
        if 'Chromium browser consumer passed:' not in output:
            raise ValueError('Actual browser numerical qualification missing')
        record['browserPassed'] = True
    print(json.dumps(record), flush=True)
    return output


def extract(archive, destination):
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive member')
        handle.extractall(destination, filter='data')


def seal(source, base):
    identity(source)
    base.mkdir(parents=True, exist_ok=False)
    commands = []
    env = {'RUSTFLAGS': '', 'CARGO_BUILD_JOBS': '2', 'CARGO_INCREMENTAL': '0', 'CARGO_PROFILE_DEV_DEBUG': '0'}
    for args in (
        ['cargo', 'test', '--locked', '-p', CRATE, '--all-features', '--lib', '--tests'],
        ['cargo', 'test', '--locked', '-p', 'ruvector-core', '--no-default-features', '--test', 'e_means_quantization'],
        ['cargo', 'test', '--locked', '-p', 'ruvector-pq-search', '--test', 'e_means'],
        ['cargo', 'test', '--locked', '-p', 'ruvector-rairs', '--test', 'e_means'],
    ):
        execute(args, ROOT, base, commands, 'rust', env)
    execute(['cargo', 'clippy', '--locked', '-p', CRATE, '--all-features', '--all-targets', '--', '-D', 'warnings'], ROOT, base, commands, env=env)
    target = base / 'package-build'
    execute(['cargo', 'package', '--locked', '-p', CRATE, '--target-dir', str(target)], ROOT, base, commands, env=env)
    crate = base / f'{CRATE}-{VERSION}.crate'
    shutil.copyfile(target / 'package' / crate.name, crate)
    extract(crate, base / 'extracted')
    rust = base / 'extracted' / f'{CRATE}-{VERSION}'
    execute(['cargo', 'audit', '--file', str(rust / 'Cargo.lock'), '--deny', 'warnings'], ROOT, base, commands, env=env)
    execute(['cargo', '+1.81.0', 'check', '--locked', '--manifest-path', str(rust / 'Cargo.toml'), '--no-default-features', '--target-dir', str(base / 'msrv-target')], ROOT, base, commands, env=env)
    execute(['cargo', 'test', '--locked', '--manifest-path', str(rust / 'Cargo.toml'), '--all-features', '--lib', '--tests', '--target-dir', str(base / 'packed-target')], ROOT, base, commands, 'rust', env)
    npm = base / 'npm-source'
    shutil.copytree(ROOT / 'npm/packages/e-means', npm, ignore=shutil.ignore_patterns('node_modules', '*.tgz'))
    execute(['cargo', 'build', '--locked', '--manifest-path', str(rust / 'Cargo.toml'), '--release', '--target', 'wasm32-unknown-unknown', '--features', 'wasm', '--target-dir', str(base / 'wasm-target')], ROOT, base, commands, env=env)
    version = subprocess.check_output(['wasm-bindgen', '--version'], text=True).strip()
    if version != 'wasm-bindgen 0.2.125':
        raise ValueError('Wrong wasm-bindgen CLI')
    execute(['wasm-bindgen', str(base / 'wasm-target/wasm32-unknown-unknown/release/ruvector_e_means.wasm'), '--target', 'web', '--out-dir', str(npm / 'wasm'), '--out-name', 'ruvector_e_means'], ROOT, base, commands)
    execute(['npm', 'ci', '--ignore-scripts', '--workspaces=false', '--no-audit', '--no-fund'], npm, base, commands)
    execute(['npm', 'audit', '--audit-level=low', '--workspaces=false'], npm, base, commands)
    execute(['npm', 'test', '--workspaces=false'], npm, base, commands, 'node')
    execute(['npm', 'run', 'typecheck', '--workspaces=false'], npm, base, commands)
    execute(['npm', 'run', 'test:consumer', '--workspaces=false'], npm, base, commands)
    execute(['npm', 'run', 'test:browser', '--workspaces=false'], npm, base, commands, 'browser')
    execute(['npm', 'pack', '--json', '--workspaces=false', '--pack-destination', str(base)], npm, base, commands)
    archive = base / f'ruvector-e-means-{VERSION}.tgz'
    if not archive.is_file():
        raise ValueError('Missing packed npm archive')
    identity(source)
    receipt = {'schema': 'ruvector.e-means-candidate/1', 'sourceSha': source,
               'commands': commands, 'artifacts': {crate.name: sha(crate), archive.name: sha(archive)},
               'wasmSha256': sha(npm / 'wasm/ruvector_e_means_bg.wasm'), 'wasmBindgen': version}
    (base / 'candidate.json').write_text(json.dumps(receipt, indent=2) + '\n')
    # Transfer only immutable archives, actual logs and the test harness; no
    # build trees or source checkout become publication authority.
    harness = base / 'harness'
    shutil.copytree(npm / 'test', harness / 'test')
    shutil.copyfile(npm / 'package.json', harness / 'package.json')
    shutil.copyfile(npm / 'package-lock.json', harness / 'package-lock.json')
    for path in base.iterdir():
        if path.is_dir() and path != harness:
            shutil.rmtree(path)
    files = {str(p.relative_to(base)): sha(p) for p in base.rglob('*') if p.is_file()}
    (base / 'FILES.json').write_text(json.dumps(files, indent=2) + '\n')


def validate_candidate(base, source):
    files = json.loads((base / 'FILES.json').read_text())
    for relative, digest in files.items():
        path = (base / relative).resolve()
        if not path.is_relative_to(base.resolve()) or sha(path) != digest:
            raise ValueError('Candidate payload changed: ' + relative)
    receipt = json.loads((base / 'candidate.json').read_text())
    if receipt.get('schema') != 'ruvector.e-means-candidate/1' or receipt.get('sourceSha') != source:
        raise ValueError('Candidate identity mismatch')
    commands = receipt.get('commands', [])
    if len(commands) < 15 or any(c['exitCode'] != 0 or sha(base / c['log']) != c['logSha256'] for c in commands):
        raise ValueError('Missing or failed real qualification commands')
    if not any(c.get('browserPassed') for c in commands) or sum(c.get('testsPassed', 0) for c in commands) <= 0:
        raise ValueError('Missing executed numerical/browser tests')
    expected = {f'{CRATE}-{VERSION}.crate', f'ruvector-e-means-{VERSION}.tgz'}
    if set(receipt['artifacts']) != expected or any(sha(base / name) != digest for name, digest in receipt['artifacts'].items()):
        raise ValueError('Archive seal mismatch')
    data = registry.metadata(base / f'{CRATE}-{VERSION}.crate')
    if (data['name'], data['vers']) != (CRATE, VERSION):
        raise ValueError('Rust archive identity mismatch')
    with tarfile.open(base / f'ruvector-e-means-{VERSION}.tgz') as tar:
        manifest = json.load(tar.extractfile('package/package.json'))
        if (manifest['name'], manifest['version']) != (NPM_NAME, VERSION):
            raise ValueError('npm archive identity mismatch')
        if hashlib.sha256(tar.extractfile('package/wasm/ruvector_e_means_bg.wasm').read()).hexdigest() != receipt['wasmSha256']:
            raise ValueError('Bundled WASM differs from actual source build')
    return receipt


def node20(source, base):
    identity(source)
    validate_candidate(base, source)
    consumer = base.parent / 'node20-consumer'
    consumer.mkdir()
    (consumer / 'package.json').write_text('{"private":true,"type":"module"}')
    commands = []
    execute(['npm', 'install', '--ignore-scripts', '--offline', '--no-audit', '--no-fund', str(base / f'ruvector-e-means-{VERSION}.tgz')], consumer, consumer, commands)
    installed = consumer / 'node_modules/@ruvector/e-means'
    shutil.copytree(base / 'harness/test', installed / 'test')
    selected = ['e-means', 'runtime', 'typed-bridge', 'quantization', 'ivf']
    execute(['node', '--test', *[str(installed / f'test/{name}.test.mjs') for name in selected]], consumer, consumer, commands, 'node')


def select_and_download(source, base):
    protected(source)
    runs = pages(f'repos/{REPO}/actions/workflows/e-means-preflight.yml/runs?head_sha={source}&event=push', 'workflow_runs')
    runs = [r for r in runs if r['head_branch'].startswith('release/') and r['path'] == PREFLIGHT and r['status'] == 'completed' and r['conclusion'] == 'success']
    if not runs:
        raise ValueError('No successful exact-source pre-main candidate')
    run = max(runs, key=lambda r: r['id'])
    jobs = pages(f'repos/{REPO}/actions/runs/{run["id"]}/jobs?filter=latest', 'jobs')
    if not jobs or any(j['status'] != 'completed' or j['conclusion'] != 'success' for j in jobs):
        raise ValueError('Incomplete or failed preflight jobs')
    artifacts = pages(f'repos/{REPO}/actions/runs/{run["id"]}/artifacts', 'artifacts')
    matches = [a for a in artifacts if a['name'] == f'e-means-candidate-{source}' and not a['expired']]
    if len(matches) != 1 or not re.fullmatch('sha256:[a-f0-9]{64}', matches[0].get('digest', '')):
        raise ValueError('Missing unique sealed candidate')
    item = matches[0]
    base.mkdir(parents=True, exist_ok=False)
    archive = base / 'candidate.zip'
    with archive.open('xb') as out:
        subprocess.run(['gh', 'api', f'repos/{REPO}/actions/artifacts/{item["id"]}/zip'], stdout=out, check=True)
    if 'sha256:' + sha(archive) != item['digest']:
        raise ValueError('GitHub artifact digest mismatch')
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe candidate ZIP')
        handle.extractall(base / 'candidate')
    (base / 'ci.json').write_text(json.dumps({'sourceSha': source, 'runId': run['id'], 'jobs': jobs, 'artifact': item}, indent=2))
    validate_candidate(base / 'candidate', source)


def npm_public():
    url = f'https://registry.npmjs.org/@ruvector%2fe-means/{VERSION}'
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            metadata = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    if (metadata.get('name'), metadata.get('version')) != (NPM_NAME, VERSION):
        raise ValueError('Registry npm identity mismatch')
    with urllib.request.urlopen(metadata['dist']['tarball'], timeout=60) as response:
        data = response.read()
    integrity = 'sha512-' + base64.b64encode(hashlib.sha512(data).digest()).decode()
    if integrity != metadata['dist']['integrity']:
        raise ValueError('npm public integrity mismatch')
    return data


def publish(source, base):
    protected(source)
    candidate = base / 'candidate'
    receipt = validate_candidate(candidate, source)
    crate = candidate / f'{CRATE}-{VERSION}.crate'
    npm = candidate / f'ruvector-e-means-{VERSION}.tgz'
    # Check both immutable registries before either upload.
    rust_bytes = registry.downloaded(CRATE, VERSION)
    npm_bytes = npm_public()
    for path, data in ((crate, rust_bytes), (npm, npm_bytes)):
        if data is not None and hashlib.sha256(data).hexdigest() != sha(path):
            raise ValueError('Public version already exists with different bytes: ' + path.name)
    if rust_bytes is None:
        registry.confirm_version_absent(CRATE, VERSION)
        token = os.environ.get('CARGO_REGISTRY_TOKEN')
        if not token:
            raise ValueError('Missing crate publication credential')
        meta = json.dumps(registry.metadata(crate)).encode()
        payload = crate.read_bytes()
        body = struct.pack('<I', len(meta)) + meta + struct.pack('<I', len(payload)) + payload
        request = urllib.request.Request('https://crates.io/api/v1/crates/new', data=body, method='PUT', headers={'User-Agent': registry.USER_AGENT, 'Authorization': token, 'Content-Type': 'application/octet-stream'})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        if result.get('errors'):
            raise RuntimeError(str(result['errors']).replace(token, '[REDACTED]'))
        rust_bytes = registry.wait_public(CRATE, VERSION)
    if rust_bytes is None or hashlib.sha256(rust_bytes).hexdigest() != sha(crate):
        raise ValueError('Public crate unavailable or differs from seal')
    if npm_bytes is None:
        protected(source)
        subprocess.run(['npm', 'publish', str(npm), '--access', 'public', '--provenance'], check=True)
        for _ in range(60):
            npm_bytes = npm_public()
            if npm_bytes is not None:
                break
            time.sleep(10)
    if npm_bytes is None or hashlib.sha256(npm_bytes).hexdigest() != sha(npm):
        raise ValueError('Public npm archive unavailable or differs from seal')
    public = base / 'public'
    public.mkdir()
    (public / crate.name).write_bytes(rust_bytes)
    (public / npm.name).write_bytes(npm_bytes)
    report = {'schema': 'ruvector.e-means-publication/1', 'sourceSha': source, 'artifacts': receipt['artifacts']}
    (base / 'publication.json').write_text(json.dumps(report, indent=2) + '\n')


def verify_public(source, base):
    protected(source)
    candidate = base / 'candidate'
    receipt = validate_candidate(candidate, source)
    publication = json.loads((base / 'publication.json').read_text())
    if publication.get('sourceSha') != source or publication.get('artifacts') != receipt['artifacts']:
        raise ValueError('Publication receipt mismatch')
    for name, digest in receipt['artifacts'].items():
        if sha(base / 'public' / name) != digest:
            raise ValueError('Canonical public archive changed')
    fresh = base / 'fresh'
    fresh.mkdir()
    cargo = fresh / 'cargo'
    cargo.mkdir()
    (cargo / 'src').mkdir()
    (cargo / 'Cargo.toml').write_text('[package]\nname="e-means-public-consumer"\nversion="0.0.0"\nedition="2021"\n[workspace]\n[dependencies]\nruvector-e-means="=0.1.0"\n')
    shutil.copyfile(ROOT / 'scripts/e-means/public-consumer.rs', cargo / 'src/main.rs')
    commands = []
    env = {'CARGO_HOME': str(fresh / 'cargo-home'), 'CARGO_TARGET_DIR': str(fresh / 'target'),
           'RUSTFLAGS': '', 'CARGO_BUILD_JOBS': '2', 'CARGO_PROFILE_DEV_DEBUG': '0'}
    output = execute(['cargo', 'run', '--manifest-path', str(cargo / 'Cargo.toml')], cargo, fresh, commands, env=env)
    checks = re.search(r'E_MEANS_PUBLIC_CHECKS=(\d+)', output)
    if not checks or int(checks[1]) < 10:
        raise ValueError('Public Rust numerical checks missing')
    lock = (cargo / 'Cargo.lock').read_text()
    if receipt['artifacts'][f'{CRATE}-{VERSION}.crate'] not in lock:
        raise ValueError('Fresh Cargo consumer did not resolve sealed public checksum')
    npm = fresh / 'npm'
    npm.mkdir()
    tools = json.loads((candidate / 'harness/package.json').read_text())
    manifest = {'private': True, 'type': 'module', 'dependencies': {NPM_NAME: VERSION},
                'devDependencies': tools['devDependencies'], 'overrides': tools['overrides']}
    (npm / 'package.json').write_text(json.dumps(manifest))
    execute(['npm', 'install', '--ignore-scripts', '--workspaces=false', '--no-audit', '--no-fund'], npm, fresh, commands)
    installed = npm / 'node_modules/@ruvector/e-means'
    with tarfile.open(base / 'public' / f'ruvector-e-means-{VERSION}.tgz') as tar:
        for member in tar.getmembers():
            if member.isfile():
                relative = Path(member.name).relative_to('package')
                if (installed / relative).read_bytes() != tar.extractfile(member).read():
                    raise ValueError('Installed npm payload differs from sealed public archive')
    shutil.copytree(candidate / 'harness/test', installed / 'test')
    execute(['node', '--test', *[str(p) for p in sorted((installed / 'test').glob('*.test.mjs'))]], npm, fresh, commands, 'node')
    execute(['node', str(npm / 'node_modules/typescript/bin/tsc'), '--noEmit', '-p', str(installed / 'test/tsconfig.json')], npm, fresh, commands)
    execute(['node', str(installed / 'test/browser-consumer.mjs')], npm, fresh, commands, 'browser')
    report = {'schema': 'ruvector.e-means-public-consumers/1', 'sourceSha': source,
              'artifacts': receipt['artifacts'], 'rustChecks': int(checks[1]), 'commands': commands}
    (base / 'public-consumers.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('seal', 'download', 'publish', 'validate', 'verify-public', 'node20'))
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    base = args.directory.resolve()
    if args.mode == 'seal':
        seal(args.expected_sha, base)
    elif args.mode == 'download':
        select_and_download(args.expected_sha, base)
    elif args.mode == 'publish':
        publish(args.expected_sha, base)
    elif args.mode == 'verify-public':
        verify_public(args.expected_sha, base)
    elif args.mode == 'node20':
        node20(args.expected_sha, base)
    else:
        validate_candidate(base, args.expected_sha)


if __name__ == '__main__':
    main()
