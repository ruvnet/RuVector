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


# These are the required publication gates, not caller-provided receipt claims.
# Portable paths keep the owned command proof valid after the sealed artifact is
# transferred from the preflight runner to the protected publication runner.
SOURCE = '${SOURCE}'
CANDIDATE = '${CANDIDATE}'
RUST = f'{CANDIDATE}/extracted/{CRATE}-{VERSION}'
QUALIFICATION = {
    'rust-source': (['cargo', 'test', '--locked', '-p', CRATE, '--all-features', '--lib', '--tests'], SOURCE, 'rust', 4),
    'clippy': (['cargo', 'clippy', '--locked', '-p', CRATE, '--all-features', '--all-targets', '--', '-D', 'warnings'], SOURCE, None, None),
    'rust-package': (['cargo', 'package', '--locked', '-p', CRATE, '--target-dir', f'{CANDIDATE}/package-build'], SOURCE, None, None),
    'rust-audit': (['cargo', 'audit', '--file', f'{RUST}/Cargo.lock', '--deny', 'warnings'], SOURCE, None, None),
    'rust-msrv': (['cargo', '+1.81.0', 'check', '--locked', '--manifest-path', f'{RUST}/Cargo.toml', '--no-default-features', '--target-dir', f'{CANDIDATE}/msrv-target'], SOURCE, None, None),
    'rust-packed': (['cargo', 'test', '--locked', '--manifest-path', f'{RUST}/Cargo.toml', '--all-features', '--lib', '--tests', '--target-dir', f'{CANDIDATE}/packed-target'], SOURCE, 'rust', 4),
    'wasm-build': (['cargo', 'build', '--locked', '--manifest-path', f'{RUST}/Cargo.toml', '--release', '--target', 'wasm32-unknown-unknown', '--features', 'wasm', '--target-dir', f'{CANDIDATE}/wasm-target'], SOURCE, None, None),
    'wasm-bindgen': (['wasm-bindgen', f'{CANDIDATE}/wasm-target/wasm32-unknown-unknown/release/ruvector_e_means.wasm', '--target', 'web', '--out-dir', f'{CANDIDATE}/npm-source/wasm', '--out-name', 'ruvector_e_means'], SOURCE, None, None),
    'npm-install': (['npm', 'ci', '--ignore-scripts', '--workspaces=false', '--no-audit', '--no-fund'], f'{CANDIDATE}/npm-source', None, None),
    'npm-audit': (['npm', 'audit', '--audit-level=low', '--workspaces=false'], f'{CANDIDATE}/npm-source', None, None),
    'npm-tests': (['npm', 'test', '--workspaces=false'], f'{CANDIDATE}/npm-source', 'node', None),
    'npm-types': (['npm', 'run', 'typecheck', '--workspaces=false'], f'{CANDIDATE}/npm-source', None, None),
    'npm-consumer': (['npm', 'run', 'test:consumer', '--workspaces=false'], f'{CANDIDATE}/npm-source', 'consumer', None),
    'npm-browser': (['npm', 'run', 'test:browser', '--workspaces=false'], f'{CANDIDATE}/npm-source', 'browser', None),
    'npm-package': (['npm', 'pack', '--json', '--workspaces=false', '--pack-destination', CANDIDATE], f'{CANDIDATE}/npm-source', None, None),
}
BROWSER_MARKER = 'Chromium browser consumer passed: default WASM fetch, fit, predict and numerical oracle'
CONSUMER_MARKERS = (
    'Installed archive runtime/evaluator passed without the optional SDK',
    'Offline installed MetaHarness/WASM consumer passed',
    'Offline installed integration declarations passed',
)


def evidence(output, kind, summaries=None):
    """Derive evidence from actual command output, including when validating it."""
    if kind == 'rust':
        lines = re.findall(r'^test result: (.+)$', output, re.M)
        parsed = [re.fullmatch(r'ok\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out; finished in .+', line) for line in lines]
        if (not parsed or (summaries is not None and len(parsed) != summaries)
                or any(not row or int(row[1]) == 0 or any(int(row[n]) for n in range(2, 6)) for row in parsed)):
            raise ValueError('Rust qualification needs nonzero test suites with no failures, ignored tests, measured or filtered tests')
        return {'testsPassed': sum(int(row[1]) for row in parsed)}
    if kind == 'node':
        values = {key: re.findall(rf'^# {key} (\d+)$', output, re.M) for key in ('pass', 'fail', 'cancelled', 'skipped', 'todo')}
        if any(len(rows) != 1 for rows in values.values()) or int(values['pass'][0]) == 0 or any(int(values[key][0]) for key in ('fail', 'cancelled', 'skipped', 'todo')):
            raise ValueError('Node qualification needs nonzero tests and zero failures/skips/todos')
        return {'testsPassed': int(values['pass'][0])}
    if kind == 'browser':
        if output.splitlines().count(BROWSER_MARKER) != 1:
            raise ValueError('Actual browser numerical qualification missing')
        return {'browserPassed': True}
    if kind == 'consumer':
        if any(output.splitlines().count(marker) != 1 for marker in CONSUMER_MARKERS):
            raise ValueError('Installed archive consumer numerical/type qualification missing')
    return {}


def portable(value, base):
    value = str(value)
    for directory, placeholder in ((base, CANDIDATE), (ROOT, SOURCE)):
        prefix = str(directory)
        if value == prefix or value.startswith(prefix + '/'):
            return placeholder + value[len(prefix):]
    return value


def execute(args, cwd, base, commands, kind=None, env=None, role=None):
    record = {'args': [str(x) for x in args]}
    summaries = None
    if role is not None:
        expected_args, expected_cwd, expected_kind, summaries = QUALIFICATION[role]
        record.update(role=role, args=[portable(x, base) for x in args], cwd=portable(cwd, base))
        if (record['args'], record['cwd'], kind) != (expected_args, expected_cwd, expected_kind):
            raise ValueError('Owned qualification command differs from required role: ' + role)
    log = base / f'command-{len(commands)}.log'
    with log.open('w') as stream:
        result = subprocess.run(args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, **(env or {})})
    output = log.read_text(errors='replace')
    record.update(exitCode=result.returncode, log=log.name, logSha256=sha(log))
    if env:
        # Only this explicit, non-secret build override map is recorded.
        record['environment'] = env
    commands.append(record)
    if result.returncode:
        print(output[-12000:], file=sys.stderr)
        raise RuntimeError(f'Command failed: {args}')
    try:
        record.update(evidence(output, kind, summaries))
    except ValueError:
        print(output[-12000:], file=sys.stderr)
        raise
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
    for role, args in (
        ('rust-source', ['cargo', 'test', '--locked', '-p', CRATE, '--all-features', '--lib', '--tests']),
    ):
        execute(args, ROOT, base, commands, 'rust', env, role=role)
    execute(['cargo', 'clippy', '--locked', '-p', CRATE, '--all-features', '--all-targets', '--', '-D', 'warnings'], ROOT, base, commands, env=env, role='clippy')
    target = base / 'package-build'
    execute(['cargo', 'package', '--locked', '-p', CRATE, '--target-dir', str(target)], ROOT, base, commands, env=env, role='rust-package')
    crate = base / f'{CRATE}-{VERSION}.crate'
    shutil.copyfile(target / 'package' / crate.name, crate)
    extract(crate, base / 'extracted')
    rust = base / 'extracted' / f'{CRATE}-{VERSION}'
    execute(['cargo', 'audit', '--file', str(rust / 'Cargo.lock'), '--deny', 'warnings'], ROOT, base, commands, env=env, role='rust-audit')
    execute(['cargo', '+1.81.0', 'check', '--locked', '--manifest-path', str(rust / 'Cargo.toml'), '--no-default-features', '--target-dir', str(base / 'msrv-target')], ROOT, base, commands, env=env, role='rust-msrv')
    execute(['cargo', 'test', '--locked', '--manifest-path', str(rust / 'Cargo.toml'), '--all-features', '--lib', '--tests', '--target-dir', str(base / 'packed-target')], ROOT, base, commands, 'rust', env, role='rust-packed')
    npm = base / 'npm-source'
    shutil.copytree(ROOT / 'npm/packages/e-means', npm, ignore=shutil.ignore_patterns('node_modules', '*.tgz'))
    wasm_env = {**env, 'CARGO_PROFILE_RELEASE_OPT_LEVEL': '3', 'CARGO_PROFILE_RELEASE_LTO': 'fat',
                'CARGO_PROFILE_RELEASE_CODEGEN_UNITS': '1', 'CARGO_PROFILE_RELEASE_STRIP': 'true',
                'CARGO_PROFILE_RELEASE_PANIC': 'unwind'}
    execute(['cargo', 'build', '--locked', '--manifest-path', str(rust / 'Cargo.toml'), '--release', '--target', 'wasm32-unknown-unknown', '--features', 'wasm', '--target-dir', str(base / 'wasm-target')], ROOT, base, commands, env=wasm_env, role='wasm-build')
    version = subprocess.check_output(['wasm-bindgen', '--version'], text=True).strip()
    if version != 'wasm-bindgen 0.2.125':
        raise ValueError('Wrong wasm-bindgen CLI')
    execute(['wasm-bindgen', str(base / 'wasm-target/wasm32-unknown-unknown/release/ruvector_e_means.wasm'), '--target', 'web', '--out-dir', str(npm / 'wasm'), '--out-name', 'ruvector_e_means'], ROOT, base, commands, role='wasm-bindgen')
    execute(['npm', 'ci', '--ignore-scripts', '--workspaces=false', '--no-audit', '--no-fund'], npm, base, commands, role='npm-install')
    execute(['npm', 'audit', '--audit-level=low', '--workspaces=false'], npm, base, commands, role='npm-audit')
    execute(['npm', 'test', '--workspaces=false'], npm, base, commands, 'node', role='npm-tests')
    execute(['npm', 'run', 'typecheck', '--workspaces=false'], npm, base, commands, role='npm-types')
    execute(['npm', 'run', 'test:consumer', '--workspaces=false'], npm, base, commands, 'consumer', role='npm-consumer')
    execute(['npm', 'run', 'test:browser', '--workspaces=false'], npm, base, commands, 'browser', role='npm-browser')
    execute(['npm', 'pack', '--json', '--workspaces=false', '--pack-destination', str(base)], npm, base, commands, role='npm-package')
    archive = base / f'ruvector-e-means-{VERSION}.tgz'
    if not archive.is_file():
        raise ValueError('Missing packed npm archive')
    identity(source)
    receipt = {'schema': 'ruvector.e-means-candidate/1', 'sourceSha': source,
               'commands': commands, 'artifacts': {crate.name: sha(crate), archive.name: sha(archive)},
               'wasmSha256': sha(npm / 'wasm/ruvector_e_means_bg.wasm'), 'wasmBindgen': version,
               'rustcVersion': subprocess.check_output(['rustc', '--version'], text=True).strip(),
               'cargoVersion': subprocess.check_output(['cargo', '--version'], text=True).strip(),
               'nodeVersion': subprocess.check_output(['node', '--version'], text=True).strip()}
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


def validate_payload(base):
    files = json.loads((base / 'FILES.json').read_text())
    if not isinstance(files, dict) or not files:
        raise ValueError('Missing complete candidate FILES manifest')
    paths = list(base.rglob('*'))
    if any(path.is_symlink() for path in paths):
        raise ValueError('Candidate payload may not contain symlinks')
    actual = {path.relative_to(base).as_posix() for path in paths if path.is_file() and path != base / 'FILES.json'}
    if set(files) != actual:
        raise ValueError('Candidate FILES manifest must cover the complete payload')
    for relative, digest in files.items():
        path = (base / relative).resolve()
        if (not isinstance(relative, str) or Path(relative).as_posix() != relative
                or '\\' in relative or not path.is_relative_to(base.resolve())
                or not re.fullmatch('[a-f0-9]{64}', str(digest)) or sha(path) != digest):
            raise ValueError('Candidate payload changed: ' + relative)
    # The transferred harness is reviewed source at this exact checkout, not a
    # second authority supplied by the artifact producer.
    npm = ROOT / 'npm/packages/e-means'
    expected_harness = {'harness/' + name: sha(npm / name) for name in ('package.json', 'package-lock.json')}
    expected_harness.update({'harness/test/' + path.relative_to(npm / 'test').as_posix(): sha(path)
                             for path in (npm / 'test').rglob('*') if path.is_file()})
    if {name: digest for name, digest in files.items() if name.startswith('harness/')} != expected_harness:
        raise ValueError('Candidate harness differs from exact-source checkout')
    return files


def validate_qualification(base, receipt, files):
    commands = receipt.get('commands', [])
    if not isinstance(commands, list) or [c.get('role') for c in commands] != list(QUALIFICATION):
        raise ValueError('Missing, duplicated or reordered required qualification roles')
    for index, command in enumerate(commands):
        role = command['role']
        args, cwd, kind, summaries = QUALIFICATION[role]
        log = f'command-{index}.log'
        if (command.get('args') != args or command.get('cwd') != cwd or type(command.get('exitCode')) is not int or command['exitCode'] != 0
                or command.get('log') != log or files.get(log) != command.get('logSha256')):
            raise ValueError('Qualification command/log differs from required role: ' + role)
        derived = evidence((base / log).read_text(errors='replace'), kind, summaries)
        claims = {key: command[key] for key in ('testsPassed', 'browserPassed') if key in command}
        if claims != derived or any(type(claims[key]) is not type(value) for key, value in derived.items()):
            raise ValueError('Qualification claims differ from actual owned log: ' + role)


def validate_candidate(base, source):
    files = validate_payload(base)
    receipt = json.loads((base / 'candidate.json').read_text())
    if (not re.fullmatch('[a-f0-9]{40}', source)
            or receipt.get('schema') != 'ruvector.e-means-candidate/1' or receipt.get('sourceSha') != source):
        raise ValueError('Candidate identity mismatch')
    if receipt.get('wasmBindgen') != 'wasm-bindgen 0.2.125':
        raise ValueError('Candidate WASM binding tool differs from pinned version')
    validate_qualification(base, receipt, files)
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
    shutil.copyfile(base / 'harness/package-lock.json', installed / 'package-lock.json')
    selected = ['e-means', 'runtime', 'typed-bridge', 'quantization', 'ivf']
    execute(['node', '--test', '--test-reporter=tap', *[str(installed / f'test/{name}.test.mjs') for name in selected]], consumer, consumer, commands, 'node')
    # The portable runtime first proves optional-SDK absence. Warm only the six
    # source-lock tarballs, then require real offline Darwin tuning and types.
    execute(['node', str(installed / 'test/offline-tooling-lock.mjs'), 'warm', str(installed / 'package-lock.json')], consumer, consumer, commands)
    execute(['node', str(installed / 'test/packed-consumer.mjs')], consumer, consumer, commands, 'consumer')


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
    metadata = json.loads(execute(['cargo', 'metadata', '--quiet', '--locked', '--format-version', '1', '--manifest-path', str(cargo / 'Cargo.toml')], cargo, fresh, commands, env=env))
    resolved = [p for p in metadata['packages'] if p['name'] == CRATE and p['version'] == VERSION]
    if len(resolved) != 1 or resolved[0].get('source') != 'registry+https://github.com/rust-lang/crates.io-index':
        raise ValueError('Fresh Cargo consumer did not resolve the official registry package')
    import tomllib
    lock = tomllib.loads((cargo / 'Cargo.lock').read_text())
    locked = [p for p in lock['package'] if p['name'] == CRATE and p['version'] == VERSION]
    if len(locked) != 1 or locked[0].get('checksum') != receipt['artifacts'][f'{CRATE}-{VERSION}.crate']:
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
    execute(['node', '--test', '--test-reporter=tap', *[str(p) for p in sorted((installed / 'test').glob('*.test.mjs'))]], npm, fresh, commands, 'node')
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
        identity(args.expected_sha)
        validate_candidate(base, args.expected_sha)


if __name__ == '__main__':
    main()
