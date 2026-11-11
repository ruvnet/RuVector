#!/usr/bin/env python3
"""Publish the exact-source shared e-means release and verify every asset byte.

Only the protected main job may mutate GitHub. Existing tags and asset bytes
are immutable; retries may complete an otherwise identical partial draft.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import tomllib
import zipfile

import release

TAG = 'e-means-v0.1.0'
TITLE = 'Shared e-means 0.1.0 (Rust and WebAssembly)'


def gh(*args):
    return subprocess.check_output(['gh', *args], text=True).strip()


def api_optional(endpoint):
    result = subprocess.run(['gh', 'api', endpoint], capture_output=True, text=True)
    if result.returncode:
        if re.search(r'\(HTTP 404\)', result.stderr):
            return None
        raise ValueError('Cannot establish GitHub identity or absence: ' + endpoint)
    return json.loads(result.stdout)


def inventory(endpoint):
    rows = []
    for page in range(1, 101):
        batch = json.loads(gh('api', endpoint + f'?per_page=100&page={page}'))
        if not isinstance(batch, list):
            raise ValueError('Invalid authenticated GitHub inventory')
        rows.extend(batch)
        if len(batch) < 100:
            return rows
    raise ValueError('GitHub inventory exceeded explicit pagination bound')


def guard(source):
    release.protected(source)
    if os.environ.get('E_MEANS_RELEASE_ENVIRONMENT') != 'npm-production':
        raise ValueError('GitHub release requires the protected npm-production environment')


def tag_source():
    ref = api_optional(f'repos/{release.REPO}/git/ref/tags/{TAG}')
    if ref is None:
        return None
    obj = ref['object']
    for _ in range(5):
        if obj['type'] == 'commit' and re.fullmatch('[a-f0-9]{40}', obj['sha']):
            return obj['sha']
        if obj['type'] != 'tag':
            break
        obj = json.loads(gh('api', f'repos/{release.REPO}/git/tags/{obj["sha"]}'))['object']
    raise ValueError('Release tag does not identify a commit')


def ensure_tag(source):
    current = tag_source()
    if current is None:
        guard(source)
        gh('api', '--method', 'POST', f'repos/{release.REPO}/git/refs',
           '-f', f'ref=refs/tags/{TAG}', '-f', f'sha={source}')
        current = tag_source()
    if current != source:
        raise ValueError('Conflicting immutable release tag; never move or overwrite it')


def assert_tag(source):
    if tag_source() != source:
        raise ValueError('Immutable release tag is missing or changed')


def existing_release():
    # Authenticated listing includes drafts, unlike relying on the public tag URL.
    matches = [row for row in inventory(f'repos/{release.REPO}/releases') if row.get('tag_name') == TAG]
    if len(matches) > 1:
        raise ValueError('Multiple releases identify the same tag')
    return matches[0] if matches else None


def validate_ci(base, source):
    ci = json.loads((base / 'ci.json').read_text())
    item = ci.get('artifact', {})
    if (ci.get('sourceSha') != source or not ci.get('jobs')
            or any(j.get('status') != 'completed' or j.get('conclusion') != 'success' for j in ci['jobs'])
            or item.get('name') != f'e-means-candidate-{source}' or item.get('expired') is not False
            or not re.fullmatch('sha256:[a-f0-9]{64}', item.get('digest', ''))):
        raise ValueError('Missing successful exact-source preflight evidence')
    run = release.gh(f'repos/{release.REPO}/actions/runs/{ci["runId"]}')
    if (run.get('head_sha') != source or run.get('event') != 'push'
            or not run.get('head_branch', '').startswith('release/') or run.get('path') != release.PREFLIGHT
            or run.get('status') != 'completed' or run.get('conclusion') != 'success'):
        raise ValueError('Preflight run does not identify the accepted source')
    jobs = release.pages(f'repos/{release.REPO}/actions/runs/{ci["runId"]}/jobs?filter=latest', 'jobs')
    def job_identity(job):
        return (job.get('id'), job.get('name'), job.get('status'), job.get('conclusion'))
    if (not {'qualify', 'node20'}.issubset({job.get('name') for job in jobs})
            or any(job.get('status') != 'completed' or job.get('conclusion') != 'success' for job in jobs)
            or sorted(map(job_identity, jobs)) != sorted(map(job_identity, ci['jobs']))):
        raise ValueError('Preflight job evidence differs from GitHub authority')
    official = release.gh(f'repos/{release.REPO}/actions/artifacts/{item["id"]}')
    if (any(official.get(key) != item.get(key) for key in ('id', 'name', 'digest', 'expired'))
            or official.get('workflow_run', {}).get('id') != ci['runId']
            or official.get('workflow_run', {}).get('head_sha') != source):
        raise ValueError('Candidate artifact differs from GitHub authority')
    if 'sha256:' + release.sha(base / 'candidate.zip') != item['digest']:
        raise ValueError('Candidate artifact ZIP digest mismatch')
    with zipfile.ZipFile(base / 'candidate.zip') as archive:
        entries = [row for row in archive.infolist() if not row.is_dir()]
        names = [row.filename for row in entries]
        actual = {p.relative_to(base / 'candidate').as_posix() for p in (base / 'candidate').rglob('*') if p.is_file()}
        if len(set(names)) != len(names) or set(names) != actual:
            raise ValueError('Extracted candidate differs from accepted artifact inventory')
        for row in entries:
            if archive.read(row) != (base / 'candidate' / row.filename).read_bytes():
                raise ValueError('Extracted candidate differs from accepted artifact bytes')
    return ci


def validate_public(base, source, candidate):
    publication = json.loads((base / 'publication.json').read_text())
    consumers = json.loads((base / 'public-consumers.json').read_text())
    for row, schema in ((publication, 'ruvector.e-means-publication/1'), (consumers, 'ruvector.e-means-public-consumers/1')):
        if row.get('schema') != schema or row.get('sourceSha') != source or row.get('artifacts') != candidate['artifacts']:
            raise ValueError('Public evidence identity or archive mismatch')
    for name, digest in candidate['artifacts'].items():
        if release.sha(base / 'public' / name) != digest:
            raise ValueError('Canonical public archive changed')
    fresh = base / 'fresh'
    if release.sha(fresh / 'cargo/src/main.rs') != release.sha(release.ROOT / 'scripts/e-means/public-consumer.rs'):
        raise ValueError('Public Rust consumer differs from exact-source fixture')
    installed = fresh / 'npm/node_modules/@ruvector/e-means'
    manifest = str(fresh / 'cargo/Cargo.toml')
    expected = [
        (['cargo', 'run', '--manifest-path', manifest], None),
        (['cargo', 'metadata', '--quiet', '--locked', '--format-version', '1', '--manifest-path', manifest], None),
        (['npm', 'install', '--ignore-scripts', '--workspaces=false', '--no-audit', '--no-fund'], None),
        (['node', '--test', *[str(installed / 'test' / p.name) for p in sorted((base / 'candidate/harness/test').glob('*.test.mjs'))]], 'node'),
        (['node', str(fresh / 'npm/node_modules/typescript/bin/tsc'), '--noEmit', '-p', str(installed / 'test/tsconfig.json')], None),
        (['node', str(installed / 'test/browser-consumer.mjs')], 'browser'),
    ]
    commands = consumers.get('commands', [])
    if len(commands) != len(expected):
        raise ValueError('Missing required fresh public consumer commands')
    logs = []
    for index, (command, (args, kind)) in enumerate(zip(commands, expected)):
        path = fresh / f'command-{index}.log'
        if (command.get('args') != args or type(command.get('exitCode')) is not int or command['exitCode'] != 0
                or command.get('log') != path.name or release.sha(path) != command.get('logSha256')):
            raise ValueError('Public consumer command or owned log changed')
        output = path.read_text(errors='replace')
        derived = release.evidence(output, kind)
        claims = {key: command[key] for key in ('testsPassed', 'browserPassed') if key in command}
        if claims != derived or any(type(claims[key]) is not type(value) for key, value in derived.items()):
            raise ValueError('Public consumer claims differ from actual logs')
        logs.append(output)
    checks = re.findall(r'^E_MEANS_PUBLIC_CHECKS=(\d+)$', logs[0], re.M)
    if len(checks) != 1 or int(checks[0]) < 10 or type(consumers.get('rustChecks')) is not int or consumers['rustChecks'] != int(checks[0]):
        raise ValueError('Missing actual public Rust numerical checks')
    metadata = json.loads(logs[1])
    resolved = [p for p in metadata['packages'] if (p.get('name'), p.get('version')) == (release.CRATE, release.VERSION)]
    lock = tomllib.loads((fresh / 'cargo/Cargo.lock').read_text())
    locked = [p for p in lock['package'] if (p.get('name'), p.get('version')) == (release.CRATE, release.VERSION)]
    registry = 'registry+https://github.com/rust-lang/crates.io-index'
    if (len(resolved) != 1 or resolved[0].get('source') != registry or len(locked) != 1
            or locked[0].get('source') != registry or locked[0].get('checksum') != candidate['artifacts'][f'{release.CRATE}-{release.VERSION}.crate']):
        raise ValueError('Fresh public Rust consumer did not resolve sealed canonical registry bytes')
    # Recheck canonical registries immediately before exposing the release.
    for name, public in ((f'{release.CRATE}-{release.VERSION}.crate', release.registry.downloaded(release.CRATE, release.VERSION)),
                         (f'ruvector-e-means-{release.VERSION}.tgz', release.npm_public())):
        if public is None or hashlib.sha256(public).hexdigest() != candidate['artifacts'][name]:
            raise ValueError('Canonical registry archive unavailable or differs from candidate')
    return consumers


def assets(base, source):
    candidate = release.validate_candidate(base / 'candidate', source)
    validate_ci(base, source)
    consumers = validate_public(base, source, candidate)
    folder = base / 'github-release-assets'
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    inputs = {name: base / 'public' / name for name in candidate['artifacts']}
    inputs.update({'candidate.json': base / 'candidate/candidate.json', 'FILES.json': base / 'candidate/FILES.json',
                   'ci.json': base / 'ci.json', 'publication.json': base / 'publication.json',
                   'public-consumers.json': base / 'public-consumers.json', 'candidate-artifact.zip': base / 'candidate.zip',
                   'public-Cargo.lock': base / 'fresh/cargo/Cargo.lock'})
    inputs.update({'candidate-' + row['log']: base / 'candidate' / row['log'] for row in candidate['commands']})
    inputs.update({'public-' + row['log']: base / 'fresh' / row['log'] for row in consumers['commands']})
    for name, path in inputs.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing regular release evidence asset: ' + name)
        shutil.copyfile(path, folder / name)
    manifest = {name: release.sha(folder / name) for name in inputs}
    (folder / 'release-assets.json').write_text(json.dumps({'schema': 'ruvector.e-means-github-assets/1',
        'sourceSha': source, 'tag': TAG, 'assets': manifest}, indent=2, sort_keys=True) + '\n')
    manifest['release-assets.json'] = release.sha(folder / 'release-assets.json')
    (folder / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(manifest.items())))
    manifest['SHA256SUMS'] = release.sha(folder / 'SHA256SUMS')
    return folder, manifest


def verify_assets(row, folder, manifest, complete=True):
    remote = inventory(f'repos/{release.REPO}/releases/{row["id"]}/assets')
    names = [asset.get('name') for asset in remote]
    if len(names) != len(set(names)) or not set(names).issubset(manifest) or (complete and set(names) != set(manifest)):
        raise ValueError('Conflicting or incomplete GitHub release asset inventory')
    with tempfile.TemporaryDirectory(prefix='e-means-release-download-') as temporary:
        for asset in remote:
            name = asset['name']
            if asset.get('state') != 'uploaded' or asset.get('size') != (folder / name).stat().st_size:
                raise ValueError('GitHub asset upload is incomplete')
            path = Path(temporary) / name
            with path.open('xb') as stream:
                subprocess.run(['gh', 'api', f'repos/{release.REPO}/releases/assets/{asset["id"]}',
                                '-H', 'Accept: application/octet-stream'], stdout=stream, check=True)
            if release.sha(path) != manifest[name] or (asset.get('digest') and asset['digest'] != 'sha256:' + manifest[name]):
                raise ValueError('Conflicting GitHub release asset bytes; never overwrite')
    return set(names)


def publish(source, base):
    guard(source)
    folder, manifest = assets(base, source)
    row = existing_release()
    current = tag_source()
    if current is not None and current != source:
        raise ValueError('Conflicting immutable release tag; never overwrite')
    if row is not None and (type(row.get('draft')) is not bool or current is None or row.get('target_commitish') != source or row.get('name') != TITLE):
        raise ValueError('Existing release does not identify the accepted immutable source')
    if row is None:
        ensure_tag(source)
        notes = base / 'github-release-notes.md'
        notes.write_text(f'Shared e-means 0.1.0 from `{source}`.\n\n'
            'Canonical public Rust and npm archives, actual source-bound qualification logs, '
            'fresh public Cargo/Node/Worker/Chromium consumer evidence and the original sealed candidate are attached.\n')
        guard(source)
        gh('release', 'create', TAG, '--repo', release.REPO, '--target', source, '--verify-tag', '--draft',
           '--latest=false', '--title', TITLE, '--notes-file', str(notes))
        row = existing_release()
        if row is None or row.get('draft') is not True or row.get('target_commitish') != source:
            raise ValueError('Created draft identity mismatch')
    existing = verify_assets(row, folder, manifest, complete=row.get('draft') is not True)
    if row.get('draft') is True:
        for name in sorted(set(manifest) - existing):
            guard(source)
            assert_tag(source)
            gh('release', 'upload', TAG, str(folder / name), '--repo', release.REPO)
        verify_assets(row, folder, manifest)
        assert_tag(source)
        guard(source)
        gh('release', 'edit', TAG, '--repo', release.REPO, '--draft=false')
        published = existing_release()
        if published is None or published.get('id') != row['id'] or published.get('draft') is not False:
            raise ValueError('GitHub release did not become public')
        row = published
        status = 'published-and-byte-verified'
    else:
        status = 'already-published-and-byte-verified'
    assert_tag(source)
    verify_assets(row, folder, manifest)
    final_source = tag_source()
    if final_source != source:
        raise ValueError('Release tag changed during final asset verification')
    report = {'schema': 'ruvector.e-means-github-publication/1', 'sourceSha': source,
              'tag': TAG, 'tagSourceSha': final_source, 'status': status,
              'url': f'https://github.com/{release.REPO}/releases/tag/{TAG}', 'assets': manifest}
    (base / 'github-release.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    publish(args.expected_sha, args.directory.resolve())


if __name__ == '__main__':
    main()
