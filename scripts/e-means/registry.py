# Registry transport adapted from ruvnet/ruv-FANN scripts/publish-candidates.py.
# MIT; exact-archive upload protocol: https://doc.rust-lang.org/cargo/reference/registry-web-api.html#publish
import hashlib, json, tarfile, time, tomllib, urllib.error, urllib.request
USER_AGENT = "ruvector-e-means-release (https://github.com/ruvnet/RuVector)"
VISIBILITY_TIMEOUT = 600
VISIBILITY_DELAYS = (5, 10, 20, 30, 60)

def sha(data):
    return hashlib.sha256(data).hexdigest()


def metadata(archive):
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        manifests = [n for n in names if n.count('/') == 1 and n.endswith('/Cargo.toml')]
        if len(manifests) != 1:
            raise ValueError('Expected exactly one normalized manifest')
        prefix = manifests[0].rsplit('/', 1)[0]
        document = tomllib.loads(tar.extractfile(manifests[0]).read().decode())
        package = document['package']
        if prefix != package['name'] + '-' + package['version']:
            raise ValueError('Archive identity mismatch')
        result = {'name': package['name'], 'vers': package['version'],
                  'deps': [], 'features': document.get('features', {}),
                  'authors': package.get('authors', []), 'keywords': package.get('keywords', []),
                  'categories': package.get('categories', []), 'badges': document.get('badges', {})}
        for key in ['description', 'documentation', 'homepage', 'license', 'repository', 'links']:
            result[key] = package.get(key)
        result['license_file'] = package.get('license-file')
        result['rust_version'] = package.get('rust-version')
        readme = package.get('readme')
        result['readme_file'] = readme if isinstance(readme, str) else None
        result['readme'] = tar.extractfile(prefix + '/' + readme).read().decode() if isinstance(readme, str) else None
        for target, table in [(None, document), *document.get('target', {}).items()]:
            for section, kind in [('dependencies', 'normal'), ('dev-dependencies', 'dev'), ('build-dependencies', 'build')]:
                for name, dep in table.get(section, {}).items():
                    dep = {'version': dep} if isinstance(dep, str) else dep
                    if 'version' not in dep or 'path' in dep or 'git' in dep:
                        raise ValueError('Unpublishable dependency: ' + name)
                    result['deps'].append({'name': dep.get('package', name), 'version_req': dep['version'],
                        'features': dep.get('features', []), 'optional': dep.get('optional', False),
                        'default_features': dep.get('default-features', True), 'target': target, 'kind': kind,
                        'registry': dep.get('registry-index'), 'explicit_name_in_toml': name if 'package' in dep else None})
        return result


def downloaded(name, version):
    url = f'https://static.crates.io/crates/{name}/{name}-{version}.crate'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': USER_AGENT}), timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):
            return None
        raise


def wait_public(name, version, timeout=VISIBILITY_TIMEOUT, sleep=time.sleep, clock=time.monotonic):
    """Return the public crate bytes, or None if static.crates.io still lacks them past the deadline."""
    deadline = clock() + timeout
    attempt = 0
    while True:
        public = downloaded(name, version)
        if public is not None:
            return public
        remaining = deadline - clock()
        if remaining <= 0:
            return None
        sleep(min(VISIBILITY_DELAYS[min(attempt, len(VISIBILITY_DELAYS) - 1)], remaining))
        attempt += 1


def confirm_version_absent(name, version):
    """Static S3 403/404 alone is not authority to treat an immutable version as new."""
    key = name.lower()
    if len(key) <= 2:
        relative = f'{len(key)}/{key}'
    elif len(key) == 3:
        relative = f'3/{key[0]}/{key}'
    else:
        relative = f'{key[:2]}/{key[2:4]}/{key}'
    request = urllib.request.Request('https://index.crates.io/' + relative, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            records = [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return
        raise
    if not records or any(not isinstance(row.get('vers'), str) or row.get('name', '').lower() != key for row in records):
        raise ValueError('Invalid or empty sparse registry index: ' + name)
    if any(row['vers'] == version for row in records):
        raise ValueError(f'Published version {name} {version} exists in the index but its public archive is unavailable; nothing may be uploaded')

