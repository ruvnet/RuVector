"""E57 publication metadata guard. No dataset, model or scientific entrypoint.

Actual CLI modes are Root-only. A publication receipt grants no SCI authority.
Every returned mutation object is durably captured BEFORE response parsing or
post-call owner checks. Any incomplete/invalid transaction stops; no replay.
"""
import sys
sys.dont_write_bytecode = True
import argparse, base64, copy, fcntl, hashlib, json, os, re, stat, time
from pathlib import Path, PurePosixPath

W = Path('/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada')
R = Path('/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada/project')
BASE = '5e426edd8ada8c0fdcec38ec8807054affc89e3c'
BASE_ROOT = '3d1b5edcf40713720ce3830eb97944c0b6a758e1'
BASE_BENCH = 'f02086b61bd89e483837082950baa75521b93829'
BASE_SCOPE = '92b30fdb3a9a6430b31b594042b8cdbc82cc4422'
KIND = 'REAL_NATIVE_QUANTIZED_CPU_THROUGHPUT_AND_RETRIEVAL_DEV'
REPO = 'ruvnet/RuVector'
BRANCH = 'benchmarks/ruvector-sota-2026-10-22'
MAIN = 'c44d7337c0c5ad2ecd2294178475387facb96516'
SCOPE = 'benchmarks/ruvector-sota-2026-10-22/'
ROOTS = tuple('experiments/sota-2026-10-22/' + p for p in ('epoch57/', 'audit/epoch57/', 'runtime/epoch57/'))
CONTROL = {'CONTINUATION.json', 'CHECKPOINT_MANIFEST.json', 'ACTIVE_RUN.json'}
DENY = {'486888de9602bbb5d817a55f707b4fcfd11c65e0', '3d5a61f9068acf2dabd2ae5c6bd531d9a9b464c5','a68b7ba587332f176f65d0348de3e910aa16e77a'}
OWNER = {'path': '/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada/owner.json', 'bytes': 709,
         'sha256': 'a98d0e355925de7cf2fbe8711dc35afc4c3765e07fe5dfab8bd59f0309f733e5'}
TOOLS = {'blob': 'mcp__codex_apps__github_create_blob', 'tree': 'mcp__codex_apps__github_create_tree',
         'commit': 'mcp__codex_apps__github_create_commit', 'ref': 'mcp__codex_apps__github_update_ref'}

def need(ok, why):
    if not ok:
        raise ValueError(why)

def sha(b): return hashlib.sha256(b).hexdigest()
def git(b): return hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
def encoded(x): return (json.dumps(x, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()

def same(a, b):
    if type(a) is not type(b): return False
    if type(a) is dict: return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if type(a) is list: return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b

def relative(p):
    need(type(p) is str and p and '\\' not in p and not PurePosixPath(p).is_absolute()
         and all(x not in ('', '.', '..') for x in p.split('/')), 'canonical relative path')
    return p

def new_path(p): return any(relative(p).startswith(root) for root in ROOTS)

def body(path):
    p = Path(path)
    need(p.is_absolute() and all(not q.is_symlink() for q in (p, *p.parents)), 'absolute non-symlink metadata')
    a = p.stat(); need(stat.S_ISREG(a.st_mode), 'regular metadata')
    b = p.read_bytes(); z = p.stat()
    sig = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(sig(a) == sig(z) and len(b) == z.st_size, 'stable metadata bytes')
    return b

def unique(items):
    out = {}
    for k, v in items:
        need(k not in out, 'duplicate JSON key'); out[k] = v
    return out

def parse(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def load(p): return parse(body(p))
def ref(p):
    b = body(p); return {'path': str(p), 'bytes': len(b), 'sha256': sha(b)}

def checked(row, project=False):
    need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and type(row['bytes']) is int
         and row['bytes'] >= 0 and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'exact typed3ref')
    p = R / relative(row['path']) if project else Path(row['path'])
    b = body(p); need(len(b) == row['bytes'] and sha(b) == row['sha256'], 'reference bytes/hash')
    return b

def write(p, b):
    p = Path(p); need(p.is_absolute() and all(not q.is_symlink() for q in (p, *p.parents)), 'safe output')
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f: f.write(b); f.flush(); os.fsync(f.fileno())
    d = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(d); os.close(d)

def public_text(path, b):
    relative(path)
    need(Path(path).suffix in {'.py', '.js', '.cjs', '.json', '.jsonl', '.md', '.txt', '.toml', '.yaml', '.yml', '.csv', '.sh', '.rs', '.c', '.cpp', '.h'}, 'public text suffix')
    s = b.decode('utf-8'); need('\0' not in s, 'binary excluded')
    need(not re.search(r'<(?:html|body|head)[\s>]', s, re.I), 'raw page excluded')
    need(not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)', s), 'credential-like content')

def tree_sha(rows):
    rows = sorted(rows, key=lambda r: (r['path'] + ('/' if r['type'] == 'tree' else '')).encode())
    b = b''.join((('40000' if r['type'] == 'tree' else r['mode']) + ' ' + r['path']).encode()
                 + b'\0' + bytes.fromhex(r['sha']) for r in rows)
    return hashlib.sha1(b'tree ' + str(len(b)).encode() + b'\0' + b).hexdigest()

def scope_sha(rows):
    root = {}
    is_leaf = lambda v: type(v.get('sha')) is str and v.get('type') == 'blob'
    for row in rows:
        parts = relative(row['path']).split('/'); d = root
        for p in parts[:-1]:
            need(p not in d or not is_leaf(d[p]), 'file/directory collision'); d = d.setdefault(p, {})
        need(parts[-1] not in d, 'duplicate scope path'); d[parts[-1]] = row
    def build(d):
        return tree_sha([{'path': p, 'mode': '040000', 'type': 'tree', 'sha': build(v)}
                         if not is_leaf(v) else {**v, 'path': p} for p, v in d.items()])
    return build(root)

def owner_fence():
    o = parse(checked(OWNER))
    need(o['epoch'] == 57 and o['generation'] == 23 and o['pid'] == 186899 and o['startticks'] == 44193256,
         'exact guardian23 owner')
    need(o['lock'] == '/tmp/ruvector-sota-2026-10-22/exclusive.work.lock'
         and o['lock_device'] == 33 and o['lock_inode'] == 13626, 'original lock identity')
    need(time.time() < o['expires_unix'] and not (Path('/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada') / 'release.request').exists(), 'owner expired/releasing')
    need(o['kernel_boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
         and o['PID_namespace'] == os.readlink('/proc/self/ns/pid'), 'boot/namespace')
    f = Path('/proc/186899/stat').read_text().rsplit(')', 1)[1].split()
    need(int(f[19]) == 44193256 and f[0] not in ('Z', 'X') and os.sched_getaffinity(186899) == {0, 1}, 'live exact owner')
    a = Path('/proc/186899/fd/3').stat(); z = Path(o['lock']).stat()
    need((a.st_dev, a.st_ino) == (z.st_dev, z.st_ino) == (33, 13626), 'FD3 original inode')
    need(any('FLOCK' in s and 'WRITE' in s and s.split()[4] == '186899'
             and s.split()[5] == '00:21:13626' for s in Path('/proc/locks').read_text().splitlines()), 'kernel FLOCK')
    fd = os.open(o['lock'], os.O_RDWR | os.O_NOFOLLOW)
    try:
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else:
            fcntl.flock(fd, fcntl.LOCK_UN); raise ValueError('scope lock was free')
    finally: os.close(fd)
    return o

def validate_request(kind, a):
    need(type(a) is dict and a.get('repository_full_name') == REPO and kind in TOOLS.values(), 'fixed repository/tool')
    def reject(v):
        if type(v) is dict:
            for k, x in v.items():
                if k in ('sha', 'base_tree_sha', 'tree_sha', 'parent_sha', 'expected_sha'):
                    need(x not in DENY, 'retired object live reference')
                reject(x)
        elif type(v) is list:
            for x in v: reject(x)
    reject(a)
    for k in ('base_tree_sha', 'tree_sha', 'parent_sha', 'expected_sha'):
        if k in a: need(type(a[k]) is str and re.fullmatch('[0-9a-f]{40}', a[k]), 'typed Git object SHA')
    if kind == TOOLS['blob']:
        need(set(a) == {'repository_full_name', 'encoding', 'content'} and a['encoding'] == 'utf-8'
             and type(a['content']) is str and git(a['content'].encode()) not in DENY, 'exact UTF8 public blob')
    elif kind == TOOLS['tree']:
        need(set(a) == {'repository_full_name', 'base_tree_sha', 'tree_elements'}
             and type(a['tree_elements']) is list and 1 <= len(a['tree_elements']) <= 32, 'bounded sparse root tree')
        for e in a['tree_elements']:
            need(set(e) == {'path', 'mode', 'type', 'sha'} and e['path'].startswith(SCOPE)
                 and (relative(e['path'][len(SCOPE):]) in CONTROL or new_path(e['path'][len(SCOPE):]))
                 and e['mode'] == '100644' and e['type'] == 'blob'
                 and re.fullmatch('[0-9a-f]{40}', e['sha']) and e['sha'] not in DENY, 'scoped regular tree update')
    elif kind == TOOLS['commit']:
        need(set(a) == {'repository_full_name', 'tree_sha', 'parent_sha', 'message'}
             and type(a['message']) is str and a['message'], 'sole-parent commit')
    else:
        need(set(a) == {'repository_full_name', 'branch_name', 'sha', 'force', 'expected_sha'}
             and a['branch_name'] == BRANCH and a['force'] is False
             and re.fullmatch('[0-9a-f]{40}', a['sha']) and re.fullmatch('[0-9a-f]{40}', a['expected_sha']), 'nonforce exact-parent CAS')

def unwrap(raw):
    need(type(raw) is dict and not raw.get('transport_error'), 'transport outcome unknown; STOP')
    need('isError' not in raw or type(raw['isError']) is bool, 'typed MCP error flag')
    need(raw.get('isError') is not True, 'MCP error; STOP')
    x = raw.get('structuredContent')
    return parse(x['content']) if type(x) is dict and type(x.get('content')) is str else x

def acknowledge(kind, request, result, expected_sha=None, fresh_ref=None):
    if kind != TOOLS['ref']:
        need(type(result) is dict and type(result.get('sha')) is str
             and re.fullmatch('[0-9a-f]{40}', result['sha']) and result['sha'] not in DENY, 'identifying Git ACK required')
        if expected_sha is not None: need(result['sha'] == expected_sha, 'ACK SHA differs from frozen request')
        return result['sha']
    need(type(fresh_ref) is dict and fresh_ref.get('ref') == 'refs/heads/' + BRANCH
         and fresh_ref.get('object', {}).get('sha') == request['sha'], 'fresh postCAS GET required')
    if type(result) is dict and set(result) == {'success'} and result.get('success') is True:
        need(type(result['success']) is bool, 'bare ACK boolean'); return None
    need(type(result) is dict and result.get('ref') == 'refs/heads/' + BRANCH
         and result.get('object', {}).get('sha') == request['sha'], 'identifying ref ACK differs')
    return request['sha']

def journal_path(): return W / 'publication/mutation-journal.jsonl'
def journal():
    p = journal_path(); return [parse(line) for line in body(p).splitlines()] if p.exists() else []

def append(row):
    p = journal_path(); p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'ab') as f: f.write(encoded(row).replace(b'\n', b'') + b'\n'); f.flush(); os.fsync(f.fileno())
    d = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(d); os.close(d)

def clean(rows):
    need(len(rows) % 2 == 0, 'outstanding mutation intent; no replay')
    for i in range(0, len(rows), 2):
        a, b = rows[i:i+2]
        need(a['phase'] == 'intent' and b['phase'] == 'response' and a['id'] == b['id']
             and a['kind'] == b['kind'] and b.get('ok') is True, 'invalid/uncertain mutation history; no replay')

def load_stage(a, fence=True, allow_intent=False):
    if fence: owner_fence()
    p = W / ('stage-' + a.phase) / 'metadata.json'; mb = body(p); m = parse(mb)
    approval = parse(checked({'path': str(a.approval), 'bytes': Path(a.approval).stat().st_size, 'sha256': a.approval_sha}))
    for k in ('checks_passed', 'publication_phase_allowed', 'source_and_denylist_review_passed', 'dual_unknown_denylist_reviewed'):
        need(approval.get(k) is True, 'independent publication approval: ' + k)
    need(m['epoch'] == 57 and m['phase'] == a.phase and m['kind'] == KIND and m['source_checkpoint'] == BASE
         and same(m['owner'], OWNER) and approval.get('epoch') == 57 and approval.get('phase') == a.phase
         and approval.get('stage_metadata_sha256') == sha(mb)
         and approval.get('expected_final_root_sha') == m['expected_final_root']
         and approval.get('source_parent') == m['source_parent'] and approval.get('owner_sha256') == OWNER['sha256']
         and approval.get('source_freeze_sha256') == m['source_freeze']['sha256'], 'exact phase/owner/tree/freeze approval')
    need(m['forbidden_Git_objects'] == sorted(DENY) and approval.get('forbidden_Git_objects') == sorted(DENY)
         and approval.get('original_blob5_execution') == 'UNKNOWN_PERMANENTLY', 'permanent dual uncertainty')
    checked(m['source_freeze'], True); checked(m['plan'], True)
    sf = parse(checked(m['source_freeze'], True))
    for row in sf['files']: checked(row, True)
    frozen = {r['path']: r for r in sf['files']}
    for name, row in m['transport_sources'].items():
        need(row == frozen.get(row['path']), 'transport is source-frozen: ' + name)
        checked(row, True)
    need(sha(body(Path(__file__).resolve())) == m['transport_sources']['guard']['sha256'], 'operational guard matches frozen source')
    leafraw = body(p.parent / m['expected_leaves']['file'])
    need(sha(leafraw) == m['expected_leaves']['sha256'] and len(leafraw) == m['expected_leaves']['bytes'], 'expected leafmap pin')
    rows = journal(); prefix = m['mutation_ID_prefix']
    need(prefix == 'e57-' + a.phase + '-g23', 'new scoped phase prefix')
    if a.phase == 'source':
        need(m['source_parent'] == BASE and all(r['publication_phase'] == 'source' for r in rows), 'fresh SOURCE history')
        current = rows
    else:
        sr = load(W / 'publication/publication-source-result.json')
        source_meta = load(W / 'stage-source/metadata.json')
        need(sr['commit'] == m['source_parent'] and sr['parent'] == BASE and sr['verified_actual_full_tree'] is True
             and sr['stage_metadata_sha256'] == sha(body(W / 'stage-source/metadata.json')), 'FINAL actual SOURCE receipt')
        source_rows = [r for r in rows if r['publication_phase'] == 'source']; clean(source_rows)
        need([r['id'] for r in source_rows[::2]] == source_meta['mutation_ID_order'], 'SOURCE phase complete first')
        need(rows[:len(source_rows)] == source_rows, 'SOURCE precedes FINAL')
        current = rows[len(source_rows):]
    need(all(r['generation'] == 23 and r['publication_phase'] == a.phase and r['owner_sha256'] == OWNER['sha256'] for r in current), 'phase/generation/owner history')
    complete = current[:-1] if allow_intent else current
    clean(complete)
    need([r['id'] for r in complete[::2]] == m['mutation_ID_order'][:len(complete)//2], 'exact completed prefix')
    if allow_intent:
        need(len(current) % 2 == 1 and current[-1]['phase'] == 'intent'
             and current[-1]['id'] == a.id and current[-1]['kind'] == a.kind
             and m['mutation_ID_order'][len(complete)//2] == a.id, 'only own outstanding next intent')
    return p.parent, m, current

def request_for(stage, m, ident):
    slot = next((r for r in m['requests'] if r['id'] == ident), None); need(slot is not None, 'known stage request ID')
    b = body(stage / slot['file']); need(len(b) == slot['bytes'] and sha(b) == slot['sha256'], 'request file pin')
    args = parse(b)
    if slot['kind'] == TOOLS['ref']:
        proof = load(W / 'publication' / (m['phase'] + '-commit-verified.json'))
        need(proof['tree_sha'] == m['expected_final_root'] and proof['parent'] == m['source_parent'], 'fresh sole-parent commit proof')
        args['sha'] = proof['commit']
    validate_request(slot['kind'], args)
    return slot, args

def main(a):
    # Capture is deliberately before ANY owner fence or response parsing.
    if a.mode == 'capture-raw':
        need(re.fullmatch('e57-(source|final)-g23-[a-z0-9-]+', a.id or ''), 'current raw ID')
        write(W / 'publication/requests' / (a.id + '.raw-tool-return.json'), base64.b64decode(a.payload, validate=True))
        return {'raw_durable': True, 'owner_checked_after_capture': False}
    if a.mode == 'capture-proof':
        need(a.proof in ('commit', 'ref'), 'proof kind')
        write(W / 'publication' / (a.phase + '-' + a.proof + '-readback.raw.json'), base64.b64decode(a.payload, validate=True))
        return {'proof_durable': True}
    if a.mode == 'owner': owner_fence(); return {'owner_held': True}
    stage, m, rows = load_stage(a)
    if a.mode == 'check': return {'checks_passed': True, 'completed': len(rows)//2}
    if a.mode == 'verify-commit':
        need(rows and rows[-1]['kind'] == TOOLS['commit'], 'commit ACK precedes proof')
        commit = rows[-1]['ack_sha']; proof = unwrap(load(W / 'publication' / (a.phase + '-commit-readback.raw.json')))
        need(proof['sha'] == commit and proof['tree']['sha'] == m['expected_final_root']
             and len(proof['parents']) == 1 and proof['parents'][0]['sha'] == m['source_parent'], 'fresh sole-parent exact-tree commit')
        write(W / 'publication' / (a.phase + '-commit-verified.json'), encoded({'commit': commit, 'tree_sha': m['expected_final_root'], 'parent': m['source_parent']}))
        return {'commit': commit, 'verified': True}
    need(a.id == m['mutation_ID_order'][len(rows)//2], 'next nonreplayed mutation only')
    slot, request = request_for(stage, m, a.id)
    need(a.kind == slot['kind'], 'fixed stage tool kind')
    rp = W / 'publication/requests' / (a.id + '.json')
    if a.mode == 'reserve': write(rp, encoded(request)); return {'request': ref(rp)}
    need(same(load(rp), request), 'concrete request identity')
    record = {'generation': 23, 'publication_phase': a.phase, 'id': a.id, 'kind': a.kind,
              'owner_sha256': OWNER['sha256'], 'time': time.time(), 'request': ref(rp)}
    if a.mode == 'intent':
        need(not any(r['id'] == a.id for r in journal()), 'already attempted ID')
        append(dict(record, phase='intent')); return {'intent_durable': True}
    raise ValueError('response uses dedicated response_entry')

def response_entry(a):
    stage, m, current = load_stage(a, allow_intent=True); intent = current[-1]
    slot, expected_request = request_for(stage, m, a.id)
    need(slot['kind'] == a.kind, 'response tool is frozen')
    rawpath = W / 'publication/requests' / (a.id + '.raw-tool-return.json'); raw = load(rawpath)
    request_path = W / 'publication/requests' / (a.id + '.json'); request = load(request_path)
    need(same(request, expected_request) and same(ref(request_path), intent['request']), 'intent/request exact identity')
    validate_request(a.kind, request)
    fresh = unwrap(load(W / 'publication' / (a.phase + '-ref-readback.raw.json'))) if a.kind == TOOLS['ref'] else None
    try:
        result = unwrap(raw); ack = acknowledge(a.kind, request, result, slot.get('expected_sha'), fresh)
    except Exception as error:
        append(dict(intent, phase='response', ok=False, raw=ref(rawpath), error_type=type(error).__name__))
        raise
    append(dict(intent, phase='response', ok=True, raw=ref(rawpath), result=result, ack_sha=ack,
                fresh_ref=fresh if a.kind == TOOLS['ref'] else None))
    return {'result': result, 'ack_sha': ack}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['check', 'owner', 'reserve', 'intent', 'capture-raw', 'capture-proof', 'response', 'verify-commit'])
    p.add_argument('--phase', choices=['source', 'final'], required=True); p.add_argument('--approval', type=Path); p.add_argument('--approval-sha'); p.add_argument('--id'); p.add_argument('--kind'); p.add_argument('--payload'); p.add_argument('--proof')
    a = p.parse_args(); print(json.dumps(response_entry(a) if a.mode == 'response' else main(a), sort_keys=True))
