"""Root-only exact kit-file twin retirement; no directory, R, model or science mutation."""
from pathlib import Path
import hashlib, json, os, stat, sys, time
W = Path('/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada')
D = Path('/dev/shm/ruvector-e57-data/kit-twin-retirement')
K, R = W / 'kit', W / 'project'
PLAN_SHA = 'efcbdc2c834f902ecc6bb40b4f287f506b08d419463845870bf376ab8ccb9319'
OWNER_SHA = 'a98d0e355925de7cf2fbe8711dc35afc4c3765e07fe5dfab8bd59f0309f733e5'

def sha(p):
    x = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): x.update(b)
    return x.hexdigest()
def ref(p): return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
def read(r):
    p = Path(r['path']); assert ref(p) == r; return json.loads(p.read_bytes())
def sig(p):
    s = p.lstat(); return {'dev': s.st_dev, 'ino': s.st_ino, 'mode': s.st_mode, 'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns, 'nlink': s.st_nlink}
def put(p, value):
    with p.open('x') as f: json.dump(value, f, indent=2, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)
def fence():
    p = W / 'owner.json'; assert sha(p) == OWNER_SHA; o = json.loads(p.read_bytes())
    assert o['epoch'] == 57 and o['generation'] == 23 and o['pid'] == 186899 and o['startticks'] == 44193256
    assert time.time() < o['expires_unix'] and not (W / 'release.request').exists()
    assert o['kernel_boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip() and o['PID_namespace'] == os.readlink('/proc/self/ns/pid')
    z = Path('/proc/186899/stat').read_text().rsplit(')', 1)[1].split(); assert int(z[19]) == 44193256 and z[0] not in ('Z', 'X')
    a, b = Path('/proc/186899/fd/3').stat(), Path(o['lock']).stat(); assert (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino) == (33, 13626)
    assert any('FLOCK' in row and 'WRITE' in row and row.split()[4:6] == ['186899', '00:21:13626'] for row in Path('/proc/locks').read_text().splitlines())
def snapshot():
    c = Path('/sys/fs/cgroup'); current = int((c / 'memory.current').read_text()); cap = int((c / 'memory.max').read_text())
    return {'memory_current_bytes': current, 'memory_max_bytes': cap, 'shared_headroom_bytes': cap - current,
            'shm_free_bytes': os.statvfs('/dev/shm').f_bavail * os.statvfs('/dev/shm').f_frsize}

def main():
    assert sys.flags.optimize == 0 and sorted(os.sched_getaffinity(0)) == [0, 1]
    fence(); pp = D / 'KIT_TWIN_RETIREMENT_PLAN.v1.json'; assert sha(pp) == PLAN_SHA; plan = json.loads(pp.read_bytes())
    assert plan['epoch'] == 57 and plan['kit_root'] == str(K) and plan['preserved_project_root'] == str(R)
    assert plan['delete_exact_individual_kit_files_only'] is True and plan['directory_removal_authorized'] is False
    manifest = read(plan['full_manifest']); rows = manifest['twins']; keep = manifest['preserved_kit_files']; dirs = manifest['unchanged_kit_directory_topology']
    assert len(rows) == plan['kit_twin_files_to_retire'] == 9048 and sum(r['bytes'] for r in rows) == plan['logical_bytes_reclaimable'] == 566927615
    approval_ref = json.loads((W / 'KIT_TWIN_RETIREMENT_APPROVAL_REF.v1.json').read_bytes()); approval = read(approval_ref)
    assert approval['checks_passed'] is True and approval['kit_twin_retirement_allowed'] is True and approval['plan_sha256'] == PLAN_SHA and approval['source_sha256'] == sha(Path(__file__).resolve())
    assert not any(p.is_symlink() for p in (K, R, *K.parents))
    actual = list(K.rglob('*')); assert not any(p.is_symlink() for p in actual)
    assert {str(p) for p in actual if p.is_file()} == {r['kit'] for r in rows} | {r['path'] for r in keep}
    for row in dirs:
        p = Path(row['path']); s = p.lstat(); assert p.is_relative_to(K) and (s.st_dev, s.st_ino, s.st_mode) == (row['dev'], row['ino'], row['mode'])
    for row in rows:
        a, b = Path(row['kit']), Path(row['project_mirror']); name = Path(row['relative_path'])
        assert not name.is_absolute() and '..' not in name.parts and a == K / name and b == R / name
        assert not any(p.is_symlink() for p in (a, b, *a.parents, *b.parents))
        assert sig(a) == row['kit_signature'] and sig(b) == row['project_signature']
        assert (a.stat().st_dev, a.stat().st_ino) != (b.stat().st_dev, b.stat().st_ino)
        assert a.stat().st_size == b.stat().st_size == row['bytes'] and sha(a) == sha(b) == row['sha256']
    for row in keep:
        p = Path(row['path']); assert p.is_relative_to(K) and sig(p) == row['signature'] and sha(p) == row['sha256']
    fence(); before = snapshot(); put(W / 'KIT_TWIN_RETIREMENT_INTENT.actual.v1.json',
        {'epoch': 57, 'plan': ref(pp), 'approval': approval_ref, 'exact_kit_file_count': len(rows), 'full_manifest': plan['full_manifest'],
         'all_R_and_K_only_files_preserved': True, 'before': before, 'Root_sole_deletion_actor': True, 'automatic_retry': False})
    start = time.monotonic()
    for row in rows:
        p = Path(row['kit']); assert sig(p) == row['kit_signature']; p.unlink()
    fence(); assert {str(p) for p in K.rglob('*') if p.is_file()} == {r['path'] for r in keep}
    for row in rows:
        p = Path(row['project_mirror']); assert sig(p) == row['project_signature'] and sha(p) == row['sha256'] and not Path(row['kit']).exists()
    for row in keep:
        p = Path(row['path']); assert sig(p) == row['signature'] and sha(p) == row['sha256']
    for row in dirs:
        p = Path(row['path']); s = p.lstat(); assert (s.st_dev, s.st_ino, s.st_mode) == (row['dev'], row['ino'], row['mode'])
    value = {'schema_version': 1, 'epoch': 57, 'checks_passed': True, 'plan': ref(pp), 'approval': approval_ref,
        'exact_kit_files_retired': len(rows), 'logical_bytes_retired': sum(r['bytes'] for r in rows), 'preserved_kit_files': len(keep),
        'all9048_project_mirrors_and46_kit_only_files_full_hash_preserved': True, 'all8962_CP_bytes_remain_in_R': True,
        'all_kit_directories_preserved': True, 'all_private_metadata_model_runtime_and_R_paths_preserved': True,
        'before': before, 'after': snapshot(), 'wall_seconds': time.monotonic() - start,
        'shared_snapshot_scope': 'Shared snapshots; no sole-cause or guaranteed incremental-memory claim.',
        'scientific_dispatch_authorized': False, 'new_native_model_Gold_paid_GPU_Git_calls': 0}
    put(W / 'KIT_TWIN_RETIREMENT.actual.v1.json', value); print(json.dumps(value))

if __name__ == '__main__': main()
