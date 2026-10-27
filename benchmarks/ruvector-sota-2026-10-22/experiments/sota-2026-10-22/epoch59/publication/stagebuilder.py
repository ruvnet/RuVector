"""Root-only E59 SOURCE/FINAL metadata staging. No Git or scientific call.

Independent preparation approval precedes this CLI; independent full desired-tree
approval follows it. The resulting publication stage grants no SCI authority.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse, copy, json
import guard as g

def canonical(row):
    g.need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'canonical public3ref')
    g.relative(row['path']); g.checked(row, True); return row

def baseline(row):
    value = g.parse(g.checked(row))
    return g.unwrap(value) if 'structuredContent' in value else value

def typed_ref_shape(row, project=False):
    g.need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}
           and type(row['path']) is str and type(row['bytes']) is int and row['bytes'] >= 0
           and type(row['sha256']) is str and g.re.fullmatch('[0-9a-f]{64}', row['sha256']),
           'phase input exact typed3 reference shape')
    if project: g.relative(row['path'])
    else: g.need(Path(row['path']).is_absolute(), 'absolute phase input reference')

def phase_input_shape(x):
    common = {'ACTIVE', 'CONT_patch', 'CP_phase_record', 'artifacts', 'bench_baseline',
              'commit_message', 'current_CONT', 'current_CP', 'epoch', 'kind', 'owner',
              'parent', 'phase', 'plan', 'root_baseline', 'source_checkpoint',
              'source_freeze', 'source_map', 'transport_sources'}
    g.need(type(x) is dict and x.get('phase') in ('source', 'final'), 'explicit phase input object')
    expected = common | ({'actual_SOURCE_receipt', 'epoch_manifest', 'independent_SOURCE_readback'}
                         if x['phase'] == 'final' else set())
    g.need(set(x) == expected, 'complete exact phase input field contract')
    g.need(type(x['epoch']) is int and x['epoch'] == 59 and x['kind'] == g.KIND
           and type(x['parent']) is str and g.re.fullmatch('[0-9a-f]{40}', x['parent'])
           and x['source_checkpoint'] == g.BASE, 'typed exact phase identity')
    for key in ('bench_baseline', 'current_CONT', 'current_CP', 'owner', 'root_baseline', 'source_map'):
        typed_ref_shape(x[key])
    for key in ('plan', 'source_freeze'): typed_ref_shape(x[key], True)
    if x['phase'] == 'final':
        for key in ('actual_SOURCE_receipt', 'independent_SOURCE_readback'): typed_ref_shape(x[key])
        typed_ref_shape(x['epoch_manifest'], True)
    g.need(type(x['artifacts']) is list and type(x['transport_sources']) is dict
           and set(x['transport_sources']) == {'guard', 'stagebuilder', 'controller'}, 'phase artifact/transport contract')
    for row in x['artifacts']:
        g.need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256', 'local_path'}
               and type(row['local_path']) is str and Path(row['local_path']).is_absolute(), 'phase artifact4 contract')
        typed_ref_shape({k: row[k] for k in ('path', 'bytes', 'sha256')}, True)
    for row in x['transport_sources'].values(): typed_ref_shape(row, True)
    for key in ('ACTIVE', 'CONT_patch', 'CP_phase_record'): g.need(type(x[key]) is dict, 'phase controls are objects')

def independent_readback(row):
    value = g.parse(g.checked(row))
    required = {'checks_passed', 'verified_actual_remote_readback', 'commit', 'root', 'tree_sha',
                'source_plan_sha256', 'source_freeze_sha256'}
    g.need(type(value) is dict and required <= set(value), 'complete independent SOURCE readback contract')
    g.need(value['checks_passed'] is True and value['verified_actual_remote_readback'] is True
           and type(value['root']) is str and g.re.fullmatch('[0-9a-f]{40}', value['root'])
           and value['tree_sha'] == value['root'], 'independent readback explicit equal root/tree_sha aliases')
    return value

def current_rows():
    rows = {}
    for prefix in g.ROOTS:
        root = g.R / prefix
        if not root.exists(): continue
        for p in sorted(root.rglob('*')):
            g.need(not p.is_symlink(), 'public project symlink')
            if p.is_file():
                row = g.ref(p); row['path'] = p.relative_to(g.R).as_posix(); rows[row['path']] = row
    return rows

def layout(leaves, root, bench):
    scope_sha = g.scope_sha(leaves)
    b = copy.deepcopy(bench['tree']); r = copy.deepcopy(root['tree'])
    g.need(sum(x['path'] == 'ruvector-sota-2026-10-22' for x in b) == 1
           and sum(x['path'] == 'benchmarks' for x in r) == 1, 'unique root/bench scope entries')
    next(x for x in b if x['path'] == 'ruvector-sota-2026-10-22')['sha'] = scope_sha
    bench_sha = g.tree_sha(b); next(x for x in r if x['path'] == 'benchmarks')['sha'] = bench_sha
    return g.tree_sha(r), bench_sha, scope_sha

def preserve_controls(oldcont, oldcp, x, newrows):
    phase = x['phase']; patch = x['CONT_patch']
    source_keys = {'epoch59_source', 'epoch59_frozen_study', 'epoch59_data_roles'}
    final_keys = {'epoch59_final', 'rejection_ledger', 'latest_learning', 'next_research_priority',
                  'cumulative_budget_ledger', 'benchmark_refresh', 'publisher_status', 'exposed_development_sets'}
    g.need(type(patch) is dict and set(patch) <= (source_keys if phase == 'source' else final_keys), 'current metadata patch keys only')
    cont = copy.deepcopy(oldcont); cont.update(copy.deepcopy(patch)); active = copy.deepcopy(x['ACTIVE'])
    g.need(active['epoch'] == 59 and active['phase'] == phase and active['source_checkpoint'] == g.BASE
           and active['owner_sha256'] == g.OWNER['sha256'] and active['kind'] == g.KIND
           and active['work_active'] is (phase == 'source') and active['experiments_active'] is False
           and active['OS_lock_observation_scope'] == 'PREPUBLICATION_SNAPSHOT_NOT_CURRENT_REMOTE_LEASE'
           and active['publication_is_not_scientific_authorization'] is True, 'honest current ACTIVE snapshot')
    cont['active_run'] = active
    for key, value in oldcont.items():
        if key not in patch and key != 'active_run': g.need(g.same(cont[key], value), 'old CONT field/type changed')
    for key in ('goal_met', 'official_rank', 'champion', 'unused_confirmation_sets'):
        g.need(g.same(cont[key], oldcont[key]), 'champion/goal/eligibility/unused confirmation preserved')
    if 'cumulative_budget_ledger' in patch:
        before = oldcont['cumulative_budget_ledger']; after = cont['cumulative_budget_ledger']
        g.need(type(before) is dict and type(after) is dict and set(after) - set(before) <= {'epoch59_actual_execution'}, 'one current actual budget row only')
        for key, value in before.items(): g.need(key in after and g.same(after[key], value), 'old cumulative actual/uncertain fact changed')
    if 'exposed_development_sets' in patch:
        before = oldcont['exposed_development_sets']; after = cont['exposed_development_sets']
        g.need(type(before) is type(after), 'DEV history type')
        if type(before) is list: g.need(g.same(after[:len(before)], before), 'old DEV history prefix')
        elif type(before) is dict:
            for key, value in before.items(): g.need(key in after and g.same(after[key], value), 'old DEV record')
        else: g.need(g.same(before, after), 'old DEV history')
    cp = copy.deepcopy(oldcp); key = 'epoch59_' + phase
    g.need(key not in cp, 'new phase record only'); cp[key] = copy.deepcopy(x['CP_phase_record'])
    cp['project_files'] = sorted([*oldcp['project_files'], *newrows], key=lambda row: row['path'])
    g.need(len({r['path'] for r in cp['project_files']}) == len(cp['project_files']), 'CP duplicate path')
    for key, value in oldcp.items():
        if key != 'project_files': g.need(g.same(cp[key], value), 'old CP field/type changed')
    cont['checkpoint_manifest_sha256'] = g.sha(g.encoded(cp))
    return cont, cp, active

def build(a, *, check_only=False):
    g.need(type(check_only) is bool, 'explicit boolean check-only mode')
    g.owner_fence(); raw = g.body(a.input); g.need(g.sha(raw) == a.input_sha, 'phase input SHA')
    x = g.parse(raw); phase_input_shape(x); approval = g.parse(g.body(a.approval))
    g.need(g.sha(g.body(a.approval)) == a.approval_sha and approval.get('checks_passed') is True
           and approval.get('phase_preparation_allowed') is True and approval.get('phase_input_sha256') == a.input_sha,
           'independent exact preparation approval')
    phase = x['phase']; g.need(phase in ('source', 'final') and x['epoch'] == 59 and x['kind'] == g.KIND
                             and x['source_checkpoint'] == g.BASE and g.same(x['owner'], g.OWNER), 'new E59 phase/owner/kind')
    out = Path(a.out); g.need(out == g.W / ('stage-' + phase) and not out.exists(), 'fresh exact operational stage outside R')
    plan = g.parse(g.checked(x['plan'], True)); sf = g.parse(g.checked(x['source_freeze'], True))
    g.need(x['plan']['path'] == g.ROOTS[0] + 'plan.json' and x['source_freeze']['path'] == g.ROOTS[0] + 'SOURCE_FREEZE.json'
           and plan['epoch'] == sf['epoch'] == 59 and plan['kind'] == x['kind']
           and plan['source_checkpoint'] == sf['source_checkpoint'] == g.BASE
           and plan['source_freeze_path'] == x['source_freeze']['path'], 'minimal acyclic frozen plan/source binding')
    frozen = {r['path']: canonical(r) for r in sf['files']}; g.need(len(frozen) == len(sf['files']) and frozen, 'unique source freeze')
    g.need(frozen.get(g.FENCE_PUBLIC_PATH) == {'path': g.FENCE_PUBLIC_PATH, 'bytes': g.FENCE_SOURCE['bytes'], 'sha256': g.FENCE_SOURCE['sha256']}, 'shared owner fence source frozen')
    transport = x['transport_sources']; g.need(set(transport) == {'guard', 'stagebuilder', 'controller'}, 'three exact publication sources')
    for name, row in transport.items():
        canonical(row); g.need(row == frozen.get(row['path']), 'publication source frozen: ' + name)
    g.need(g.sha(g.body(Path(g.__file__).resolve())) == transport['guard']['sha256']
           and g.sha(g.body(Path(__file__).resolve())) == transport['stagebuilder']['sha256'], 'operational helpers match source freeze')
    sm = g.parse(g.checked(x['source_map'])); root = baseline(x['root_baseline']); bench = baseline(x['bench_baseline'])
    g.need(sm['commit'] == x['parent'] and root['sha'] == sm['root'] and root['sha'] == g.tree_sha(root['tree'])
           and bench['sha'] == g.tree_sha(bench['tree']) and not root.get('truncated') and not bench.get('truncated'), 'actual parent baseline trees')
    leaves = sm['files']; oldgit = {r['path']: r for r in leaves}
    g.need(len(oldgit) == len(leaves) and all(r['mode'] == '100644' and r['type'] == 'blob' and type(r['size']) is int
           and r['sha'] not in g.DENY for r in leaves) and g.scope_sha(leaves) == sm['scope_sha'], 'complete nonforbidden typed scope')
    g.need(next(r['sha'] for r in root['tree'] if r['path'] == 'benchmarks') == bench['sha']
           and next(r['sha'] for r in bench['tree'] if r['path'] == 'ruvector-sota-2026-10-22') == sm['scope_sha'], 'root/bench/scope chain')
    cpraw = g.checked(x['current_CP']); cont_raw = g.checked(x['current_CONT']); cp = g.parse(cpraw); cont = g.parse(cont_raw)
    for name, payload in [('CHECKPOINT_MANIFEST.json', cpraw), ('CONTINUATION.json', cont_raw)]:
        g.need(g.git(payload) == oldgit[name]['sha'] and len(payload) == oldgit[name]['size'], 'actual control Git identity')
    oldrows = {r['path']: r for r in cp['project_files']}; g.need(len(oldrows) == len(cp['project_files']), 'unique old CP rows')
    cp_only = set(oldrows) - set(oldgit); g.need(len(cp_only) == 70, 'all70 existing CP-only transports')
    overlay = cont['epoch44_publication_recovery9']['archive_overlay']
    g.need(overlay['CP_project_files_includes_overlay55'] is False and overlay['project_additions_AFTER_standard_restore'] == 55
           and overlay['canonical_members'] == 137, 'E44 separate137 overlay55')
    if phase == 'source':
        g.need(x['parent'] == g.BASE and sm['root'] == g.BASE_ROOT and bench['sha'] == g.BASE_BENCH
               and sm['scope_sha'] == g.BASE_SCOPE and len(oldgit) == 9232 and len(oldrows) == 9225,
               'fresh actual E59 source base')
    else:
        sr = g.parse(g.checked(x['actual_SOURCE_receipt'])); ind = independent_readback(x['independent_SOURCE_readback'])
        g.need(sr['epoch'] == 59 and sr['phase'] == 'source' and sr['parent'] == g.BASE and sr['commit'] == x['parent']
               and sr['verified_actual_full_tree'] is True and sr['root'] == sm['root'] and sr['scope'] == sm['scope_sha'], 'actual successful SOURCE receipt')
        g.need(ind['checks_passed'] is True and ind['verified_actual_remote_readback'] is True
               and ind['commit'] == x['parent'] and ind['tree_sha'] == sm['root']
               and ind['source_plan_sha256'] == x['plan']['sha256'] and ind['source_freeze_sha256'] == x['source_freeze']['sha256'], 'independent actual SOURCE readback')
    payloads = {}; newrows = []
    for row in x['artifacts']:
        p = g.relative(row['path']); g.need(g.new_path(p) and p not in oldgit and p not in oldrows and p not in payloads, 'new current public path only')
        g.need(Path(row['local_path']) == g.R / p, 'canonical actual R artifact body')
        b = g.checked({'path': row['local_path'], 'bytes': row['bytes'], 'sha256': row['sha256']}); g.public_text(p, b)
        payloads[p] = b; newrows.append({k: row[k] for k in ('path', 'bytes', 'sha256')})
    now = current_rows()
    if phase == 'source':
        g.need(set(payloads) == set(now), 'SOURCE exact current public inventory')
        current_frozen = {p for p in frozen if g.new_path(p)}
        g.need(current_frozen == set(payloads) - {x['source_freeze']['path']}, 'all SOURCE artifacts including plan frozen except freeze self')
    else:
        ep = x['epoch_manifest']; epoch = g.parse(g.checked(ep, True)); canonical(ep)
        g.need(ep['path'] == g.ROOTS[0] + 'EPOCH_MANIFEST.json' and epoch['epoch'] == 59
               and epoch['source_checkpoint'] == g.BASE and epoch['no_project_additions_after_manifest'] is True, 'immutable current epoch manifest')
        closed = {r['path']: canonical(r) for r in epoch['files']}
        g.need(len(closed) == len(epoch['files']) and set(closed) | {ep['path']} == set(now), 'all current files + manifest itself closed')
        for p, row in closed.items(): g.need(g.same(row, now[p]), 'manifest exact body hash')
        oldcurrent = {p for p in oldgit if g.new_path(p)}
        g.need(set(payloads) == set(now) - oldcurrent, 'FINAL exact new artifact inventory')
        for p in oldcurrent:
            b = g.body(g.R / p); g.need(g.git(b) == oldgit[p]['sha'] and len(b) == oldgit[p]['size'], 'published SOURCE immutable')
    for p, row in frozen.items():
        if not g.new_path(p):
            g.need(p in oldrows or p in oldgit, 'inherited source must have actual baseline provenance')
            if p in oldrows: g.need(g.same(row, oldrows[p]), 'inherited CP source unchanged')
            if p in oldgit: g.need(g.git(g.body(g.R / p)) == oldgit[p]['sha'], 'inherited Git source unchanged')
    newcont, newcp, active = preserve_controls(cont, cp, x, newrows)
    payloads.update({'CONTINUATION.json': g.encoded(newcont), 'CHECKPOINT_MANIFEST.json': g.encoded(newcp), 'ACTIVE_RUN.json': g.encoded(active)})
    g.need(set(newcp['project_files'][i]['path'] for i in range(len(newcp['project_files']))) - (set(oldgit) | set(payloads)) == cp_only,
           'same70 CP-only set; overlay not added')
    updates = [{'path': p, 'mode': '100644', 'type': 'blob', 'sha': g.git(b), 'size': len(b)} for p, b in sorted(payloads.items())]
    g.need(all(r['sha'] not in g.DENY for r in updates), 'no retired decoded object recreation')
    emitted = {}
    def emit(path, raw):
        path = Path(path)
        g.need(path.is_absolute() and path.is_relative_to(out)
               and all(not q.is_symlink() for q in (path, *path.parents)), 'safe exact stage output')
        name = path.relative_to(out).as_posix()
        g.relative(name)
        g.need(name not in emitted and not path.exists(), 'new nonrepeated stage output only')
        emitted[name] = {'file': name, 'bytes': len(raw), 'sha256': g.sha(raw)}
        if not check_only: g.write(path, raw)
    if not check_only: out.mkdir(mode=0o700)
    requests = []; prefix = 'e59-' + phase + '-g27'
    def request(name, kind, value, expected=None):
        g.validate_request(kind, value); b = g.encoded(value); g.need(len(b) < (8384512 if kind == g.TOOLS['blob'] else 175904), 'tool request transport cap')
        emit(out / (name + '.json'), b); row = {'id': prefix + '-' + name, 'kind': kind, 'file': name + '.json', 'bytes': len(b), 'sha256': g.sha(b)}
        if expected is not None: row['expected_sha'] = expected
        requests.append(row); return row
    blobs = []
    for i, row in enumerate(updates):
        req = request('blob-' + str(i), g.TOOLS['blob'], {'repository_full_name': g.REPO, 'encoding': 'utf-8', 'content': payloads[row['path']].decode()}, row['sha'])
        blobs.append({**req, 'project_path': row['path'], 'payload_bytes': row['size'], 'GitSHA': row['sha']})
    working = copy.deepcopy(oldgit); tree = sm['root']; trees = []
    for i in range(0, len(updates), 32):
        chunk = updates[i:i+32]
        for row in chunk: working[row['path']] = row
        desired_root, desired_bench, desired_scope = layout(list(working.values()), root, bench)
        value = {'repository_full_name': g.REPO, 'base_tree_sha': tree, 'tree_elements': [{k: (g.SCOPE + row[k] if k == 'path' else row[k]) for k in ('path', 'mode', 'type', 'sha')} for row in chunk]}
        trees.append(request('tree-' + str(i//32), g.TOOLS['tree'], value, desired_root)); tree = desired_root
    finalroot, finalbench, finalscope = layout(list(working.values()), root, bench)
    request('commit', g.TOOLS['commit'], {'repository_full_name': g.REPO, 'tree_sha': finalroot, 'parent_sha': x['parent'], 'message': x['commit_message']})
    # The template is not sent. reserve() substitutes only the freshly verified commit ACK.
    template = {'repository_full_name': g.REPO, 'branch_name': g.BRANCH, 'sha': '0' * 40, 'force': False, 'expected_sha': x['parent']}
    request('ref', g.TOOLS['ref'], template)
    leafraw = g.encoded({'scope': g.SCOPE, 'files': sorted(working.values(), key=lambda row: row['path'])}); emit(out / 'expected-leaves.json', leafraw)
    meta = {'schema_version': 1, 'epoch': 59, 'kind': g.KIND, 'phase': phase, 'source_checkpoint': g.BASE,
            'source_parent': x['parent'], 'owner': g.OWNER, 'plan': x['plan'], 'source_freeze': x['source_freeze'], 'transport_sources': transport,
            'phase_input_sha256': a.input_sha, 'expected_source_root': sm['root'], 'expected_source_bench': bench['sha'], 'expected_source_scope': sm['scope_sha'],
            'expected_final_root': finalroot, 'expected_final_bench': finalbench, 'expected_final_scope': finalscope,
            'new_Git_leaves': len(working), 'CP_rows': len(newcp['project_files']), 'CP_only_paths': sorted(cp_only),
            'E44_overlay55_remains_separate': True, 'mutation_ID_prefix': prefix, 'mutation_ID_order': [r['id'] for r in requests],
            'requests': requests, 'blobs': blobs, 'trees': trees, 'expected_leaves': {'file': 'expected-leaves.json', 'bytes': len(leafraw), 'sha256': g.sha(leafraw)},
            'forbidden_Git_objects': sorted(g.DENY), 'original_blob5_execution': 'UNKNOWN_PERMANENTLY', 'publication_only': True, 'scientific_authorization_granted': False}
    emit(out / 'metadata.json', g.encoded(meta))
    if check_only:
        g.need(not out.exists(), 'check-only created no stage directory')
        return {'metadata': meta, 'planned_outputs': sorted(emitted.values(), key=lambda row: row['file'])}
    return meta

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True); p.add_argument('--input-sha', required=True)
    p.add_argument('--approval', type=Path, required=True); p.add_argument('--approval-sha', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args(); result = build(a, check_only=a.check_only)
    if a.check_only:
        m = result['metadata']; row = next(x for x in result['planned_outputs'] if x['file'] == 'metadata.json')
        print(json.dumps({'check_only': True, 'all_real_preparation_checks_passed': True,
                          'stage_metadata_planned': row, 'expected_final_root': m['expected_final_root'],
                          'new_Git_leaves': m['new_Git_leaves'], 'CP_rows': m['CP_rows'],
                          'planned_calls': len(m['mutation_ID_order']), 'planned_output_files': len(result['planned_outputs']),
                          'actual_stage_or_request_or_R_writes': 0, 'actual_Git_or_SCI_calls': 0}, sort_keys=True))
    else:
        m = result
        print(json.dumps({'stage_metadata': g.ref(Path(a.out) / 'metadata.json'), 'expected_final_root': m['expected_final_root'], 'new_Git_leaves': m['new_Git_leaves'], 'CP_rows': m['CP_rows'], 'planned_calls': len(m['mutation_ID_order']), 'actual_Git_or_SCI_calls': 0}, sort_keys=True))
