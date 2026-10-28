"""Fresh E59 recovery publication guard. No scientific calls or old-controller replay.
Every mutation result is durably captured before parsing; unknown outcomes stop.
All 28 prior attempted blob bodies are forbidden from create_blob in this run.
One fixed independently reviewed capsule transports completed outcomes separately.
"""
import sys
sys.dont_write_bytecode=True
import argparse,base64,copy,fcntl,hashlib,importlib.util,json,os,re,stat,time
from pathlib import Path,PurePosixPath
W=Path('/dev/shm/ruvector-sota-2026-10-22/recovery59-993dce6842bf-20261028')
R=Path('/dev/shm/ruvector-sota-2026-10-22/recovery59-993dce6842bf-20261028/project')
BASE='9b677a91698352bd563f189fee861d6ff2a1ad3a'
SOURCE_PARENT='993dce6842bf602dd87864d00c1621a56eebc613'
KIND='PUBLICATION_RECOVERY_ONLY_E59_ZERO_SCIENCE'
REPO='ruvnet/RuVector'
BRANCH='benchmarks/ruvector-sota-2026-10-22'
SCOPE='benchmarks/ruvector-sota-2026-10-22/'
SCIROOTS=('experiments/sota-2026-10-22/epoch59/', 'experiments/sota-2026-10-22/audit/epoch59/', 'experiments/sota-2026-10-22/runtime/epoch59/')
RECOVERY_ROOT='experiments/sota-2026-10-22/recovery59-20261028/'
CONTROL={'CHECKPOINT_MANIFEST.json', 'ACTIVE_RUN.json', 'CONTINUATION.json'}
DENY={'3d5a61f9068acf2dabd2ae5c6bd531d9a9b464c5', 'a68b7ba587332f176f65d0348de3e910aa16e77a', '486888de9602bbb5d817a55f707b4fcfd11c65e0'}
OWNER={'path': '/dev/shm/ruvector-sota-2026-10-22/recovery59-993dce6842bf-20261028/guardian28/owner.json', 'bytes': 848, 'sha256': '39acf0db29a8dff63e76a550927644fc4170d662152961fc0f244147ea9c069e'}
FENCE_SOURCE={'path': '/dev/shm/ruvector-sota-2026-10-22/recovery59-993dce6842bf-20261028/guardian28/owner_fence.py', 'bytes': 3877, 'sha256': 'f4502591a06b4518d77eecd4f7f0eecee2d1cf5383cf24f8b17e6c5663df3791'}
PRIOR_ATTEMPTED_BLOB_SHAS={'544d81f36f3c760f17dee1e1cd0266b710486dbb', 'dc5c5d9cafc35ea4ca0073f2cc061f2bbe987685', '53f13c3ed4c50c521459f186da22fdf7066007db', '07bf80160f5d797ccb845b4dffffb787e16ffccf', 'd96c80472b484bc858186f9fa730510c82fa6a34', '5a113347c8466d72e346dce8130342a8ac5b89de', 'b4d9cb2700b8c5044f94052e7f525ea570257728', 'ca4064e43ccc9c2dca59e5441f234c190db8a851', '09cc1a05368eb6deb111d81843ff0ee58a19f04f', '238aba760ff2674dd5081a33c68aced78a92a6b9', 'c8a8757521490a4b0c7f6ac2f26024cc428bbaf9', '15c4e0b1a0a7dc22d76ec2531c7a7648440c8123', 'b541dadc9402b4fc781e712a86db238e4b34b9ac', '1089df824a50b6c95f0cd42b4d9c18b16439f809', '784832adc97416932334c075e1edcc1c140d9c4b', '76c0a79db634455f48495d859cee1e6d853185dd', 'cdc3a6ce00c3cac70fc6b83261652207bf7d748b', 'ecbeda0c5ff845ed8960e273bb00f1acd13855df', 'a799498cfe9d9117e47227185e6b28abff47cc53', '4445a0b19239e85e83384cdc312cbfded999d4a6', '2f49331bc33c489c1acb18469c7db40f9248e6d8', '6a74fc4a21c59254bb20258a33db17d8b2f69815', '18269afd275bb4dd0df46598dddc5589aaf149f6', '8e24326bfbefe1a08c78888c96c86d7e3d98107d', '493794e759dbea4d05ad30dbac6db0fc990cdfc7', 'b1af014cf3a57f40bb30d29df9ee1eec3b69ef9a', 'abc2c1ec1da3cd82fa44efb189202f235718c30e', 'f0329e5f3109afa210a978af044ad08065334c74'}
TOOLS={'blob': 'mcp__codex_apps__github_create_blob', 'tree': 'mcp__codex_apps__github_create_tree', 'commit': 'mcp__codex_apps__github_create_commit', 'ref': 'mcp__codex_apps__github_update_ref'}
ROOTS=(RECOVERY_ROOT,)
FENCE_PUBLIC_PATH=RECOVERY_ROOT+'ownership/owner_fence.py'
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
 checked(FENCE_SOURCE)
 spec=importlib.util.spec_from_file_location('recovery59_pinned_guardian28',FENCE_SOURCE['path'])
 need(spec is not None and spec.loader is not None,'pinned owner28 loader')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m.verify_owner(OWNER)

def validate_request(kind, a):
    need(type(a) is dict and a.get('repository_full_name') == REPO and kind in TOOLS.values(), 'fixed repository/tool')
    def reject(v):
        if type(v) is dict:
            for k, x in v.items():
                if k in ('sha', 'base_tree_sha', 'tree_sha', 'parent_sha', 'expected_sha'):
                    need(x not in DENY and x not in PRIOR_ATTEMPTED_BLOB_SHAS, 'retired or prior-attempted object live reference')
                reject(x)
        elif type(v) is list:
            for x in v: reject(x)
    reject(a)
    for k in ('base_tree_sha', 'tree_sha', 'parent_sha', 'expected_sha'):
        if k in a: need(type(a[k]) is str and re.fullmatch('[0-9a-f]{40}', a[k]), 'typed Git object SHA')
    if kind == TOOLS['blob']:
        need(set(a) == {'repository_full_name', 'encoding', 'content'} and a['encoding'] == 'utf-8'
             and type(a['content']) is str and git(a['content'].encode()) not in DENY and git(a['content'].encode()) not in PRIOR_ATTEMPTED_BLOB_SHAS, 'exact UTF8 public blob')
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
             and type(a['message']) is str and a['message'] and a['parent_sha']==SOURCE_PARENT, 'sole SOURCE-parent commit')
    else:
        need(set(a) == {'repository_full_name', 'branch_name', 'sha', 'force', 'expected_sha'}
             and a['branch_name'] == BRANCH and a['force'] is False and a['expected_sha']==SOURCE_PARENT
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
             and re.fullmatch('[0-9a-f]{40}', result['sha']) and result['sha'] not in DENY and result['sha'] not in PRIOR_ATTEMPTED_BLOB_SHAS, 'identifying Git ACK required')
        if expected_sha is not None: need(result['sha'] == expected_sha, 'ACK SHA differs from frozen request')
        return result['sha']
    need(type(fresh_ref) is dict and fresh_ref.get('ref') == 'refs/heads/' + BRANCH
         and fresh_ref.get('object', {}).get('sha') == request['sha'], 'fresh postCAS GET required')
    if type(result) is dict and set(result) == {'success'} and result.get('success') is True:
        need(type(result['success']) is bool, 'bare ACK boolean'); return None
    need(type(result) is dict and result.get('ref') == 'refs/heads/' + BRANCH
         and result.get('object', {}).get('sha') == request['sha'], 'identifying ref ACK differs')
    return request['sha']

def scientific_closure(science_manifest):
 ep=parse(checked(science_manifest,True));need(science_manifest['path']==SCIROOTS[0]+'EPOCH_MANIFEST.json'and ep['epoch']==59 and ep['source_checkpoint']==BASE and ep['no_project_additions_after_manifest']is True,'immutable E59 closure')
 closed={r['path']:r for r in ep['files']};need(len(closed)==160 and len(closed)==len(ep['files']),'160 closed scientific entries')
 closed[science_manifest['path']]=science_manifest
 actual={p.relative_to(R).as_posix():ref(p)for root in SCIROOTS for p in(R/root).rglob('*')if p.is_file()}
 need(set(actual)==set(closed)and len(actual)==161,'exact161 E59 sealed files')
 for p,row in closed.items():
  checked(row,True);need((actual[p]['bytes'],actual[p]['sha256'])==(row['bytes'],row['sha256']),'sealed E59 body changed')
 return ep

def canonical_capsule(envelope_ref,overlay_ref):
 eb=checked(envelope_ref,True);ob=checked(overlay_ref,True)
 need(len(eb)==1264415 and sha(eb)=='4c34f2da154c903a05d6ede3b5c42842c7fa9b7ba7218a460c32747af20188d0'and git(eb)=='7be438edb75e9a09cd27b6e4e4eb81ca835bb142','ONE fixed canonical public capsule encoding')
 need(len(ob)==41715 and sha(ob)=='d62375c852ae08aace29138d44de97a99ff9c3197f7bc635521c2897705d7ac4','fixed archive overlay descriptor')
 e=parse(eb);need(set(e)=={'schema_version','format','archive_bytes','archive_sha256','content_base64'}and e['schema_version']==1 and e['format']=='tar.gz'and type(e['archive_bytes'])is int and e['archive_bytes']==948170 and e['archive_sha256']=='1c43224ad190f7514d750b2a265f12376447cbd21ac24603426579f98996c4e9','fixed capsule semantics')
 raw=base64.b64decode(e['content_base64'],validate=True);need(len(raw)==948170 and sha(raw)==e['archive_sha256'],'exact reviewed archive body')
 o=parse(ob);need(o['epoch']==59 and o['source_commit']==SOURCE_PARENT and o['canonical_scientific_members']==161 and o['SOURCE_members_already_Git_visible']==83 and o['project_additions_AFTER_standard_restore']==78 and o['CP_project_files_includes_overlay78']is False and o['no_scientific_execution_authorized']is True,'archive explicit83/78 separate restoration semantics')
 return o

def journal_path():return W/'publication/mutation-journal.jsonl'
def journal():
 p=journal_path();return[parse(line)for line in body(p).splitlines()]if p.exists()else[]
def append(row):
 p=journal_path();p.parent.mkdir(parents=True,exist_ok=True);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'ab')as f:f.write(encoded(row).replace(b'\n',b'')+b'\n');f.flush();os.fsync(f.fileno())
 d=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d)
def clean(rows):
 need(len(rows)%2==0,'outstanding mutation intent; no replay')
 for i in range(0,len(rows),2):
  a,b=rows[i:i+2];need(a['phase']=='intent'and b['phase']=='response'and a['id']==b['id']and a['kind']==b['kind']and b.get('ok')is True,'invalid or uncertain recovery history; no replay')

def load_stage(a,fence=True,allow_intent=False):
 if fence:owner_fence()
 need(a.phase=='recovery','recovery phase only')
 p=W/'stage-recovery/metadata.json';mb=body(p);m=parse(mb)
 approval=parse(checked({'path':str(a.approval),'bytes':Path(a.approval).stat().st_size,'sha256':a.approval_sha}))
 for k in('checks_passed','publication_phase_allowed','source_and_denylist_review_passed','prior_failed_blob_404_now_history_unknown','all_prior_28_recreation_forbidden','one_fixed_distinct_capsule_transport_only'):
  need(approval.get(k)is True,'independent recovery approval: '+k)
 need(m['epoch']==59 and m['phase']=='recovery'and m['kind']==KIND and m['source_checkpoint']==BASE and m['source_parent']==SOURCE_PARENT and same(m['owner'],OWNER),'exact recovery phase identity')
 need(approval.get('epoch')==59 and approval.get('phase')=='recovery'and approval.get('stage_metadata_sha256')==sha(mb)and approval.get('expected_final_root_sha')==m['expected_final_root']and approval.get('source_parent')==SOURCE_PARENT and approval.get('owner_sha256')==OWNER['sha256']and approval.get('recovery_freeze_sha256')==m['recovery_freeze']['sha256'],'exact recovery approval pins')
 need(m['forbidden_Git_objects']==sorted(DENY)and approval.get('forbidden_Git_objects')==sorted(DENY)and approval.get('original_blob5_execution')=='UNKNOWN_PERMANENTLY','original permanent uncertainty preserved')
 sf=parse(checked(m['scientific_source_freeze'],True));need(sf['epoch']==59 and sf['source_checkpoint']==BASE,'original scientific freeze')
 for row in sf['files']:checked(row,True)
 scientific_closure(m['science_manifest'])
 rf=parse(checked(m['recovery_freeze'],True));need(rf['epoch']==59 and rf['kind']==KIND and rf['source_parent']==SOURCE_PARENT and rf['scientific_calls_authorized']==0,'recovery-only frozen source')
 frozen={r['path']:r for r in rf['files']};need(len(frozen)==len(rf['files'])and len(frozen)>0,'unique recovery source inventory')
 for row in rf['files']:checked(row,True)
 need(frozen.get(FENCE_PUBLIC_PATH)=={'path':FENCE_PUBLIC_PATH,'bytes':FENCE_SOURCE['bytes'],'sha256':FENCE_SOURCE['sha256']},'owner28 fence frozen')
 checked(m['recovery_plan'],True);need(m['recovery_plan']==frozen.get(m['recovery_plan']['path']),'recovery plan frozen')
 canonical_capsule(m['capsule_envelope'],m['archive_overlay']);need(m['capsule_envelope']==frozen.get(m['capsule_envelope']['path'])and m['archive_overlay']==frozen.get(m['archive_overlay']['path']),'canonical capsule and overlay frozen')
 policy=parse(checked(m['recovery_policy'],True));need(policy['checks_passed']is True and policy['read_outcome']=='GITHUB_NOT_FOUND_404_AT_SINGLE_OBSERVATION'and policy['original_mutation_history']=='UNKNOWN_PENDING_INDEPENDENT_RECONCILIATION; current404doesnotproveoriginalcreationhistory'and policy['failed_blob_create_authorized']is False and policy['failed_blob_reuse_authorized']is False and policy['new_alternative_transport_eligible_for_review_only']is True,'404 now is not historical creation ACK or recreate authority');need(m['recovery_policy']==frozen.get(m['recovery_policy']['path']),'independent recovery policy frozen')
 for name,row in m['transport_sources'].items():need(row==frozen.get(row['path']),'transport source frozen '+name);checked(row,True)
 need(sha(body(Path(__file__).resolve()))==m['transport_sources']['guard']['sha256'],'actual guard source matches freeze')
 need(m['mutation_ID_prefix']=='recovery59-g28'and m['prior_attempted_blob_SHAs']==sorted(PRIOR_ATTEMPTED_BLOB_SHAS),'fresh IDs and full prior recreation block')
 leafraw=body(p.parent/m['expected_leaves']['file']);need(sha(leafraw)==m['expected_leaves']['sha256']and len(leafraw)==m['expected_leaves']['bytes'],'expected leaf map pinned')
 need(len({x['GitSHA']for x in m['blobs']})==len(m['blobs'])and not({x['GitSHA']for x in m['blobs']}&PRIOR_ATTEMPTED_BLOB_SHAS),'global blob dedup and no prior attempted recreation')
 need(sum(x['GitSHA']=='7be438edb75e9a09cd27b6e4e4eb81ca835bb142'for x in m['blobs'])==1,'exactly ONE fixed distinct capsule object attempt')
 rows=journal();need(all(r['generation']==28 and r['publication_phase']=='recovery'and r['owner_sha256']==OWNER['sha256']for r in rows),'own generation28 recovery journal only')
 complete=rows[:-1]if allow_intent else rows;clean(complete)
 need([r['id']for r in complete[::2]]==m['mutation_ID_order'][:len(complete)//2],'exact completed recovery prefix')
 if allow_intent:need(len(rows)%2==1 and rows[-1]['phase']=='intent'and rows[-1]['id']==a.id and rows[-1]['kind']==a.kind and m['mutation_ID_order'][len(complete)//2]==a.id,'only own outstanding next intent')
 return p.parent,m,rows

def request_for(stage,m,ident):
 slot=next((r for r in m['requests']if r['id']==ident),None);need(slot is not None,'known recovery requestID')
 b=body(stage/slot['file']);need(len(b)==slot['bytes']and sha(b)==slot['sha256'],'request file pinned');args=parse(b)
 if slot['kind']==TOOLS['ref']:
  proof=load(W/'publication/recovery-commit-verified.json');need(proof['tree_sha']==m['expected_final_root']and proof['parent']==SOURCE_PARENT,'fresh sole-parent exact-tree commit proof');args['sha']=proof['commit']
 validate_request(slot['kind'],args);return slot,args

def main(a):
 if a.mode=='capture-raw':
  need(re.fullmatch('recovery59-g28-[a-z0-9-]+',a.id or''),'fresh recovery raw ID');write(W/'publication/requests'/(a.id+'.raw-tool-return.json'),base64.b64decode(a.payload,validate=True));return{'raw_durable':True,'owner_checked_after_capture':False}
 if a.mode=='capture-proof':
  need(a.proof in('commit','ref'),'proof kind');write(W/'publication'/('recovery-'+a.proof+'-readback.raw.json'),base64.b64decode(a.payload,validate=True));return{'proof_durable':True}
 if a.mode=='owner':owner_fence();return{'owner_held':True}
 stage,m,rows=load_stage(a)
 if a.mode=='check':return{'checks_passed':True,'completed':len(rows)//2}
 if a.mode=='verify-commit':
  need(rows and rows[-1]['kind']==TOOLS['commit'],'commitACKbeforeproof');commit=rows[-1]['ack_sha'];proof=unwrap(load(W/'publication/recovery-commit-readback.raw.json'))
  need(proof['sha']==commit and proof['tree']['sha']==m['expected_final_root']and len(proof['parents'])==1 and proof['parents'][0]['sha']==SOURCE_PARENT,'fresh soleparent exacttree commit')
  write(W/'publication/recovery-commit-verified.json',encoded({'commit':commit,'tree_sha':m['expected_final_root'],'parent':SOURCE_PARENT}));return{'commit':commit,'verified':True}
 need(len(rows)//2<len(m['mutation_ID_order'])and a.id==m['mutation_ID_order'][len(rows)//2],'next nonreplayed mutation only');slot,request=request_for(stage,m,a.id);need(a.kind==slot['kind'],'frozen stage tool kind');rp=W/'publication/requests'/(a.id+'.json')
 if a.mode=='reserve':write(rp,encoded(request));return{'request':ref(rp)}
 need(same(load(rp),request),'concrete request identity');record={'generation':28,'publication_phase':'recovery','id':a.id,'kind':a.kind,'owner_sha256':OWNER['sha256'],'time':time.time(),'request':ref(rp)}
 if a.mode=='intent':need(not any(r['id']==a.id for r in journal()),'alreadyattemptedID');append(dict(record,phase='intent'));return{'intent_durable':True}
 raise ValueError('response uses dedicated response_entry')

def response_entry(a):
 stage,m,current=load_stage(a,allow_intent=True);intent=current[-1];slot,expected_request=request_for(stage,m,a.id);need(slot['kind']==a.kind,'response tool frozen')
 rawpath=W/'publication/requests'/(a.id+'.raw-tool-return.json');raw=load(rawpath);rp=W/'publication/requests'/(a.id+'.json');request=load(rp)
 need(same(request,expected_request)and same(ref(rp),intent['request']),'intent/request exact identity');validate_request(a.kind,request)
 fresh=unwrap(load(W/'publication/recovery-ref-readback.raw.json'))if a.kind==TOOLS['ref']else None
 try:result=unwrap(raw);ack=acknowledge(a.kind,request,result,slot.get('expected_sha'),fresh)
 except Exception as error:append(dict(intent,phase='response',ok=False,raw=ref(rawpath),error_type=type(error).__name__));raise
 append(dict(intent,phase='response',ok=True,raw=ref(rawpath),result=result,ack_sha=ack,fresh_ref=fresh if a.kind==TOOLS['ref']else None));return{'result':result,'ack_sha':ack}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','owner','reserve','intent','capture-raw','capture-proof','response','verify-commit']);p.add_argument('--phase',choices=['recovery'],required=True);p.add_argument('--approval',type=Path);p.add_argument('--approval-sha');p.add_argument('--id');p.add_argument('--kind');p.add_argument('--payload');p.add_argument('--proof');a=p.parse_args();print(json.dumps(response_entry(a)if a.mode=='response'else main(a),sort_keys=True))
