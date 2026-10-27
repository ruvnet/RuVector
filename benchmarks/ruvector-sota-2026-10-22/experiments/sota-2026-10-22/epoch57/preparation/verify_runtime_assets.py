import json,hashlib,os,time
from pathlib import Path
W=Path(__file__).resolve().parent;R=W/'project';started=time.monotonic()
def h(p):
 v=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):v.update(b)
 return v.hexdigest()
def check(row,root=None):
 p=Path(row['path']) if root is None else root/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and h(p)==row['sha256'],str(p)
old=json.loads((R/'experiments/sota-2026-10-22/epoch56/preparation/CURRENT_RUNTIME_AND_TOKENIZER_ASSET_VERIFICATION.v1.json').read_text());manifest=old['python_member_manifest'];check(manifest,R);members=json.loads((R/manifest['path']).read_text());runtime=Path(old['runtime_directory'])
for row in members:check(row,runtime)
for row in old['assets'].values():check(row,None if Path(row['path']).is_absolute() else R)
tokenizer=old['tokenizer_assets']
for row in tokenizer:check(row)
out={**old,'epoch':57,'source_checkpoint':'5e426edd8ada8c0fdcec38ec8807054affc89e3c','wall_seconds':time.monotonic()-started,'installed_members_fresh_hash_verified':len(members),'installed_bytes':sum(x['bytes']for x in members),'tokenizer_assets':tokenizer,'learned_NLI_weights_read_or_loaded':False,'benchmark_archives_read_or_parsed':False,'current_cpu_affinity':sorted(os.sched_getaffinity(0))}
with(W/'CURRENT_RUNTIME_AND_TOKENIZER_ASSET_VERIFICATION.v1.json').open('x')as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'checks_passed':True,'members':len(members),'bytes':out['installed_bytes'],'wall_seconds':out['wall_seconds'],'model_tokenizer_native_initializations':0}))
