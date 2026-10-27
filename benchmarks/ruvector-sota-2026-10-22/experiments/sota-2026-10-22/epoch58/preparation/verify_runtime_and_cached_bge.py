"""Fresh source-pinned runtime/asset hashing; no model, tokenizer or native initialization."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,time
W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch58-db7b9b21f4a1');R=W/'project'
V=Path('/tmp/ruvector-sota-2026-10-22/epoch33-0522a2c521c9/runtime/.venv')
N=Path('/tmp/ruvector-sota-2026-10-22/epoch38-7c50ba99848f/project')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ref(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def checked(p,size,digest):
 assert p.is_file()and not any(x.is_symlink()for x in[p,*p.parents]);assert p.stat().st_size==size and sha(p)==digest;return ref(p)
def main():
 assert not sys.flags.optimize and os.sched_getaffinity(0)=={0,1}
 f=W/'guardian26/owner_fence_v2.py';checked(f,3376,'5b56fa209a1d947d1bbabe31e5e4ed3d52e8dc9d2258e76fa8577c36b459d693')
 s=importlib.util.spec_from_file_location('approved_owner26',f);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.verify_owner(m.EXPECTED_OWNER)
 start=time.monotonic();p=R/'experiments/sota-2026-10-22/epoch26/preparation/runtime-members/python-wheel-members.json';manifest=checked(p,4640799,'87254bbf6ec2aba6d08a4b55ff189faf29f78e1b1dd278fe8ffef9a8cc5523b9');rows=json.loads(p.read_text());assert len(rows)==17498
 total=0
 for i,row in enumerate(rows):
  if i%256==0:m.verify_owner(m.EXPECTED_OWNER)
  name=Path(row['path']);assert not name.is_absolute()and '..'not in name.parts
  checked(V/name,row['bytes'],row['sha256']);total+=row['bytes']
 origin=R/'experiments/sota-2026-10-22/runtime/epoch47/NATIVE_PYTHON_RUNTIME_ORIGIN_LOCK.json';origin_ref=checked(origin,4800,'137310d3973e23aaaea3e474aee3b545cb3af54008c01a12a010322d20b4769f');o=json.loads(origin.read_text());assets={}
 for row in o['assets']:
  name=Path(row['path']);assert not name.is_absolute()and '..'not in name.parts;assets[row['path']]=checked(N/name,row['bytes'],row['sha256'])
 node=checked(Path('/opt/codex/runtimes/codex-primary-runtime/dependencies/node/bin/node'),125989464,'bc17c508ffeed0ec622934f9b7fa72f8e78da65350e63c3eceb56fa688aa5e12')
 bge=checked(Path('/dev/shm/ruvector-sota-2026-10-22/epoch57-5e426edd8ada/model-assets/model_quantized.onnx'),34014426,'6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4')
 invocation=V/'bin/python';resolved=invocation.resolve(strict=True);python_ref=ref(resolved);venv_cfg=ref(V/'pyvenv.cfg')
 env=dict(os.environ);env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
 code="import json,sys,numpy,onnxruntime,tokenizers;print(json.dumps(dict(python=sys.version.split()[0],numpy=numpy.__version__,onnxruntime=onnxruntime.__version__,tokenizers=tokenizers.__version__)))"
 probe=subprocess.run([str(invocation),'-I','-B','-c',code],env=env,capture_output=True,text=True,timeout=30);assert probe.returncode==0,probe.stderr;versions=json.loads(probe.stdout)
 assert versions==dict(python='3.12.14',numpy='2.2.6',onnxruntime='1.22.1',tokenizers='0.21.4')
 n=subprocess.run([node['path'],'--version'],capture_output=True,text=True,timeout=10);assert n.returncode==0
 m.verify_owner(m.EXPECTED_OWNER)
 result=dict(schema_version=1,epoch=58,checks_passed=True,source_checkpoint='db7b9b21f4a13e5fc3444a8fe40de26583558e2e',owner=m.EXPECTED_OWNER,runtime_directory=str(V),python_invocation=str(invocation),python_resolved_binary=python_ref,python_venv_config=venv_cfg,python_versions=versions,python_member_manifest=manifest,runtime_origin_lock=origin_ref,installed_members_fresh_hash_verified=len(rows),installed_bytes=total,node=node,node_version=n.stdout.strip(),native_asset_root=str(N),native_module_root=str(N/'retrieval/node_modules'),assets=assets,bge_quantized_model=bge,bge_tokenizer=assets['models/bge-small-en-v1.5/tokenizer.json'],CPU_affinity=sorted(os.sched_getaffinity(0)),model_tokenizer_native_initializations=0,learned_NLI_weights_read=False,benchmark_archives_read_or_parsed=False,wall_seconds=time.monotonic()-start)
 raw=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode();p=W/'CURRENT_RUNTIME_AND_CACHED_BGE_VERIFICATION.v1.json';fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 d=os.open(W,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d)
 print(json.dumps({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'checks_passed':True,'wheel_members':len(rows),'native_assets':len(assets),'initializations':0}))
if __name__=='__main__':main()
