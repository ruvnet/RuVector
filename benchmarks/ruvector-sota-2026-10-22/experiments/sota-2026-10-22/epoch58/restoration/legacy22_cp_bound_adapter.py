"""Fixed verify-only metadata-bound bridge; original extractor and hashes stay unchanged."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys
D=Path('/dev/shm/ruvector-e58-data');W=Path('/dev/shm/ruvector-sota-2026-10-22/epoch58-db7b9b21f4a1');K=W/'kit'
assert not sys.flags.optimize and len(sys.argv)==1 and os.sched_getaffinity(0)=={0,1}
p=D/'reconstruct_sources.proposed.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e6632280704fe3d40efc6b3dfee3137370dcddb32a0837cbe7d41f1904e5bb7a'
s=importlib.util.spec_from_file_location('reviewed_e58_source_fence',p);f=importlib.util.module_from_spec(s);s.loader.exec_module(f);f.fence()
p=K/'experiments/sota-2026-10-22/epoch22/publication/restore_archived_artifact_v1.py';b=p.read_bytes()
assert len(b)==11810 and hashlib.sha256(b).hexdigest()=='7691256e21cb201e9b64c64b2f880fd6d0274b562cfb4acfa03c5ce673a4ba8f'
assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()=='a4d326f9c8284cb42c3e7ff9f5bb6b5c252cc290'
assert not any(x.is_symlink()for x in[p,*p.parents])
s=importlib.util.spec_from_file_location('unchanged_cp_pinned_E22_extractor',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert Path(m.__file__).absolute()==p and m.METADATA_LIMIT==2097152
m.METADATA_LIMIT=4194304
result=m.restore(str(K),'experiments/sota-2026-10-22/epoch22/publication/actualarchive.dev.raw.v1.json','8ab3d8ec7856b423e6ff295f8e5ef74fc81dcaa3d3382fa2757e21563e0a99c6',verify_only=True)
f.fence()
print(json.dumps({'original_extractor_unchanged':True,'only_runtime_override':{'METADATA_LIMIT_before':2097152,'METADATA_LIMIT_effective':4194304},'verify_only':True,'actual_model_native_Gold_Git_calls':0,'result':result},sort_keys=True))
