"""One read-only file-cache hint; no file-byte mutation or library execution."""
from pathlib import Path
import hashlib,json,os,time
W=Path(__file__).resolve().parent
P=Path('/tmp/ruvector-sota-2026-10-22/epoch33-0522a2c521c9/runtime/.venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so')
EXPECTED_BYTES=418117016
EXPECTED_SHA='5d4a2afb0c94f8c2cb44eae9c7d303ebcb64700d49933ff8894b09eea61b3f1d'
OWNER_SHA='a98d0e355925de7cf2fbe8711dc35afc4c3765e07fe5dfab8bd59f0309f733e5'
def put(p,v):
 b=(json.dumps(v,indent=2,sort_keys=True)+'\n').encode()
 with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
def snap():
 m=int(Path('/sys/fs/cgroup/memory.max').read_text());c=int(Path('/sys/fs/cgroup/memory.current').read_text());return{'max_bytes':m,'current_bytes':c,'headroom_bytes':m-c}
def fence():
 b=(W/'owner.json').read_bytes();assert hashlib.sha256(b).hexdigest()==OWNER_SHA;o=json.loads(b);assert o['epoch']==57 and o['generation']==23 and time.time()<o['expires_unix']and not(W/'release.request').exists();p=Path('/proc')/str(o['pid']);f=(p/'stat').read_text().rsplit(')',1)[1].split();assert int(f[19])==o['startticks'];s=(p/'fd/3').stat();assert(s.st_dev,s.st_ino)==(33,13626)and'FLOCK'in(p/'fdinfo/3').read_text()
def signature(s):return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink]
assert sorted(os.sched_getaffinity(0))==[0,1];fence();assert not any(p.is_symlink()for p in[P,*P.parents]);fd=os.open(P,os.O_RDONLY|os.O_NOFOLLOW);before_stat=os.fstat(fd);assert before_stat.st_size==EXPECTED_BYTES;h=hashlib.sha256()
with os.fdopen(os.dup(fd),'rb')as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()==EXPECTED_SHA and signature(os.fstat(fd))==signature(before_stat);fence();before=snap();put(W/'UNUSED_TORCH_CACHE_HINT_ATTEMPT.actual.v1.json',{'epoch':57,'exact_path':str(P),'bytes':EXPECTED_BYTES,'sha256':EXPECTED_SHA,'operation':'posix_fadvise POSIX_FADV_DONTNEED using O_RDONLY FD','file_byte_changes_authorized':False,'before':before,'automatic_retry':False});error=None
try:os.posix_fadvise(fd,0,EXPECTED_BYTES,os.POSIX_FADV_DONTNEED)
except OSError as ex:error={'type':type(ex).__name__,'errno':ex.errno}
finally:
 after_stat=os.fstat(fd);os.close(fd)
assert signature(after_stat)==signature(before_stat)and signature(P.stat())==signature(before_stat);fence();after=snap();result={'epoch':57,'hint_completed':error is None,'error':error,'before':before,'after':after,'file_stat_signature_unchanged':True,'target_bytes':EXPECTED_BYTES,'target_sha256_before_hint':EXPECTED_SHA,'descriptor_was_read_only':True,'file_mutation_or_deletion_calls':0,'model_tokenizer_native_or_library_initializations':0,'paid_API_GPU_calls':0,'cache_hint_scope':'Advisory clean file-page discard only; mapped pages may remain, no global cache drop and no host-exclusive use or sole attribution claim.','automatic_retry':False};put(W/'UNUSED_TORCH_CACHE_HINT.actual.v1.json',result);print(json.dumps(result))
