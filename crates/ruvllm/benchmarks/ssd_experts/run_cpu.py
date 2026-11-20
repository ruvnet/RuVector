"""Download the pinned checkpoint, run both cache modes and verify every output ID."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
REV = 'e5b5b3d7c3cc5593196902fd3c23964e891a6ea6'
BASE = 'https://huggingface.co/kshitijthakkar/qwen3.5-moe-0.87B-d0.8B/resolve/' + REV
WEIGHTS = '86443370b22b4ff4ee3862eaf6b153b81722f5f4b08dd01a6094c40793cca591'
CRATE_HASH = '02ccd6ca023a3941caf4bc91f1fd3268d7ce34eab2dbeb5d5773de99718537db'
COUNT = 256

def fetch(url, path, digest):
    if path.exists() and sha(path) == digest:
        return
    tmp = path.with_suffix('.part')
    with urllib.request.urlopen(url, timeout=150) as src, tmp.open('wb') as dest:
        shutil.copyfileobj(src, dest, 1<<20)
    if sha(tmp) != digest:
        raise ValueError('SHA256 mismatch: ' + path.name)
    tmp.replace(path)

def sha(path):
    with path.open('rb') as inp:
        return hashlib.file_digest(inp, 'sha256').hexdigest()

def prepare_model(work):
    model = work/'model'
    model.mkdir(exist_ok=True)
    for name, target, digest in [
        ('model-00001-of-00001.safetensors','model.safetensors',WEIGHTS),
        ('config.json','config.json','7fe62f2f26164a03cc252e601260d8378829732864aa0d1194d818b0fda7f258'),
        ('tokenizer.json','tokenizer.json','5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42')]:
        fetch(BASE+'/'+name,model/target,digest)
    return model

def prepare_engine(work):
    archive = work/'lattice.crate'
    fetch('https://static.crates.io/crates/lattice-inference/lattice-inference-0.11.1.crate',archive,CRATE_HASH)
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter='data')
    return work/'lattice-inference-0.11.1'

def check(condition, message):
    if not condition:
        raise ValueError(message)

def summarize(files):
    ids = None
    result = []
    for mode, path in files.items():
        traces=[json.loads(line) for line in path.read_text().splitlines()]
        check(len(traces)==2,'expected two runs per process')
        for n,r in enumerate(traces):
            answer=r['ids']
            check(len(answer)==COUNT and (ids is None or ids==answer),'token mismatch or short decode')
            ids=answer
            old, new = r['cache_before'],r['cache_after']
            slots=8 if mode=='resident' else 2
            check(len(old)==len(new)==24 and all(a[0]==b[0]==slots for a,b in zip(old,new)), 'incorrect actual slots')
            delta=[sum(b[j]-a[j] for a,b in zip(old,new)) for j in range(1,5)]
            if mode=='paged':
                check(delta[1]>0 and delta[2]>0 and delta[3]==delta[1]*3*1024*400*2, 'miss accounting')
            os0,os1,os2=r['os_before'],r['os_at_first'],r['os_after']
            peak=os2.get('peak_footprint_bytes') or os2['peak_rss_bytes']
            read=os2['process_read_bytes']-os1['process_read_bytes']
            result.append(dict(mode=mode,trial=n,decode_tps=r['decode_tps'],ttft_s=r['ttft_s'],
                peak_bytes=peak,decode_storage_read_bytes=read,decode_storage_bytes_per_token=read/(COUNT-1),
                host_swapout_pages=os2['host_swapouts_pages']-os0['host_swapouts_pages'],
                swap_used_delta=None if 'host_swap_used_bytes' not in os0 else os2['host_swap_used_bytes']-os0['host_swap_used_bytes'],
                hits=delta[0],misses=delta[1],evictions=delta[2],expert_logical_read_bytes=delta[3]))
    return dict(checkpoint_revision=REV,weights_sha256=WEIGHTS,token_count=COUNT,distinct_token_ids=len(set(ids)),
        tokens_sha256=hashlib.sha256(b''.join(i.to_bytes(4,'little') for i in ids)).hexdigest(),
        equal_all_four=True,rows=result)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('work',type=Path)
    args=parser.parse_args()
    work=args.work.resolve()
    work.mkdir(parents=True,exist_ok=True)
    model=prepare_model(work)
    crate=prepare_engine(work)
    subprocess.run(['patch','-p1','-i',str(HERE/'cpu.patch')],cwd=crate,check=True)
    for src,dst in [('cpu_bench.rs','bin/cpu_pager_bench.rs'),('metrics.rs','bin/metrics.rs'),
                    ('pager.rs','model/qwen35/bench_pager.rs'),('source.rs','weights/bench_file.rs')]:
        shutil.copyfile(HERE/src,crate/'src'/dst)
    with (crate/'Cargo.toml').open('a') as cargo:
        cargo.write('\n[[bin]]\nname = "cpu_pager_bench"\npath = "src/bin/cpu_pager_bench.rs"\n')
    env=dict(os.environ,RUVLLM_BENCH_SOURCE='1',CARGO_TARGET_DIR=str(work/'target'),RAYON_NUM_THREADS='4')
    env.pop('RUVLLM_BENCH_SLOTS',None)
    env.pop('RUVLLM_BENCH_PROMPT',None)
    if sys.platform=='darwin':
        subprocess.run(['cc','-O2',str(HERE/'darwin_metrics.c'),'-o',str(work/'stats')],check=True)
        env['RUVLLM_DARWIN_METRICS']=str(work/'stats')
    subprocess.run(['cargo','build','--locked','--release','--no-default-features','--features','std,f16',
                    '--bin','cpu_pager_bench','-j','3'],env=env,cwd=crate,check=True)
    files={}
    for mode in ['resident','paged']:
        files[mode]=work/(mode+'.jsonl')
        setting={} if mode=='resident' else {'RUVLLM_BENCH_SLOTS':'2'}
        with files[mode].open('w') as out:
            subprocess.run([str(work/'target/release/cpu_pager_bench'),str(model),str(COUNT),'2'],
                           env={**env,**setting},stdout=out,check=True)
    report=summarize(files)
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':
    main()
