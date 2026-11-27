"""Compare every pre-sampling logit; on Linux also verify a genuinely cold page cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_cpu import COUNT, HERE, WEIGHTS, check, sha

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('work',type=Path,help='successful run_cpu.py work directory')
    parser.add_argument('output',type=Path,help='new directory for this diagnostic')
    args=parser.parse_args()
    work=args.work.resolve()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    model=work/'model'
    check(sha(model/'model.safetensors')==WEIGHTS,'checkpoint digest mismatch')
    crate=out/'source'
    shutil.copytree(work/'lattice-inference-0.11.1',crate)
    subprocess.run(['patch','-p1','-i',str(HERE/'diagnostics.patch')],cwd=crate,check=True)
    env=dict(os.environ,CARGO_TARGET_DIR=str(out/'target'),RAYON_NUM_THREADS='4',RUVLLM_BENCH_SOURCE='1')
    for key in ('RUVLLM_BENCH_SLOTS','RUVLLM_BENCH_PROMPT','RUVLLM_COLD_SCRIPT'):
        env.pop(key,None)
    if sys.platform=='darwin': env['RUVLLM_DARWIN_METRICS']=str(work/'stats')
    subprocess.run(['cargo','build','--locked','--release','--no-default-features','--features','std,f16',
                    '--bin','cpu_pager_bench','-j','3'],cwd=crate,env=env,check=True)
    rows={}
    sums={}
    for mode,n in [('resident',1),('paged',2)]:
        extra={'RUVLLM_LOGIT_TRACE':str(out/(mode+'.sha256'))}
        if mode=='paged':
            extra['RUVLLM_BENCH_SLOTS']='2'
            if sys.platform=='linux': extra['RUVLLM_COLD_SCRIPT']=str(HERE/'evict_linux.py')
        with (out/(mode+'.jsonl')).open('w') as trace, (out/(mode+'.err')).open('w') as err:
            subprocess.run([str(out/'target/release/cpu_pager_bench'),str(model),str(COUNT),str(n)],
                           env={**env,**extra},stdout=trace,stderr=err,check=True)
        rows[mode]=[json.loads(l) for l in (out/(mode+'.jsonl')).read_text().splitlines()]
        sums[mode]=(out/(mode+'.sha256')).read_text().splitlines()
    ref=sums['resident']
    check(len(ref)==COUNT and sums['paged']==ref+ref,'logit vector mismatch or incomplete run')
    tokens=rows['resident'][0]['ids']
    check(len(tokens)==COUNT and all(r['ids']==tokens for r in rows['paged']),'token mismatch')
    measured=[]
    for r in rows['paged']:
        n=r['os_after']['process_read_bytes']-r['os_at_first']['process_read_bytes']
        measured.append({'decode_tps_including_hashing':r['decode_tps'],'decode_storage_bytes':n,
                         'storage_bytes_per_decode_interval':n/(COUNT-1)})
    residency=None
    if sys.platform=='linux':
        events=[json.loads(l[10:]) for l in (out/'paged.err').read_text().splitlines() if l.startswith('RESIDENCY ')]
        check(len(events)==1 and events[0]['after']==0,'cold cache was not observed')
        residency=events[0]
    report={'compared_values':COUNT*248320,'distinct_logit_vectors':len(set(ref)),
        'sha256_of_per_step_digests':hashlib.sha256(('\n'.join(ref)+'\n').encode()).hexdigest(),
        'residency':residency,'paged':measured}
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__': main()
