"""Measure the normal ruvllm Lattice backend with verified live Metal cache slots."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from patch_metal import instrument

BASE = '979c31acc57b970c1a7e0771ed527c1703c137e8'
HERE = Path(__file__).resolve().parent
PROMPT = 'Write a detailed technical guide to how a solid state drive stores and retrieves data. Begin with the physical structure of a NAND cell and continue through the read path.\n\n'

def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)

def check(ok, why):
    if not ok: raise ValueError(why)

def validate(runs, cfg, slots, system):
    check(set(runs)=={'resident','paged'},'both cache modes required')
    experts = cfg['num_experts']
    layers = cfg['num_hidden_layers']
    expected_ids = None
    summary=[]
    for mode,trials in runs.items():
        check(len(trials)==2,'two independent generations required per mode')
        grant=experts if mode=='resident' else slots
        for n,row in enumerate(trials):
            ids=row['ids']
            check(len(ids)==256 and row['captured_callbacks']==256,'short generation or unobservable token timing')
            check(expected_ids is None or ids==expected_ids,'greedy token mismatch')
            expected_ids=ids
            old,new=row['cache_before'],row['cache_after']
            check(len(old)==len(new)==layers and all(a[0]==b[0]==grant for a,b in zip(old,new)), 'effective slots changed')
            missed=sum(b[2]+b[5]-a[2]-a[5] for a,b in zip(old,new))
            evicted=sum(b[3]+b[6]-a[3]-a[6] for a,b in zip(old,new))
            check((missed==0 and evicted==0) if mode=='resident' else (missed>0 and evicted>0), 'reference or paged path was not exercised')
            start,first,last=[row[k] for k in ('os_before','os_at_first','os_after')]
            peak=last['peak_footprint_bytes']
            summary.append(dict(mode=mode,trial=n,actual_slots=grant,misses=missed,evictions=evicted,
                decode_tps=255/row['decode_s'],ttft_s=row['ttft_s'],peak_footprint_bytes=peak,
                storage_read_bytes_per_decode_interval=(last['process_read_bytes']-first['process_read_bytes'])/255,
                swapout_growth_pages=last['host_swapouts_pages']-system[mode]['host_swapouts_pages'],
                swap_used_growth_bytes=last['host_swap_used_bytes']-system[mode]['host_swap_used_bytes']))
    passed=all(r['peak_footprint_bytes']<=16*(1<<30) and r['swapout_growth_pages']==0 and r['swap_used_growth_bytes']<=0 for r in summary if r['mode']=='paged')
    return {'equal_greedy_tokens':256,'distinct_ids':len(set(expected_ids)),'paged_memory_and_swap_pass':passed,'trials':summary}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('repo',type=Path,help='Git checkout containing the pinned RuVector commit')
    p.add_argument('work',type=Path,help='empty dedicated scratch directory')
    p.add_argument('--q4',type=Path,required=True,help='native Lattice Q4 model including config and tokenizer')
    p.add_argument('--kit',type=Path,default=HERE,help='directory containing run_cpu.py and darwin_metrics.c')
    args=p.parse_args()
    check(platform.system()=='Darwin' and platform.machine()=='arm64','Apple Silicon macOS is required')
    kit=args.kit.resolve()
    sys.path.insert(0,str(kit))
    from run_cpu import prepare_engine
    q4=args.q4.resolve()
    cfg=json.loads((q4/'config.json').read_text())
    cfg=cfg.get('text_config',cfg)
    check(any(q4.glob('*.q4')) and (q4/'tokenizer.json').is_file(),'need prepared native Q4 directory')
    total=cfg['num_hidden_layers']*cfg['num_experts']*3*cfg['hidden_size']*cfg['moe_intermediate_size']*2
    program='import Metal; guard let d=MTLCreateSystemDefaultDevice() else { print("no Metal device"); exit(2) }; print(d.recommendedMaxWorkingSetSize)'
    available=int(subprocess.check_output(['swift','-e','import Foundation; '+program],text=True).strip())
    check(total<=int(0.85*available),f'full expert buffers require {total} bytes, Metal cache budget before dense weights {int(0.85*available)}')
    work=args.work.resolve()
    work.mkdir(exist_ok=True,parents=True)
    check(not any(work.iterdir()),'use an empty benchmark working directory')
    repo=work/'RuVector'
    run('git','-C',str(args.repo.resolve()),'worktree','add','--detach',str(repo),BASE)
    crate=prepare_engine(work)
    instrument(repo,crate)
    run('cc','-O2',str(kit/'darwin_metrics.c'),'-o',str(work/'stats'))
    env=dict(os.environ,RUSTUP_TOOLCHAIN='1.93.1',CARGO_TARGET_DIR=str(work/'target'),
        RUVLLM_DARWIN_METRICS=str(work/'stats'),BENCH_GREEDY='1',RUVLLM_PARITY_STREAM_ONLY='1')
    env.pop('RUVLLM_PRIME_ALL',None)
    cargo_patch='patch.crates-io.lattice-inference.path='+json.dumps(str(crate))
    run('cargo','--config',cargo_patch,'build','--release','-p','ruvllm','--no-default-features','--features',
        'async-runtime,lattice','--example','lattice_bench',cwd=repo,env=env)
    binary=work/'target/release/examples/lattice_bench'
    slots=cfg['num_experts_per_tok']
    runs,system={},{}
    for mode,capacity in [('resident',cfg['num_experts']),('paged',slots)]:
        factors={'LATTICE_MOE_EXPERT_CACHE_SLOTS':str(capacity)}
        if mode=='resident': factors['RUVLLM_PRIME_ALL']='1'
        log=work/(mode+'.log')
        system[mode]=json.loads(subprocess.check_output([str(work/'stats'),str(os.getpid())],text=True))
        with log.open('w') as out:
            run(str(binary),'--backend','lattice','--model',str(q4),'--runs','2','--warmup','0','--max-tokens','256',
                '--prompt',PROMPT, env={**env,**factors},stdout=out,stderr=subprocess.STDOUT)
        runs[mode]=[json.loads(l[7:]) for l in log.read_text().splitlines() if l.startswith('PARITY ')]
    report=validate(runs,cfg,slots,system)
    (work/'metal-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    check(report['paged_memory_and_swap_pass'], 'paged memory or host swap criterion failed')

if __name__=='__main__': main()
