#!/usr/bin/env python3
"""Paired CPU benchmarks against a pinned pre-optimization git revision.

python3 tools/benchmark.py --output tests/evidence/benchmark-host.json
Measures native host execution, not ESP32 silicon. Requires GCC and the baseline
git object (git fetch origin 739a5621043f9b0f0ffc263c46d1d2f7aeb9495f if absent).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'build-bench'
CORE=ROOT/'components/rvdecision/rvdecision.c'
HEADER=CORE.parent/'include'
BASELINE='739a5621043f9b0f0ffc263c46d1d2f7aeb9495f'
PATH='examples/esp32-decision/components/rvdecision/rvdecision.c'

def run(cmd,**kw):return subprocess.run(list(map(str,cmd)),check=True,**kw)
def summarize(case):
    base=[x['baseline_ns'] for x in case['rounds']]; cur=[x['candidate_ns'] for x in case['rounds']]
    speedups=[x/y for x,y in zip(base,cur)]
    case.update(baseline_median_ns=statistics.median(base),candidate_median_ns=statistics.median(cur),
                paired_median_speedup=statistics.median(speedups),paired_min_speedup=min(speedups),
                paired_max_speedup=max(speedups))
    print(case.get('fixture',str((case['dims'],case['classes'],case['bits'],case.get('head'),case.get('negatives')))),
          f'{statistics.median(base):.0f} -> {statistics.median(cur):.0f} ns',f'{statistics.median(speedups):.3f}x',flush=True)
    return case
def compile_pair(ref,profile):
    BUILD.mkdir(exist_ok=True)
    baseline=subprocess.check_output(['git','show',f'{ref}:{PATH}'],cwd=ROOT)
    (BUILD/'baseline.c').write_bytes(baseline)
    common=['gcc','-std=c11','-O3','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror','-I',HEADER]
    if profile!='portable': common+=['-DRD_PAIR_DOT']
    if profile=='esp32s3': common+=['-DRD_LIBM_ROUND']
    symbols=['rd_init','rd_predict','rd_dot_i8','rd_dot_i16']
    run(common+['-D'+s+'='+s.replace('rd_','baseline_') for s in symbols]+['-c',BUILD/'baseline.c','-o',BUILD/'baseline.o'])
    run(common+['-c',CORE,'-o',BUILD/'candidate.o'])
    run(common+[ROOT/'tests/benchmark.c',BUILD/'baseline.o',BUILD/'candidate.o','-lm','-o',BUILD/'benchmark'])
    run(common+[ROOT/'tests/differential.c',BUILD/'baseline.o',BUILD/'candidate.o','-lm','-o',BUILD/'differential'])
    return hashlib.sha256(baseline).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',default=BASELINE)
    p.add_argument('--output',type=Path,default=BUILD/'results.json')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--profile',choices=['portable','esp32s3','esp32c6'],default='portable',
                   help='select kernel compile flags; execution remains native host')
    a=p.parse_args()
    cpu=None
    if hasattr(os,'sched_getaffinity'):
        cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    digest=compile_pair(a.baseline,a.profile)
    parity=json.loads(run([BUILD/'differential'],capture_output=True,text=True).stdout)
    print('Baseline differential:',parity,flush=True)
    if a.verify_only: return
    cases=[]
    for d,k in [(32,3),(128,8),(384,3),(768,16)]:
        for bits in (8,16):
            for head in (0,1):
              for negative in (0,1,2):
                r=run([BUILD/'benchmark',d,k,bits,head,negative],capture_output=True,text=True)
                cases.append(summarize(json.loads(r.stdout)))
    fixtures=[]
    for name in ('probe','wide'):
        header=ROOT/'build-host'/name
        if not (header/'model.h').exists(): raise RuntimeError('Run tools/e2e.py to generate fixture headers first')
        binary=BUILD/f'benchmark-{name}'
        run(['gcc','-std=c11','-O3','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror','-I',HEADER,'-I',header,
             ROOT/'tests/benchmark_model.c',BUILD/'baseline.o',BUILD/'candidate.o','-lm','-o',binary])
        case=json.loads(run([binary],capture_output=True,text=True).stdout); case['fixture']=name
        case['header_sha256']=hashlib.sha256((header/'model.h').read_bytes()).hexdigest()
        fixtures.append(summarize(case))
    result={'execution':'native host; not ESP32 silicon','platform':platform.platform(),'cpu_affinity':cpu,
            'kernel_profile':a.profile,
            'synthetic_seed':17,
            'compiler_flags':'-std=c11 -O3 -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror; profile defines as selected',
            'cpu_model':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                              if line.startswith('model name')),'unknown') if Path('/proc/cpuinfo').exists() else platform.processor(),
            'compiler':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
            'baseline_commit':a.baseline,'baseline_sha256':digest,'candidate_sha256':hashlib.sha256(CORE.read_bytes()).hexdigest(),
            'method':'11 alternating paired rounds, 2048 calls each, 256 warmup calls per arm, CPU affinity; 64 synthetic inputs or 8 fixture inputs',
            'differential':parity,'cases':cases,'fixture_cases':fixtures}
    # One case per line keeps this machine-readable sample archive compact.
    fields=[]
    for key,value in result.items():
        encoded='[\n'+',\n'.join('    '+json.dumps(c) for c in value)+'\n  ]' if key in ('cases','fixture_cases') else json.dumps(value)
        fields.append('  '+json.dumps(key)+': '+encoded)
    a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text('{\n'+',\n'.join(fields)+'\n}\n')
if __name__=='__main__':main()
