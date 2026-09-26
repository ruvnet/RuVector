#!/usr/bin/env python3
"""Paired firmware trials with explicit execution provenance and fixed gates."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from transport import Device,stable
from benchmark_qemu import Firmware,BUILD
from e2e import compare

def load_manifest(path):
    path=Path(path).resolve();manifest=json.loads(path.read_text())
    for entry in manifest['variants'].values():
        if 'host_binary' in entry and not Path(entry['host_binary']).is_absolute():
            entry['host_binary']=str(path.parent/entry['host_binary'])
        for image in entry['images'].values():
            if not Path(image['file']).is_absolute():image['file']=str(path.parent/image['file'])
    return manifest

def stage_summary(answers):
    result={}
    for stage in ('capture_us','parse_us','preprocess_us','inference_us'):
        values=sorted(a[stage] for a in answers if a.get(stage) is not None)
        result[stage]=None if not values else {
            'samples':len(values),'mean':statistics.mean(values),'p50':statistics.median(values),
            'p95':values[(len(values)*95+99)//100-1],'p99':values[(len(values)*99+99)//100-1],'max':values[-1]}
    return result

def retention(rounds,metric,execution,correctness):
    ratios=[]
    for r in rounds:
        a=r['baseline']['profile'][metric];b=r['candidate']['profile'][metric]
        if a<=0 or b<=0:raise ValueError('timing resolution insufficient for selected metric')
        ratios.append(a/b)
    rng=random.Random(719)
    boot=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(2000))
    median=statistics.median(ratios);lower=boot[50]
    return {'metric':metric,'paired_speedups':ratios,'median_speedup':median,
            'bootstrap_95_lower':lower,'minimum_rounds':11,'required_speedup':1.10,
            'correctness_pass':correctness,'physical_execution':execution=='physical',
            'retain':execution=='physical' and len(rounds)>=11 and correctness and median>=1.10 and lower>1.0}

def verify(meta,manifest,variant,execution,chip):
    if not meta.get('selftest_pass'):raise ValueError('boot selftest failed')
    if meta.get('model_sha256')!=manifest['model_sha256']:raise ValueError('model digest mismatch')
    if meta.get('kernel_sha256')!=manifest['variants'][variant]['kernel_sha256']:raise ValueError('kernel digest mismatch')
    target='host' if execution=='native' else chip
    if meta.get('target')!=target:raise ValueError('target mismatch')
    if meta.get('dynamic_frequency'):raise ValueError('benchmark requires fixed CPU frequency')
    if execution!='native' and not meta.get('fixed_affinity'):raise ValueError('cycle timing requires fixed task affinity')

def open_device(entry,args,name):
    if args.execution=='native':
        if hashlib.sha256(Path(entry['host_binary']).read_bytes()).hexdigest()!=entry['host_sha256']:
            raise ValueError('native binary digest mismatch')
        return Device(command=[entry['host_binary']])
    image=entry['images'][args.chip]
    if hashlib.sha256(Path(image['file']).read_bytes()).hexdigest()!=image['sha256']:
        raise ValueError('firmware image digest mismatch')
    if args.execution=='emulator':
        if args.chip!='esp32s3' or not args.qemu:raise ValueError('only S3 QEMU is supported')
        BUILD.mkdir(exist_ok=True)
        fw=Firmware(args.qemu,Path(image['file']),name);fw.meta=fw.response();return fw
    if not args.flash or not args.port:raise ValueError('physical trials require --port and --flash')
    subprocess.run([sys.executable,'-m','esptool','--chip',args.chip,'--port',args.port,
                    'write_flash','0x0',image['file']],check=True)
    return Device(port=args.port)

def exercise(device,rows,runs):
    if not device.query('selftest').get('selftest_pass'):raise ValueError('selftest failure')
    device.query('profile 64') # initialize code and libc formatting caches
    answers=[];roundtrips=[]
    for row in rows:
        command=('sensor ' if 'raw' in row else 'infer ')+' '.join(map(str,row.get('raw',row['features'])))
        start=time.perf_counter_ns();answer=device.query(command)
        roundtrips.append((time.perf_counter_ns()-start)/1000)
        if 'error' in answer:raise ValueError(answer)
        answers.append(answer)
    before=device.query('meta')['free_heap']
    profile=device.query('profile '+str(runs))
    energy=device.query('energy 256')
    for row in rows[:16]:device.query(('sensor ' if 'raw' in row else 'infer ')+' '.join(map(str,row.get('raw',row['features']))))
    after=device.query('meta')['free_heap']
    if 'error' in profile or 'error' in energy:raise ValueError('firmware benchmark failed')
    if profile.get('runs')!=runs or energy.get('runs')!=256:raise ValueError('profile count mismatch')
    if profile['cpu_hz']!=device.meta['cpu_hz'] or profile['core']!=device.meta['core']:
        raise ValueError('CPU environment changed during benchmark')
    ordered=sorted(roundtrips)
    reference=rows[0]['answer']
    kind='choice' if 'choice' in reference else 'score' if 'score' in reference else 'noul'
    labels=sorted(reference['probabilities']) if kind=='choice' else []
    parity=compare({'labels':labels,'kind':kind},rows,answers)
    return {'profile':profile,'energy_batch':energy,'heap_before':before,'heap_after':after,
            'replay_stages':stage_summary(answers),
            'reference_parity':parity,
            'heap_stable':after>=before,'deadline_pass':profile['p99_us']<100000,
            'replay_count':len(rows),'answers_sha256':hashlib.sha256(json.dumps([stable(a) for a in answers],sort_keys=True).encode()).hexdigest(),
            'roundtrip_p50_us':statistics.median(ordered),'roundtrip_p99_us':ordered[(len(ordered)*99+99)//100-1]},answers

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('--vectors',type=Path,required=True)
    p.add_argument('--execution',choices=['native','emulator','physical'],required=True)
    p.add_argument('--chip',choices=['esp32s3','esp32c6'],default='esp32s3')
    p.add_argument('--port');p.add_argument('--flash',action='store_true');p.add_argument('--qemu')
    p.add_argument('--rounds',type=int,default=11);p.add_argument('--runs',type=int,default=2048)
    p.add_argument('--limit',type=int,default=1000);p.add_argument('--metric',choices=['mean_us','p99_us','mean_cycles','p99_cycles'],default='mean_us')
    p.add_argument('--output',type=Path,default=Path('build-rig/results.json'))
    a=p.parse_args()
    if not 1<=a.rounds<=100 or not 1<=a.runs<=2048 or a.limit<1:p.error('invalid bounded run count')
    manifest=load_manifest(a.manifest);rows=json.loads(a.vectors.read_text())[:a.limit]
    if not rows:raise ValueError('empty replay set')
    report={'schema':1,'execution':a.execution,'physical_hardware':a.execution=='physical',
        'kernel_flags':manifest.get('kernel_flags','unspecified'),
        'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        'vectors_sha256':hashlib.sha256(a.vectors.read_bytes()).hexdigest(),'model_sha256':manifest['model_sha256'],
        'provenance_note':'Physical execution requires an operator-declared serial rig; target metadata alone is not hardware proof.',
        'rounds':[]}
    for i in range(a.rounds):
        record={};answers={}
        for name in (('baseline','candidate') if i%2==0 else ('candidate','baseline')):
            device=open_device(manifest['variants'][name],a,name)
            try:
                verify(device.meta,manifest,name,a.execution,a.chip)
                data,ans=exercise(device,rows,a.runs);record[name]={**data,'meta':device.meta}
                record[name]['image']=manifest['variants'][name]['images'].get(a.chip)
                answers[name]=ans
            finally:device.close()
        same=all(stable(x)==stable(y) for x,y in zip(answers['baseline'],answers['candidate'],strict=True))
        record['difference_examples']=[{'row':j,'baseline':stable(x),'candidate':stable(y)}
            for j,(x,y) in enumerate(zip(answers['baseline'],answers['candidate'],strict=True)) if stable(x)!=stable(y)][:5]
        for key in ('cpu_hz','core','dims','classes','quant_bits'):
            if record['baseline']['meta'][key]!=record['candidate']['meta'][key]:
                raise ValueError('paired environment mismatch: '+key)
        record['exact_replay_match']=same;report['rounds'].append(record)
        print(f'Pair {i+1}/{a.rounds}: exact decisions {same}',flush=True)
    correct=all(r['exact_replay_match'] and r['baseline']['heap_stable'] and r['candidate']['heap_stable'] and r['candidate']['deadline_pass'] for r in report['rounds'])
    report['acceptance']=retention(report['rounds'],a.metric,a.execution,correct)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fields=[json.dumps(k)+': '+(('[\n'+',\n'.join(json.dumps(r) for r in v)+'\n]') if k=='rounds' else json.dumps(v)) for k,v in report.items()]
    a.output.write_text('{\n'+',\n'.join(fields)+'\n}\n')
    if not correct:raise SystemExit('correctness regression; candidate rejected')
if __name__=='__main__':main()
