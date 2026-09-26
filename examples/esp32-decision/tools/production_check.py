#!/usr/bin/env python3
"""Verify linked profiling storage and optionally replay S3 production firmware."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from benchmark_qemu import Firmware,BUILD
from transport import stable

def footprint(build):
    build=Path(build).resolve();project=json.loads((build/'project_description.json').read_text())
    gcc=Path(project['c_compiler'])
    if not gcc.name.endswith('gcc'):raise ValueError('unsupported compiler')
    prefix=str(gcc)[:-3]
    elf=build/'ruvector_decision.elf';image=build/'merged-binary.bin'
    component=project['build_component_info']['main']
    headers=[Path(d)/'model.h' if Path(d).is_absolute() else Path(component['dir'])/d/'model.h'
             for d in component['include_dirs']]
    header=next((h for h in headers if h.is_file()),None)
    if header is None:raise ValueError('model header provenance missing')
    match=re.search(r'#define RD_MODEL_HASH "([0-9a-f]{64})"',header.read_text())
    if not match or match[1].encode() not in image.read_bytes():raise ValueError('model digest absent from image')
    symbols=subprocess.check_output([prefix+'nm','-S',str(elf)],text=True)
    arrays={}
    for line in symbols.splitlines():
        parts=line.split()
        if len(parts)==4 and parts[2] in ('b','B') and parts[3] in ('times','cycles'):
            arrays[parts[3]]=int(parts[1],16)
    sections={}
    for line in subprocess.check_output([prefix+'size','-A',str(elf)],text=True).splitlines():
        parts=line.split()
        if len(parts)==3 and parts[0].startswith('.') and parts[1].isdigit():
            sections[parts[0]]=int(parts[1])
    return {'target':project['target'],'model_sha256':match[1],
            'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
            'application_bytes':(build/'ruvector_decision.bin').stat().st_size,
            'elf_sha256':hashlib.sha256(elf.read_bytes()).hexdigest(),
            'profile_arrays':arrays,'profile_buffer_bytes':sum(arrays.values()),'sections':sections}

def replay_pair(full,production,qemu,vectors,limit):
    rows=json.loads(Path(vectors).read_text())[:limit]
    if not rows:raise ValueError('empty replay')
    BUILD.mkdir(exist_ok=True);results={};answers={}
    for name,build in [('full',full),('production',production)]:
        fw=Firmware(qemu,Path(build)/'merged-binary.bin','storage-'+name)
        try:
            meta=fw.response()
            if not meta['selftest_pass'] or meta['target']!='esp32s3':raise ValueError('S3 boot gate failed')
            expected=2048 if name=='full' else 0
            if meta['profile_capacity']!=expected or meta['profile_buffer_bytes']!=8*expected:
                raise ValueError('incorrect compiled capacity')
            profile=fw.query('profile 64')
            if expected and profile.get('runs')!=64:raise ValueError('profile failed')
            if not expected and profile.get('error')!='profile_disabled':raise ValueError('profile not disabled')
            command=lambda r:'sensor '+' '.join(map(str,r['raw']))
            # Newlib's float formatter grows caches for different magnitudes.
            # Warm the entire replay once, then require a second identical
            # pass to keep heap constant. Preserve the initial growth too.
            for row in rows:
                if 'error' in fw.query(command(row)):raise ValueError('warmup input failure')
            if fw.query('energy 16').get('runs')!=16:raise ValueError('energy warmup failed')
            before=fw.query('meta')['free_heap']
            answers[name]=[stable(fw.query(command(r))) for r in rows]
            if any('error' in a for a in answers[name]):raise ValueError('replay input failure')
            if fw.query('energy 256').get('runs')!=256:raise ValueError('energy command failed')
            after=fw.query('meta')['free_heap']
            if before!=after:raise ValueError(f'warmed heap changed: {name} {before} -> {after}')
            results[name]={'meta':meta,'heap_before':before,'heap_after':after,
                           'warmup_rows':len(rows),'initial_heap_growth':meta['free_heap']-before,
                           'replay_rows':len(rows),'answers_sha256':hashlib.sha256(json.dumps(answers[name],sort_keys=True).encode()).hexdigest()}
        finally:fw.close()
    if answers['full']!=answers['production']:raise ValueError('production reply regression')
    for key in ('model_sha256','kernel_sha256','quant_bits','cpu_hz','core'):
        if results['full']['meta'][key]!=results['production']['meta'][key]:raise ValueError('identity mismatch')
    results['exact_replay_match']=True
    results['free_heap_recovered']=results['production']['heap_after']-results['full']['heap_after']
    if results['free_heap_recovered']<16384:raise ValueError('expected heap recovery missing')
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-build',type=Path,required=True);p.add_argument('--production-build',type=Path,required=True)
    p.add_argument('--qemu');p.add_argument('--vectors',type=Path);p.add_argument('--limit',type=int,default=128)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.qemu and (not a.vectors or a.limit<1):p.error('QEMU requires vectors and a positive limit')
    report={'execution':'linked ELF analysis','physical_hardware':False,
            'full':footprint(a.full_build),'production':footprint(a.production_build)}
    if report['full']['profile_buffer_bytes']!=16384 or report['production']['profile_buffer_bytes']!=0:
        raise ValueError('linked profiling storage mismatch')
    if report['full']['target']!=report['production']['target']:raise ValueError('target mismatch')
    if report['full']['model_sha256']!=report['production']['model_sha256']:raise ValueError('model mismatch')
    report['static_bytes_recovered']=16384
    report['application_bytes_recovered']=report['full']['application_bytes']-report['production']['application_bytes']
    if a.qemu:report['emulator']=replay_pair(a.full_build,a.production_build,a.qemu,a.vectors,a.limit)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report':str(a.output),'static_bytes_recovered':16384,'physical_hardware':False}))
if __name__=='__main__':main()
