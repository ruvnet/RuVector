#!/usr/bin/env python3
"""Run real Rust export -> INT8 quantization -> C firmware protocol parity tests.

python3 tools/e2e.py
Optional --port /dev/ttyUSB0 --model probe tests already flashed hardware using
the same held-out vectors. Never flashes or changes hardware automatically.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
from quantize import export

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT/'build-host'
INCLUDE = ROOT/'components/rvdecision/include'
CORE = ROOT/'components/rvdecision/rvdecision.c'

def run(cmd, **kw):
    subprocess.run(list(map(str, cmd)), check=True, cwd=ROOT, **kw)

def compile_host(model, scalar=False, bits=16):
    folder = BUILD/(model if bits == 16 else model+'-int8')
    folder.mkdir(exist_ok=True, parents=True)
    snapshot = json.loads((BUILD/f'fixtures/{model}.json').read_text())
    rows = json.loads((BUILD/f'fixtures/{model}.vectors.json').read_text())
    report = export(snapshot, folder/'model.h', rows, bits)
    binary = folder/('firmware-scalar' if scalar else 'firmware')
    cmd = ['gcc','-std=c11','-O3','-fno-fast-math','-Wall','-Wextra','-Werror',
           '-I',folder,'-I',INCLUDE,'-I',ROOT/'main',CORE,ROOT/'main/app.c',ROOT/'tests/host_main.c','-lm','-o',binary]
    if scalar: cmd.insert(1,'-DRD_SCALAR_DOT')
    run(cmd)
    return binary, snapshot, rows, report

def compare(m, rows, answers):
    max_error, matching, accepted, accepted_same, source_accepted = 0.0, 0, 0, 0, 0
    for row, actual in zip(rows, answers, strict=True):
        if 'error' in actual: raise AssertionError(actual)
        ref = row['answer']
        index = m['labels'].index(ref['choice']) if 'choice' in ref else ref.get('score', int(ref.get('noul',0)>=.5))
        matching += actual['index'] == index
        expected_accept = ref['confidence'] >= .6 and ref['abstain'] <= .4
        source_accepted += expected_accept
        accepted_same += actual['accepted'] == expected_accept
        accepted += actual['accepted']
        probs = ref.get('probabilities', [ref.get('noul',0)])
        if isinstance(probs,dict): probs = [probs[label] for label in m['labels']]
        for x,y in zip(probs, actual['probabilities'], strict=True):
            assert math.isfinite(y) and 0<=y<=1
            max_error = max(max_error, abs(x-y))
        for key in ('confidence','abstain','noul'):
            value = actual[key]
            assert math.isfinite(value) and 0 <= value <= 1
            max_error = max(max_error,abs(ref.get(key,0)-value))
        assert actual['calibrated'] is False
    agreement = matching/len(rows)
    # Fixed gates, including ambiguous/OOD rows; no tuning on the held-out set.
    assert agreement >= .99, (agreement, m['kind'])
    assert max_error <= .025, max_error
    assert accepted_same/len(rows) >= .99
    return {'vectors':len(rows),'decision_agreement':agreement,'max_probability_error':max_error,
            'acceptance_agreement':accepted_same/len(rows),'accepted':accepted,'source_accepted':source_accepted}

def host_test(name, bits=16):
    binary, m, rows, report = compile_host(name, bits=bits)
    commands = ['selftest','meta'] + ['infer '+' '.join(map(str,r['features'])) for r in rows] + ['bench']
    # malformed input, overlong line, NUL injection, recovery and finite/zero guards
    malformed = ['infer 1','infer nan '+'0 '*(m['dims']-1),'infer inf',
                 'infer '+'0 '*m['dims'], 'infer 1junk', 'x'*20000, 'infer 1\x00 2']
    commands += malformed+['selftest']
    p = subprocess.run([binary],input='\n'.join(commands)+'\n',text=True,capture_output=True,check=True,timeout=30)
    results = [json.loads(line) for line in p.stdout.splitlines()]
    assert len(results) == len(commands)+1
    assert results[0]['selftest_pass'] and results[1]['selftest_pass']
    assert results[0]['model_sha256'] == report['sha256']
    parity = compare(m,rows,results[3:3+len(rows)])
    bench = results[3+len(rows)]
    assert all('error' in r and r['accepted'] is False for r in results[4+len(rows):-1])
    assert results[-1]['selftest_pass']
    scalar, _, _, _ = compile_host(name, True, bits)
    ref_run = subprocess.run([scalar],input='\n'.join(commands[:3+len(rows)])+'\n',text=True,capture_output=True,check=True,timeout=30)
    scalar_results = [json.loads(line) for line in ref_run.stdout.splitlines()]
    for optimized, reference in zip(results[3:3+len(rows)], scalar_results[3:3+len(rows)],strict=True):
        assert {k:v for k,v in optimized.items() if k!='inference_us'} == {k:v for k,v in reference.items() if k!='inference_us'}
    return {**report,**parity,'host_benchmark':bench,'boot':results[0]}

def hardware(port,name):
    import serial
    _, m, rows, report = compile_host(name)
    with serial.Serial(port,115200,timeout=2) as s:
        def command(text):
            s.write((text+'\n').encode()); s.flush()
            deadline = time.monotonic()+10
            while time.monotonic()<deadline:
                line=s.readline()
                if line.startswith(b'{'): return json.loads(line)
            raise TimeoutError('no JSON reply from firmware')
        time.sleep(2)  # allow USB bridge reset/boot before protocol commands
        s.reset_input_buffer()
        meta=command('meta')
        assert meta['target'] in ('esp32s3','esp32c6') and meta['selftest_pass']
        assert meta['model_sha256']==report['sha256'], 'flashed model differs from fixture'
        answers=[command('infer '+' '.join(map(str,row['features']))) for row in rows]
        parity=compare(m,rows,answers)
        assert command('selftest')['selftest_pass']
        bench=command('bench')
        heap_baseline=command('meta')['free_heap']  # after formatting all replay values
        for row in rows[:50]: command('infer '+' '.join(map(str,row['features'])))
        assert command('meta')['free_heap'] >= heap_baseline, 'heap loss after warmup'
        times=sorted(a['inference_us'] for a in answers)
        assert times[int(len(times)*.99)]<100000, 'p99 >=100ms'
        return {'hardware':True,'meta':meta,**parity,'benchmark':bench,'p99_us':times[int(len(times)*.99)]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port'); p.add_argument('--model',default='probe'); p.add_argument('--skip-rust',action='store_true')
    p.add_argument('--seed',type=int,default=987654321,help='independent held-out feature stream; does not change training')
    a=p.parse_args(); BUILD.mkdir(exist_ok=True)
    if not a.skip_rust:
        run(['cargo','test','--manifest-path','tests/reference/Cargo.toml','--features','hash-embedder','--locked'])
        run(['cargo','run','--release','--locked','--manifest-path','tests/reference/Cargo.toml','--bin','export-reference','--',BUILD/'fixtures',str(a.seed)])
    run(['python3','-m','unittest','discover','-s','tests','-p','test_export.py'])
    if a.port:
        report=hardware(a.port,a.model)
        filename=f'hardware-{report["meta"]["target"]}.json'
    else:
        run(['gcc','-std=c11','-g','-O1','-fsanitize=address,undefined','-fno-omit-frame-pointer','-Wall','-Wextra','-Werror','-I',INCLUDE,CORE,ROOT/'tests/test_kernel.c','-lm','-o',BUILD/'test-kernel'])
        run([BUILD/'test-kernel'])
        report={'hardware':False,'fixture_seed':a.seed,'models':{name:host_test(name) for name in ('prototype','probe','score','logistic','similarity','wide')}}
        report['int8_models'] = {name:host_test(name, 8) for name in ('prototype','similarity')}
        filename='results.json'
    (BUILD/filename).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
