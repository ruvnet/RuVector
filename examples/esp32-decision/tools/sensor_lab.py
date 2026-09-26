#!/usr/bin/env python3
"""Train real occupancy measurements through RuVector and evaluate INT8/INT16 C firmware."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from e2e import ROOT,INCLUDE,CORE,compare
from quantize import export
from sensor_data import prepare,metrics

def compile_model(snapshot,rows,folder,bits,source=CORE,profile='portable',extra_sources=()):
    folder.mkdir(parents=True,exist_ok=True)
    quant=export(snapshot,folder/'model.h',rows,bits)
    binary=folder/'firmware'
    flags=[]
    if profile!='portable':flags+=['-DRD_PAIR_DOT']
    if profile=='esp32s3':flags+=['-DRD_LIBM_ROUND']
    digest=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    subprocess.run(['gcc','-std=c11','-O3','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror',*flags,
        '-DRD_KERNEL_SHA256="'+digest+'"','-I',str(folder),'-I',str(INCLUDE),'-I',str(ROOT/'main'),str(source),
        str(ROOT/'main/app.c'),str(ROOT/'main/profile.c'),str(ROOT/'main/sensor.c'),str(ROOT/'tests/host_main.c'),
        *map(str,extra_sources),'-lm','-o',str(binary)],check=True)
    return binary,quant

def replay(binary,rows):
    commands=['sensor '+' '.join(map(str,r['raw'])) for r in rows]
    proc=subprocess.run([binary],input='\n'.join(commands)+'\n',text=True,capture_output=True,check=True,timeout=120)
    results=[json.loads(line) for line in proc.stdout.splitlines()]
    if len(results)!=len(rows)+1 or not results[0]['selftest_pass']: raise AssertionError('boot or row count')
    return results[1:]

def evaluate(snapshot,rows,answers):
    try: parity={**compare(snapshot,rows,answers),'pass':True}
    except AssertionError as error: parity={'pass':False,'reason':str(error)}
    return {'reference_parity':parity,'task':metrics(rows,answers)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ROOT/'build-data/occupancy.zip')
    p.add_argument('--output',type=Path,default=ROOT/'build-sensor')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data,provenance=prepare(a.archive)
    prepared=a.output/'prepared.json';prepared.write_text(json.dumps(data))
    subprocess.run(['cargo','run','--release','--locked','--manifest-path',str(ROOT/'tests/reference/Cargo.toml'),
                    '--bin','train-sensor','--',str(prepared),str(a.output)],check=True)
    snapshot=json.loads((a.output/'snapshot.json').read_text())
    validation=json.loads((a.output/'validation.json').read_text());test=json.loads((a.output/'test.json').read_text())
    report={'dataset':provenance,'physical_hardware':False,'training':json.loads((a.output/'training.json').read_text()),
            'preprocessing':data['preprocessing'],'candidates':{}}
    binaries={}
    # Selection reads validation only. Test is evaluated after this decision.
    for bits in (8,16):
        binary,q=compile_model(snapshot,validation,a.output/f'int{bits}',bits);binaries[bits]=binary
        answers=replay(binary,validation)
        report['candidates'][str(bits)]={'quantization':q,'validation':evaluate(snapshot,validation,answers)}
    eligible=[bits for bits in (8,16) if report['candidates'][str(bits)]['validation']['reference_parity']['pass']]
    if not eligible:raise AssertionError('no precision satisfies validation parity')
    selected=eligible[0];report['selected_bits']=selected
    report['selection_rule']='smallest precision passing fixed validation parity gates; no test tuning'
    for bits in (8,16):
        answers=replay(binaries[bits],test)
        report['candidates'][str(bits)]['test']=evaluate(snapshot,test,answers)
    chosen=report['candidates'][str(selected)]['test']
    report['demonstration_gate_pass']=chosen['reference_parity']['pass'] and chosen['task']['accuracy']>=.90
    report['deployment_ready']=False
    report['deployment_limits']=['One office only; device and building transfer untested',
        'Confidence calibration and useful acceptance coverage require a separate validation campaign',
        'Physical sensor latency, energy and driver acquisition remain unmeasured']
    # Preserve the selected model even on a failed test gate for inspection,
    # but exit nonzero and never silently select another using test labels.
    export(snapshot,a.output/'selected/model.h',validation,selected)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['demonstration_gate_pass']:raise SystemExit('test gate failed; no deployment promotion')
if __name__=='__main__':main()
