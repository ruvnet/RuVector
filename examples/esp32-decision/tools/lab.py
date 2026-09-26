#!/usr/bin/env python3
"""One software acceptance entrypoint; physical measurements remain explicit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from e2e import ROOT

def run(*args):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets',nargs='*',choices=['esp32s3','esp32c6'],default=[])
    p.add_argument('--qemu',help='optional S3 emulator executable; requires --targets esp32s3')
    a=p.parse_args()
    if a.qemu and 'esp32s3' not in a.targets:p.error('--qemu requires the S3 build')
    run('tools/e2e.py','--seed','987654321')
    run('tools/sensor_lab.py')
    run('-m','unittest','discover','-s','tests','-p','test_*.py','-v')
    run('tools/prepare_pair.py','--model-dir','build-sensor/selected','--output','build-lab','--targets',*a.targets)
    run('tools/rig.py','build-lab/manifest.json','--vectors','build-sensor/test.json','--execution','native',
        '--rounds','11','--limit','128','--output','build-lab/native.json')
    if a.qemu:
        run('tools/rig.py','build-lab/manifest.json','--vectors','build-sensor/test.json','--execution','emulator',
            '--qemu',a.qemu,'--rounds','1','--limit','1000','--output','build-lab/qemu.json')
    print(json.dumps({'software_acceptance':'pass','built_targets':a.targets,'physical_acceptance':'not performed',
                      'sensor_report':'build-sensor/report.json','paired_report':'build-lab/native.json'}))
if __name__=='__main__':main()
