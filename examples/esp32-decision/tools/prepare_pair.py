#!/usr/bin/env python3
"""Build both kernels with identical current instrumentation and frozen model."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from e2e import ROOT,CORE,INCLUDE

BASELINE='739a5621043f9b0f0ffc263c46d1d2f7aeb9495f'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def prepare(model_dir,out,targets=(),baseline=BASELINE):
    model_dir=Path(model_dir).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    frozen=out/'frozen-model';frozen.mkdir(exist_ok=True)
    if (model_dir/'model.h').resolve()!=(frozen/'model.h').resolve():shutil.copy2(model_dir/'model.h',frozen/'model.h')
    model_dir=frozen
    old=out/'baseline.c'
    old.write_bytes(subprocess.check_output(['git','show',baseline+':examples/esp32-decision/components/rvdecision/rvdecision.c'],cwd=ROOT))
    match=re.search(r'#define RD_MODEL_HASH "([0-9a-f]{64})"',(model_dir/'model.h').read_text())
    if not match:raise ValueError('model header has no digest')
    result={'schema':1,'baseline_commit':baseline,'model_sha256':match[1],
            'model_header_sha256':sha(model_dir/'model.h'),'model_dir':str(model_dir),
            'kernel_flags':'-O3 -fno-fast-math -ffp-contract=off','variants':{}}
    for name,source in [('baseline',old),('candidate',CORE)]:
        entry={'kernel_sha256':sha(source),'images':{}}
        binary=out/f'host-{name}'
        subprocess.run(['gcc','-std=c11','-O3','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror',
            '-DRD_KERNEL_SHA256="'+sha(source)+'"','-I',str(model_dir),'-I',str(INCLUDE),'-I',str(ROOT/'main'),
            str(source),str(ROOT/'main/app.c'),str(ROOT/'main/profile.c'),str(ROOT/'main/sensor.c'),str(ROOT/'tests/host_main.c'),'-lm','-o',str(binary)],check=True)
        entry['host_binary']=str(binary);entry['host_sha256']=sha(binary)
        for chip in targets:
            build=out/f'{chip}-{name}'
            subprocess.run(['idf.py','-B',str(build),'-DIDF_TARGET='+chip,
                '-DSDKCONFIG='+str(out/f'sdkconfig.{chip}.{name}'),'-DRD_MODEL_DIR='+str(model_dir),
                '-DRD_KERNEL_SOURCE='+str(source),'build','merge-bin'],cwd=ROOT,check=True)
            image=build/'merged-binary.bin'
            entry['images'][chip]={'file':str(image),'sha256':sha(image),'bytes':image.stat().st_size}
        result['variants'][name]=entry
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=ROOT/'main')
    p.add_argument('--output',type=Path,default=ROOT/'build-rig')
    p.add_argument('--targets',nargs='*',choices=['esp32s3','esp32c6'],default=[])
    p.add_argument('--baseline',default=BASELINE)
    a=p.parse_args();prepare(a.model_dir,a.output,a.targets,a.baseline)
if __name__=='__main__':main()
