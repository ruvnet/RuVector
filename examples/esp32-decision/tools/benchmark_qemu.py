#!/usr/bin/env python3
"""Compare S3 images in Espressif QEMU. Times are emulator results, not silicon.

Uses the unchanged UART bench command in both images. Both run the same model
and repeat one prototype input, with 1000 inferences per measured batch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'build-bench'

class Firmware:
    def __init__(self,qemu,image,name,icount=False):
        raw=image.read_bytes()
        if len(raw)>4*1024*1024: raise ValueError('image exceeds flash')
        self.sha256=hashlib.sha256(raw).hexdigest()
        fd,path=tempfile.mkstemp(prefix=f'qemu-{name}-',suffix='.bin',dir=BUILD)
        self.flash=Path(path)
        with os.fdopen(fd,'wb') as f: f.write(raw+b'\xff'*(4*1024*1024-len(raw)))
        self.log=open(BUILD/f'qemu-{name}.log','w')
        command=[qemu,'-M','esp32s3','-m','32M','-nographic','-monitor','none',
            '-serial','stdio','-drive',f'file={self.flash},if=mtd,format=raw','-no-reboot']
        if icount: command+=['-icount','shift=0,sleep=off']
        self.proc=subprocess.Popen(command,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0)
        self.selector=selectors.DefaultSelector(); self.selector.register(self.proc.stdout,selectors.EVENT_READ)
        self.pending=bytearray()
    def response(self):
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            if b'\n' in self.pending:
                line,_,rest=self.pending.partition(b'\n'); self.pending[:]=rest
                if line.startswith(b'{'): return json.loads(line)
                continue
            if not self.selector.select(1): continue
            data=os.read(self.proc.stdout.fileno(),4096)
            if not data: raise RuntimeError('QEMU stopped')
            self.pending.extend(data)
        raise TimeoutError('QEMU response deadline')
    def query(self,text):
        self.proc.stdin.write((text+'\n').encode()); self.proc.stdin.flush(); return self.response()
    def close(self):
        self.proc.terminate()
        try: self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait()
        self.proc.stdin.close(); self.proc.stdout.close(); self.selector.close(); self.log.close()
        self.flash.unlink()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qemu',required=True)
    p.add_argument('--baseline-image',type=Path,required=True)
    p.add_argument('--candidate-image',type=Path,default=ROOT/'build-esp32s3/merged-binary.bin')
    p.add_argument('--output',type=Path,default=BUILD/'benchmark-qemu.json')
    p.add_argument('--icount',action='store_true',help='virtual instruction clock; not hardware cycles or latency')
    a=p.parse_args(); BUILD.mkdir(exist_ok=True)
    instances=[]
    try:
        for name,path in [('baseline',a.baseline_image),('candidate',a.candidate_image)]:
            fw=Firmware(a.qemu,path,name,a.icount); instances.append(fw)
            fw.meta=fw.response()
            assert fw.meta['selftest_pass'] and fw.query('selftest')['selftest_pass']
            fw.query('bench')  # warm translated code and formatting before measurement
        baseline,candidate=instances
        assert baseline.meta['model_sha256']==candidate.meta['model_sha256']
        rounds=[]
        for i in range(11):
            if i%2: b=candidate.query('bench'); a0=baseline.query('bench')
            else: a0=baseline.query('bench'); b=candidate.query('bench')
            assert a0['runs']==b['runs']==1000
            rounds.append({'baseline':a0,'candidate':b,'speedup':a0['mean_us']/b['mean_us']})
            print(f'QEMU pair {i+1}: {a0["mean_us"]:.3f} -> {b["mean_us"]:.3f} us',flush=True)
        speedups=[r['speedup'] for r in rounds]
        result={'execution':'Espressif QEMU esp32s3; not physical silicon','hardware':False,
            'clock':'virtual instruction clock, 1 ns per instruction' if a.icount else 'QEMU realtime clock; host scheduling affects results',
            'qemu':subprocess.check_output([a.qemu,'--version'],text=True).splitlines()[0],
            'method':'11 alternating paired batches of 1000 prototype-input inferences; one warmup batch per image',
            'baseline_sha256':baseline.sha256,'candidate_sha256':candidate.sha256,
            'baseline_meta':baseline.meta,'candidate_meta':candidate.meta,
            'baseline_median_mean_us':statistics.median(r['baseline']['mean_us'] for r in rounds),
            'candidate_median_mean_us':statistics.median(r['candidate']['mean_us'] for r in rounds),
            'paired_median_speedup':statistics.median(speedups),'paired_min_speedup':min(speedups),
            'paired_max_speedup':max(speedups),'rounds':rounds}
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n')
    finally:
        for fw in instances: fw.close()
if __name__=='__main__': main()
