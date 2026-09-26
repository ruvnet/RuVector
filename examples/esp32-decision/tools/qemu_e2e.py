#!/usr/bin/env python3
"""Run compiled S3 firmware in Espressif QEMU; never a hardware latency claim.

python3 tools/qemu_e2e.py --qemu /path/to/qemu-system-xtensa
Requires idf.py -B build-esp32s3 merge-bin and tools/e2e.py fixtures.
"""
import argparse
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
from e2e import BUILD, ROOT, compare, compile_host

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qemu',required=True); a=p.parse_args()
    image=(ROOT/'build-esp32s3/merged-binary.bin').read_bytes()
    assert len(image)<=4*1024*1024
    flash=BUILD/'qemu-flash.bin'; flash.write_bytes(image+b'\xff'*(4*1024*1024-len(image)))
    _,model,rows,quant=compile_host('probe')
    command=[a.qemu,'-M','esp32s3','-m','32M','-nographic','-monitor','none','-serial','stdio',
             '-drive',f'file={flash},if=mtd,format=raw','-no-reboot']
    log=open(BUILD/'qemu-stderr.log','w')
    proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,bufsize=0)
    sel=selectors.DefaultSelector(); sel.register(proc.stdout,selectors.EVENT_READ)
    pending=bytearray()
    def response():
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            if b'\n' in pending:
                line,_,rest=pending.partition(b'\n'); pending[:]=rest
                if line.startswith(b'{'): return json.loads(line)
                continue
            if not sel.select(1): continue
            chunk=os.read(proc.stdout.fileno(),4096)
            if not chunk: raise RuntimeError('QEMU stopped before a JSON response')
            pending.extend(chunk)
        raise TimeoutError('QEMU response deadline')
    def query(text):
        proc.stdin.write((text+'\n').encode()); proc.stdin.flush(); return response()
    try:
        meta=response()
        assert meta['selftest_pass'] and meta['model_sha256']==quant['sha256']
        assert query('selftest')['selftest_pass']
        answers=[]
        for i,row in enumerate(rows):
            answers.append(query('infer '+' '.join(map(str,row['features']))))
            if (i+1)%250 == 0: print(f'QEMU replay {i+1}/{len(rows)}',flush=True)
        result=compare(model,rows,answers)
        for command in ['infer nan','infer '+'0 '*32,'unknown']:
            assert query(command)['accepted'] is False
        assert query('selftest')['selftest_pass']
        baseline=query('meta')['free_heap']
        # libc number formatting lazily caches buffers. Check steady-state
        # after every replay value has been formatted, with no allowed loss.
        for row in rows[:50]: query('infer '+' '.join(map(str,row['features'])))
        final=query('meta'); assert final['free_heap']>=baseline, (baseline,final['free_heap'])
        report={'emulator':'Espressif QEMU esp32s3','hardware':False,'boot':meta,**result,
                'steady_state_baseline_heap':baseline,'final_free_heap':final['free_heap'],
                'boot_to_warm_heap_delta':meta['free_heap']-baseline}
        (BUILD/'qemu-results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        sel.close(); log.close()
if __name__=='__main__':main()
