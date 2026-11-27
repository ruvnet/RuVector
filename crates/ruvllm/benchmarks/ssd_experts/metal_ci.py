"""Run the actual ruvllm Metal path whenever the CI runner exposes a Metal GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_cpu import prepare_engine
from run_metal import BASE

def main():
    scratch=Path(os.environ['RUNNER_TEMP'])
    model=scratch/'ruvllm-parity/model'
    here=Path(__file__).resolve().parent
    env=dict(os.environ,RUSTUP_TOOLCHAIN='1.93.1',CARGO_TARGET_DIR=str(scratch/'metal-quant-target'))
    quant=scratch/'ruvllm-quant'
    quant.mkdir()
    engine=prepare_engine(quant)
    args=['cargo','run','--locked','--release','--no-default-features','--features','std,f16',
          '--bin','quantize_q4','--','--model-dir',str(model),'--output-dir',str(quant/'q4')]
    subprocess.run(args,cwd=engine,env=env,check=True)
    for name in ('config.json','tokenizer.json'):
        shutil.copyfile(model/name,quant/'q4'/name)
    checkout=scratch/'ruvllm-source'
    subprocess.run(['git','clone','--no-checkout','--filter=blob:none','--depth','1',
                    'https://github.com/ruvnet/RuVector.git',str(checkout)],check=True)
    subprocess.run(['git','-C',str(checkout),'fetch','--depth','1','origin',BASE],check=True)
    subprocess.run([sys.executable,str(here/'run_metal.py'),str(checkout),str(scratch/'ruvllm-metal'),
                   '--q4',str(quant/'q4')],env=env,check=True)

if __name__=='__main__': main()
