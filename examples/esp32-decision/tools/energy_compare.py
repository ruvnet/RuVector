#!/usr/bin/env python3
"""Apply the same paired retention gate to measured energy reports."""
import argparse
import json
from pathlib import Path
from rig import retention

def compare_energy(pairs,correctness):
    rounds=[];seen=set();identity=None;physical=True
    for pair in pairs:
        record={}
        for arm in ('baseline','candidate'):
            r=pair[arm]
            if r['trace_sha256'] in seen:raise ValueError('reused trace is not an independent trial')
            seen.add(r['trace_sha256'])
            if r['arm']!=arm:raise ValueError('arm mismatch')
            key=(r['model_sha256'],r['target'],r['scope'],r['decisions'])
            if identity is None:identity=key
            if key!=identity:raise ValueError('energy workload or target mismatch')
            physical=physical and r['physical_hardware'] and r['execution']=='physical'
            record[arm]={'profile':{'joules_per_decision':r['gross_joules_per_decision']}}
        rounds.append(record)
    if not rounds:raise ValueError('no energy pairs')
    return retention(rounds,'joules_per_decision','physical' if physical else 'simulated',correctness)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pairs',type=Path,help='JSON array of baseline/candidate energy report paths')
    p.add_argument('--rig-report',type=Path,required=True,help='paired correctness evidence for the same model')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();paths=json.loads(a.pairs.read_text())
    pairs=[{arm:json.loads(Path(pair[arm]).read_text()) for arm in ('baseline','candidate')} for pair in paths]
    rig=json.loads(a.rig_report.read_text())
    for pair in pairs:
        for value in pair.values():
            if value['model_sha256']!=rig['model_sha256']:raise ValueError('correctness model mismatch')
            # Each energy report must belong to this exact correctness run.
            import hashlib
            if value['rig_report_sha256']!=hashlib.sha256(a.rig_report.read_bytes()).hexdigest():raise ValueError('rig evidence digest mismatch')
    result=compare_energy(pairs,rig['acceptance']['correctness_pass'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
