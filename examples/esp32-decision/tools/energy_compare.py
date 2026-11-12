#!/usr/bin/env python3
"""Apply the same paired retention gate to measured energy reports."""
import argparse
import json
from pathlib import Path
from rig import retention

def validate_provenance(pairs,rig):
    rig_rounds=rig.get('rounds')
    if not isinstance(rig_rounds,list) or len(pairs)!=len(rig_rounds):
        raise ValueError('energy pairs must cover every rig round exactly once')
    for index,(pair,rig_round) in enumerate(zip(pairs,rig_rounds,strict=True)):
        for arm in ('baseline','candidate'):
            report=pair[arm];source=rig_round[arm];batch=source.get('energy_batch')
            if report.get('round')!=index:raise ValueError('energy round mismatch')
            if report.get('execution')!=rig.get('execution'):raise ValueError('execution provenance mismatch')
            if report.get('physical_hardware') is not rig.get('physical_hardware'):
                raise ValueError('hardware provenance mismatch')
            if report.get('model_sha256')!=rig.get('model_sha256'):
                raise ValueError('correctness model mismatch')
            if report.get('kernel_sha256')!=source['meta'].get('kernel_sha256'):
                raise ValueError('kernel provenance mismatch')
            if report.get('target')!=source['meta'].get('target'):
                raise ValueError('target provenance mismatch')
            if report.get('image')!=source.get('image'):
                raise ValueError('image provenance mismatch')
            if not isinstance(batch,dict) or not batch.get('energy_batch'):
                raise ValueError('rig round has no energy batch')
            if report.get('decisions')!=batch.get('runs'):
                raise ValueError('energy decision count mismatch')
            if report.get('scope')!='kernel_batch':raise ValueError('energy scope mismatch')

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
    validate_provenance(pairs,rig)
    for pair in pairs:
        for value in pair.values():
            if value['model_sha256']!=rig['model_sha256']:raise ValueError('correctness model mismatch')
            # Each energy report must belong to this exact correctness run.
            import hashlib
            if value['rig_report_sha256']!=hashlib.sha256(a.rig_report.read_bytes()).hexdigest():raise ValueError('rig evidence digest mismatch')
    result=compare_energy(pairs,rig['acceptance']['correctness_pass'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
