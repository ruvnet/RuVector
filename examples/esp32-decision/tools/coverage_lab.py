#!/usr/bin/env python3
"""Bounded sensor abstention experiment; never promotes a deployment model."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from e2e import ROOT
from quantize import f32, export
from sensor_data import prepare, metrics
from sensor_lab import compile_model, replay, evaluate

# Frozen before running this experiment. These are empirical demonstration
# requirements, not statistical guarantees for correlated sensor readings.
TAUS=(.35,0.,-.35)
GATES=dict(accuracy=.95,accepted_accuracy=.98,coverage=.50,
           class_accepted_accuracy=.95,class_coverage=.20,class_accepted_rows=16)

def split_calibration(data):
    data=copy.deepcopy(data)
    day=max(r['date'][:10] for r in data['train'])
    data['calibration']=[r for r in data['train'] if r['date'][:10]==day]
    data['train']=[r for r in data['train'] if r['date'][:10]!=day]
    if not data['train'] or not data['calibration']:raise ValueError('empty temporal split')
    dims=len(data['preprocessing']['mean'])
    mean=[f32(statistics.mean(r['raw'][i] for r in data['train'])) for i in range(dims)]
    std=[f32(statistics.pstdev(r['raw'][i] for r in data['train'])) for i in range(dims)]
    if min(std)<=0:raise ValueError('constant training feature')
    data['preprocessing'].update(mean=mean,std=std)
    dates=[]
    for name in ('train','calibration','validation','test'):
        rows=data[name];dates.append({r['date'] for r in rows})
        for row in rows:
            row['features']=[f32(f32(x-m)/s) for x,m,s in zip(row['raw'],mean,std,strict=True)]
    if any(dates[i]&dates[j] for i in range(4) for j in range(i)):
        raise ValueError('temporal split overlap')
    return data,day

def quality(rows,answers):
    result=metrics(rows,answers);result['by_truth_class']={}
    for label in (0,1):
        pairs=[(r,a) for r,a in zip(rows,answers,strict=True) if r['label']==label]
        accepted=[(r,a) for r,a in pairs if a['accepted']]
        result['by_truth_class'][str(label)]={
            'rows':len(pairs),'accepted_rows':len(accepted),
            'coverage':len(accepted)/len(pairs) if pairs else 0,
            'accepted_accuracy':sum(a['index']==label for r,a in accepted)/len(accepted) if accepted else None}
    result['rejection_reasons']={
        'confidence_below_0_6':sum(a['confidence']<.6 for a in answers),
        'abstain_above_0_4':sum(a['abstain']>.4 for a in answers),
        'both':sum(a['confidence']<.6 and a['abstain']>.4 for a in answers)}
    return result

def passes(result):
    if any((result.get(k) or 0)<GATES[k] for k in ('accuracy','accepted_accuracy','coverage')):
        return False
    return all(c['accepted_rows']>=GATES['class_accepted_rows'] and
               c['coverage']>=GATES['class_coverage'] and
               (c['accepted_accuracy'] or 0)>=GATES['class_accepted_accuracy']
               for c in result['by_truth_class'].values()) and len(result['by_truth_class'])==2

def select(candidates):
    # Prefer the least relaxed abstention geometry. Test/validation fields
    # are deliberately absent from this API's selection contract.
    for c in sorted(candidates,key=lambda c:-c['tau']):
        if c['parity_pass'] and passes(c['calibration']):return c['name']
    return None

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=ROOT/'build-data/occupancy.zip')
    parser.add_argument('--output',type=Path,default=ROOT/'build-coverage')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    stale=args.output/'candidate/model.h'
    if stale.exists():stale.unlink()
    data,provenance=prepare(args.archive);data,day=split_calibration(data)
    report={'schema':1,'dataset':provenance,'calibration_day':day,
            'split_counts':{name:len(data[name]) for name in ('train','calibration','validation','test')},
            'gates':GATES,'fixed_taus':TAUS,'physical_hardware':False,'deployment_ready':False,
            'statistical_guarantee':False,'test_status':'previously seen regression dataset; no fresh generalization claim',
            'selection_rule':'smallest quantization on calibration parity; least relaxed tau passing calibration quality',
            'candidates':[]}
    compiled={}
    for i,tau in enumerate(TAUS):
        folder=args.output/f'arm{i}';folder.mkdir(exist_ok=True)
        prepared={**data,'engine_options':{'logitScale':5.,'abstainTau':tau}}
        path=folder/'prepared.json';path.write_text(json.dumps(prepared))
        subprocess.run(['cargo','run','--release','--locked','--manifest-path',str(ROOT/'tests/reference/Cargo.toml'),
                        '--bin','train-sensor','--',str(path),str(folder)],check=True)
        snapshot=json.loads((folder/'snapshot.json').read_text())
        calibration=json.loads((folder/'calibration.json').read_text());precisions=[];chosen=None
        for bits in (8,16):
            binary,q=compile_model(snapshot,calibration,folder/f'int{bits}',bits)
            answers=replay(binary,calibration);parity=evaluate(snapshot,calibration,answers)['reference_parity']
            precisions.append({'bits':bits,'quantization':q,'parity':parity})
            if chosen is None and parity['pass']:chosen=(bits,binary,q,answers)
        if chosen is None:raise AssertionError('no precision passes calibration parity')
        bits,binary,q,answers=chosen;name=f'arm{i}'
        candidate={'name':name,'tau':tau,'bits':bits,'quantization':q,'precisions':precisions,
                   'parity_pass':True,'calibration':quality(calibration,answers)}
        report['candidates'].append(candidate);compiled[name]=(binary,snapshot,folder,bits)
    selected=select(report['candidates']);report['selected_on_calibration']=selected
    # Persist the frozen selection before reading any arm's validation/test
    # answers. A failure after this point cannot choose another candidate.
    frozen=json.dumps({'selected':selected,'gates':GATES,'taus':TAUS,
                       'archive_sha256':provenance['archive_sha256'],
                       'models':{c['name']:c['quantization']['sha256'] for c in report['candidates']}},sort_keys=True)
    (args.output/'selection.json').write_text(frozen+'\n')
    report['selection_sha256']=hashlib.sha256((frozen+'\n').encode()).hexdigest()
    report['retained_for_demonstration']=False
    for candidate in report['candidates']:
        if candidate['name'] not in {'arm0',selected}:continue
        binary,snapshot,folder,bits=compiled[candidate['name']]
        for name in ('validation','test'):
            rows=json.loads((folder/f'{name}.json').read_text());answers=replay(binary,rows)
            candidate[name]={'quality':quality(rows,answers),
                             'parity':evaluate(snapshot,rows,answers)['reference_parity']}
        candidate['validation_gate_pass']=passes(candidate['validation']['quality']) and candidate['validation']['parity']['pass']
        if candidate['name']==selected:
            report['retained_for_demonstration']=candidate['validation_gate_pass'] and passes(candidate['test']['quality']) and candidate['test']['parity']['pass']
            if report['retained_for_demonstration']:
                rows=json.loads((folder/'calibration.json').read_text())
                export(snapshot,args.output/'candidate/model.h',rows,bits)
    # A reused output directory must never expose a stale promoted candidate.
    header=args.output/'candidate/model.h'
    if not report['retained_for_demonstration'] and header.exists():header.unlink()
    report['next_gate']='new room/device data and physical tests; default model never changed by this lab'
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'selected':selected,'retained_for_demonstration':report['retained_for_demonstration'],
                      'report':str(args.output/'report.json')}))
if __name__=='__main__':main()
