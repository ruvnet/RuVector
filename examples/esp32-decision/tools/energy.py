#!/usr/bin/env python3
"""Integrate power over marker windows. No hardware energy is inferred from timing."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

def integrate(rows,decisions_per_window,idle_current_a=0):
    if decisions_per_window<1 or not math.isfinite(idle_current_a) or idle_current_a<0:
        raise ValueError('invalid count or idle current')
    if len(rows)<3:raise ValueError('trace too short')
    parsed=[]
    for row in rows:
        t,v,a,marker=(float(row[k]) for k in ('time_s','voltage_v','current_a','marker'))
        if not all(math.isfinite(x) for x in (t,v,a,marker)) or v<=0 or a<0 or marker not in (0,1):
            raise ValueError('invalid trace value or units')
        if parsed and t<=parsed[-1][0]:raise ValueError('timestamps must increase strictly')
        parsed.append((t,v,a,int(marker)))
    if parsed[0][3] or parsed[-1][3]:raise ValueError('trace must bracket complete marker windows')
    windows=[];start=None;gross=adjusted=duration=resolution=0.0;samples=0
    for left,right in zip(parsed,parsed[1:]):
        t,v,a,marker=left;tn,vn,an,mn=right;dt=tn-t
        if marker and start is None:start=t;gross=adjusted=duration=resolution=0.0;samples=0
        if marker:
            gross+=dt*(v*a+vn*an)/2
            adjusted+=dt*(v*(a-idle_current_a)+vn*(an-idle_current_a))/2
            duration+=dt;resolution=max(resolution,dt);samples+=1
            if not mn:
                windows.append({'start_s':start,'duration_s':duration,'gross_joules':gross,
                    'idle_adjusted_joules':adjusted,'gross_joules_per_decision':gross/decisions_per_window,
                    'idle_adjusted_joules_per_decision':adjusted/decisions_per_window,
                    'decisions':decisions_per_window,'sample_intervals':samples,'max_sample_interval_s':resolution})
                start=None
    if not windows:raise ValueError('no complete marker window')
    count=sum(w['decisions'] for w in windows)
    return {'windows':windows,'decisions':count,'gross_joules_per_decision':sum(w['gross_joules'] for w in windows)/count,
            'idle_adjusted_joules_per_decision':sum(w['idle_adjusted_joules'] for w in windows)/count,
            'edge_resolution_note':'Marker edges are quantized to sample timestamps; report sample interval, do not infer unsampled peaks.'}

def analyze(trace,run_report,round_index,arm,idle_current_a=0):
    report=json.loads(Path(run_report).read_text());run=report['rounds'][round_index][arm]
    if not run['energy_batch']['energy_batch']:raise ValueError('no recorded energy batch')
    if run['energy_batch']['marker_gpio']<0:raise ValueError('batch had no GPIO marker')
    with Path(trace).open(newline='') as f:rows=list(csv.DictReader(f))
    result=integrate(rows,run['energy_batch']['runs'],idle_current_a)
    # Each recorded command produces exactly one marked batch. Multiple
    # windows cannot silently multiply the denominator for this run.
    if len(result['windows'])!=1:raise ValueError('trace window count does not match recorded batch')
    if result['windows'][0]['sample_intervals']<10:raise ValueError('trace too coarse: require >=10 sample intervals in the marker window')
    duration=result['windows'][0]['duration_s'];resolution=result['windows'][0]['max_sample_interval_s']
    expected=run['energy_batch']['batch_us']/1e6
    if abs(duration-expected)>2*resolution+expected*.05:raise ValueError('marker duration does not match batch')
    result.update(schema=1,execution=report['execution'],physical_hardware=report['physical_hardware'],
        trace_sha256=hashlib.sha256(Path(trace).read_bytes()).hexdigest(),
        rig_report_sha256=hashlib.sha256(Path(run_report).read_bytes()).hexdigest(),
        image=run['image'],kernel_sha256=run['meta']['kernel_sha256'],model_sha256=report['model_sha256'],
        target=run['meta'].get('target','unknown'),
        scope='kernel_batch',idle_current_a=idle_current_a,round=round_index,arm=arm)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path);p.add_argument('--run-report',type=Path,required=True)
    p.add_argument('--round',type=int,default=0);p.add_argument('--arm',choices=['baseline','candidate'],required=True)
    p.add_argument('--idle-current-a',type=float,default=0);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=analyze(a.trace,a.run_report,a.round,a.arm,a.idle_current_a)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
