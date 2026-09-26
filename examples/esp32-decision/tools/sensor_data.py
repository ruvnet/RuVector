"""Pinned UCI occupancy measurements; no random split of adjacent time windows."""
import csv
import hashlib
import io
import math
from pathlib import Path
import statistics
import urllib.request
import zipfile
from quantize import f32

URL='https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip'
SHA256='4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a'
FEATURES=['Temperature','Humidity','Light','CO2','HumidityRatio']
UNITS=['C','percent','lux','ppm','kg/kg']

def archive(path):
    path=Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(URL,timeout=30) as response: data=response.read(2_000_001)
        if len(data)>2_000_000 or hashlib.sha256(data).hexdigest()!=SHA256:
            raise ValueError('download digest mismatch')
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA256: raise ValueError('archive digest mismatch')
    return zipfile.ZipFile(path)

def prepare(path):
    with archive(path) as z:
        def read(name):
            stream=io.StringIO(z.read(name).decode()); reader=csv.reader(stream)
            header=['id']+next(reader)
            rows=[]
            for values in reader:
                r=dict(zip(header,values,strict=True))
                raw=[f32(float(r[k])) for k in FEATURES];label=int(r['Occupancy'])
                if label not in (0,1): raise ValueError('invalid label')
                rows.append({'date':r['date'],'raw':raw,'label':label})
            return rows
        original=read('datatraining.txt'); tests=read('datatest.txt')+read('datatest2.txt')
    # Entire last training day is validation, with no fitting on those rows.
    final_day=max(r['date'][:10] for r in original)
    train=[r for r in original if r['date'][:10]!=final_day]
    validation=[r for r in original if r['date'][:10]==final_day]
    mean=[f32(statistics.mean(r['raw'][i] for r in train)) for i in range(5)]
    std=[f32(statistics.pstdev(r['raw'][i] for r in train)) for i in range(5)]
    if min(std)<=0: raise ValueError('constant input feature')
    for rows in (train,validation,tests):
        for r in rows: r['features']=[f32(f32(x-m)/s) for x,m,s in zip(r['raw'],mean,std)]
    dates=[{r['date'] for r in rows} for rows in (train,validation,tests)]
    if any(dates[i]&dates[j] for i in range(3) for j in range(i)): raise ValueError('time split overlap')
    pre={'kind':'standardize-f32-v1','features':FEATURES,'units':UNITS,'mean':mean,'std':std}
    return {'train':train,'validation':validation,'test':tests,'preprocessing':pre}, {
        'source':URL,'doi':'10.24432/C5X01N','attribution':'Luis Candanedo (2016), UCI Occupancy Detection',
        'license':'CC BY 4.0','archive_sha256':SHA256,'validation_day':final_day,
        'split_counts':{'train':len(train),'validation':len(validation),'test':len(tests)},
        'split_time_ranges':{key:[min(r['date'] for r in rows),max(r['date'] for r in rows)]
                             for key,rows in [('train',train),('validation',validation),('test',tests)]},
        'generalization_scope':'Separate time periods in one office; no building or device transfer claim'}

def metrics(rows,answers):
    confusion=[[0,0],[0,0]];accepted=correct_accepted=0;brier=0.0;bins=[[] for _ in range(10)]
    for row,a in zip(rows,answers,strict=True):
        y=row['label'];pred=a['index'];confusion[y][pred]+=1
        if a['accepted']: accepted+=1;correct_accepted+=pred==y
        p=a['probabilities'][1];brier+=(p-y)**2
        conf=max(p,1-p);bins[min(9,int(conf*10))].append((conf,pred==y))
    n=len(rows);tn,fp=confusion[0];fn,tp=confusion[1]
    ece=sum(abs(sum(c for c,_ in b)-sum(ok for _,ok in b)) for b in bins if b)/n
    return {'rows':n,'accuracy':(tn+tp)/n,'confusion_matrix':confusion,
            'false_positive_rate':fp/max(1,tn+fp),'false_negative_rate':fn/max(1,tp+fn),
            'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'coverage':accepted/n,
            'accepted_accuracy':correct_accepted/accepted if accepted else None,
            'brier_score':brier/n,'conditional_class_ece_10_bins':ece,
            'calibration_validated':False}
