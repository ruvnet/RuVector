"""Independently verify a completed localization run and publish its evidence."""
import argparse,hashlib,json,math,statistics,re
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
for name in ['run','data','manifest','model','code','dataset','output']:
    parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--source-commit',required=True)
args=parser.parse_args()
if __debug__ is False:raise RuntimeError('verification requires assertions enabled')
if not re.fullmatch(r'[0-9a-f]{40}',args.source_commit):raise ValueError('invalid source commit')
bench=Path(__file__).parent
run=args.run;data=args.data
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((run/'receipt.json').read_text());m=json.loads(args.manifest.read_text())
assert r['manifest']==m
for name,h in m['sourceHashes'].items():assert hashfile(bench/name)==h
for name,h in m['artifacts'].items():assert hashfile(Path(name))==h
assert len(r['records'])==len(m['data']['cases'])==6
assert len({x['id'] for x in r['records']})==6
rows=[];evidence={}
for row,item in zip(r['records'],m['data']['cases']):
    ident=row['id'];assert ident==item['id']
    assert re.fullmatch(r'[a-zA-Z0-9_]+-[0-9]+',ident), 'invalid instance id'
    inp=data/(ident+'.json');lab=data/(ident+'.labels.json');pred=run/(ident+'.predictions.json')
    assert hashfile(inp)==item['inputHash']
    assert hashfile(lab)==item['labelHash']
    assert row['inputHash']==item['inputHash'] and row['labelHash']==item['labelHash']
    assert hashfile(pred)==row['predictionHash']
    d=json.loads(inp.read_text());labels=json.loads(lab.read_text());gold=labels['goldFiles'];p=json.loads(pred.read_text())
    assert labels['id']==ident and gold and len(gold)==len(set(gold))
    assert d['repo']==row['repo']==item['repo']
    assert d['baseCommit']==row['baseCommit']==item['baseCommit']
    assert set(p)=={'bm25','swerank'}
    assert d['id']==ident and row['documents']==len(d['documents'])
    universe={x['path'] for x in d['documents']}
    assert row['files']==len(universe)==item['fileCount']
    assert row['documents']==item['documentCount']
    assert universe==set(item['sourceFileHashes'])
    for arm,ranking in p.items():
        paths=[x['path'] for x in ranking]
        assert len(paths)==len(set(paths))==len(universe) and set(paths)==universe
        assert all(math.isfinite(x['score']) for x in ranking)
        assert ranking==sorted(ranking,key=lambda x:(-x['score'],x['path']))
        ranks=[paths.index(g)+1 for g in gold]
        expected=dict(mrr=1/min(ranks),hit1=int(1 in ranks),recall5=sum(x<=5 for x in ranks)/len(ranks),recall10=sum(x<=10 for x in ranks)/len(ranks),goldCount=len(gold),goldRanks=ranks)
        assert expected==row[arm]
    evidence[ident]=dict(predictions=p,goldFiles=gold)
    rows.append(dict(id=ident,bm25=row['bm25']['goldRanks'],swerank=row['swerank']['goldRanks'],indexing=row['indexing'],query=row['queryEncoding'],rankingSeconds=row['rankingSeconds'],bm25Seconds=row['bm25Seconds']))
assert set(r['aggregate'])=={'bm25','swerank'}
for arm in ['bm25','swerank']:
    assert set(r['aggregate'][arm])=={'mrr','hit1','recall5','recall10'}
    for metric,v in r['aggregate'][arm].items():assert v==statistics.mean(x[arm][metric] for x in r['records'])
provenance=[];matched=set();dataset_matched=False
for label,folder,rev in [('model',args.model,m['modelRevision']),('code',args.code,m['codeRevision']),('dataset',args.dataset,m['data']['revision'])]:
    metadata=sorted((folder/'.cache/huggingface/download').rglob('*.metadata'))
    assert metadata, 'missing download provenance'
    for p in metadata:
        fields=p.read_text().splitlines();assert fields[0]==rev
        relative=p.relative_to(folder/'.cache/huggingface/download')
        artifact=folder/str(relative).removesuffix('.metadata')
        raw=artifact.read_bytes();sha256=hashlib.sha256(raw).hexdigest()
        etag=fields[1]
        if len(etag)==64:assert sha256==etag
        elif len(etag)==40:assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==etag
        else:raise ValueError('unsupported artifact checksum')
        if label in ['model','code'] and artifact.parent.resolve()==folder.resolve():
            assert m['artifacts'][str(artifact.resolve())]==sha256
            matched.add(str(artifact.resolve()))
        if label=='dataset' and artifact.suffix=='.parquet':
            assert m['data']['parquetHash']==sha256
            dataset_matched=True
        provenance.append(dict(folder=label,file=str(p.relative_to(folder/'.cache/huggingface/download')),revision=fields[0],etag=fields[1]))
assert matched==set(m['artifacts']) and dataset_matched
r['evidence']=evidence;r['downloadMetadata']=provenance
r['verification']=dict(independentRankArithmetic=True,completeFileUniverse=True,sourceAndArtifactHashes=True,predictionHashes=True,declaredSourceCommit=args.source_commit,downloadMetadataAuthenticatesOrigin=False,verifierHash=hashfile(Path(__file__)))
args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(aggregate=r['aggregate'],initialization=r['initializationSeconds'],rows=rows,totalIndexSeconds=sum(x['indexing']['seconds'] for x in rows),meanWarmSeconds=statistics.mean(x['query']['seconds']+x['rankingSeconds'] for x in rows),meanBm25Seconds=statistics.mean(x['bm25Seconds'] for x in rows),provenanceFiles=len(provenance)),indent=2))
