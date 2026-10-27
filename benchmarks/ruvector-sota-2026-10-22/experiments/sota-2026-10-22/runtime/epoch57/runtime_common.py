"""E57 bounded CPU runtime; stdlib-only import, no Gold/scoring capability."""
import argparse
import base64
from collections import Counter
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback

EPOCH = 57
ENGINEERING_KIND = 'REAL_NATIVE_QUANTIZED_CPU_THROUGHPUT_AND_RETRIEVAL_DEV'
PILOT_KIND = 'REAL_NATIVE_SCIFACT_RETRIEVAL_DEV_PILOT'
BGE_PREFIX = 'Represent this sentence for searching relevant passages: '
FP32_SHA = '828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35'
INT8_SHA = '6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4'
TOKENIZER_SHA = 'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66'
RUNTIME_VERSIONS = {'numpy':'2.2.6', 'onnxruntime':'1.22.1', 'tokenizers':'0.21.4'}
GUARD_SHAS = {'inherited_guards':'2d3aad4713828a69eba4becbee5fb1c731164638a0000742aaa5bef24376f7dc', 'io_guard':'9bcb717c704114f9d9fb69ca6ed1838482e37a8491d81ca1162f54126056c13b', 'search_policy':'356f42fc110155bd9340ab35c8ebe69db5b88a907d903dc2eeffaba6a1f0505e'}
WORD = re.compile(r'\w+', re.UNICODE)

def need(condition, code):
    if not condition:
        raise ValueError(code)

def verify(ref):
    need(type(ref) is dict and set(ref) == {'path','bytes','sha256'}, 'CANONICAL_SOURCE_OR_ASSET_REF')
    p = Path(ref['path'])
    need(p.is_absolute() and p.is_file() and not any(q.is_symlink() for q in [p,*p.parents]), 'REGULAR_NONSYMLINK_REF')
    h = hashlib.sha256(); fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd); count = 0
        while True:
            block = os.read(fd, 1048576)
            if not block: break
            count += len(block); h.update(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'REF_STABLE_DURING_HASH')
    need(count == ref['bytes'] and h.hexdigest() == ref['sha256'], 'EXACT_REF_SIZE_SHA256')
    return p

def load_source(name, ref, expected_sha=None):
    if expected_sha is not None: need(ref['sha256'] == expected_sha, 'REVIEWED_INHERITED_SOURCE_SHA256')
    p = verify(ref)
    need(name not in sys.modules, 'SOURCE_MODULE_NOT_PREIMPORTED')
    spec = importlib.util.spec_from_file_location(name, p)
    need(spec is not None and spec.loader is not None, 'SOURCE_MODULE_SPEC')
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def put(path, value):
    p = Path(path); need(p.is_absolute() and p.parent.is_dir() and not p.is_symlink(), 'SAFE_NEW_OUTPUT')
    b = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n').encode()
    fd = os.open(p, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    try:
        need(os.write(fd,b) == len(b), 'COMPLETE_OUTPUT_WRITE'); os.fsync(fd)
    finally: os.close(fd)
    fd = os.open(p.parent, os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def percentile(values, q):
    need(values and 0 <= q <= 1 and all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in values), 'FINITE_NONNEGATIVE_LATENCY_OBSERVATIONS')
    s = sorted(values); point = (len(s)-1)*q; lo = int(point); hi = math.ceil(point)
    return s[lo] + (s[hi]-s[lo])*(point-lo)

class WorkerContext:
    def __init__(self, kind, phase):
        parser = argparse.ArgumentParser()
        for name in ['plan','expected-plan-sha256','admission','expected-admission-sha256','out-dir']:
            parser.add_argument('--'+name,required=True)
        parser.add_argument('--deadline', required=True, type=float)
        self.args = parser.parse_args(); self.started = time.monotonic(); self.kind=kind; self.phase=phase
        need(math.isfinite(self.args.deadline) and 0 < self.args.deadline-self.started <= 570, 'ABSOLUTE_WORKER_DEADLINE')
        need(sorted(os.sched_getaffinity(0)) == [0,1], 'CPU_AFFINITY_0_1')
        self.out = Path(self.args.out_dir); need(self.out.is_absolute() and self.out.is_dir() and not any(p.is_symlink() for p in [self.out,*self.out.parents]), 'PRIVATE_RUN_DIRECTORY')
        self.counters=Counter(); self.goldpaths=set(); self.network=None; self.denied=None; self.progress_fd=None; self.memory_snapshot=None; self.interrupt_signal=None
        signal.signal(signal.SIGTERM,self.defer_signal); signal.signal(signal.SIGINT,self.defer_signal)
        # A payload-name deny is active before plan/admission/source/asset reads.
        sys.addaudithook(self.no_gold)
        self.raw={'schema_version':1,'epoch':57,'kind':kind,'phase':phase,'status':'PARTIAL_NOT_SCORED','plan_sha256':self.args.expected_plan_sha256,'errors':[],'native_state':{},'rows':[],'benchmark_generalization_claim':False,'candidate_promoted':False,'used_confirmation_identities':[],'semantic_gold_decodes':0,'actual_scored_query_ids':[],'actual_feature_query_ids':[],'actual_feature_query_ids_uncertain':False,'reserved_development_query_ids':[],'identity_namespace':None,'quality_metrics':None,'metrics':None,'engineering_qualified':False,'throughput_gate_passed':False,'actual_execution_independently_derived':False,'pilot_completed':False}

    def defer_signal(self,signum,frame):
        self.interrupt_signal=signum

    def no_gold(self,event,args):
        if event != 'open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)): return
        path=os.fsdecode(args[0]); base=Path(path).name.lower()
        if os.path.abspath(path) in self.goldpaths or ('qrels' in base and Path(path).suffix.lower() in {'.tsv','.csv','.jsonl','.parquet','.gz','.zip'}):
            self.counters['qrels_or_label_file_access_attempts'] += 1
            raise RuntimeError('QRELS_LABEL_PAYLOAD_FORBIDDEN_BEFORE_IO')

    def enter(self):
        b=Path(self.args.plan).read_bytes(); need(hashlib.sha256(b).hexdigest()==self.args.expected_plan_sha256,'FROZEN_PLAN_SHA256')
        self.plan=json.loads(b); p=self.plan; p['path']=self.args.plan; p['_sha256']=self.args.expected_plan_sha256
        need(p['epoch']==57 and p['kind']==self.kind and p['phase']==self.phase,'FROZEN_WORKER_ROLE')
        self.goldpaths={os.path.abspath(x) for x in p['gold_paths_forbidden_to_worker']}; need(self.goldpaths,'EXACT_GOLD_DENY_REFS_REQUIRED')
        refs=p['guard_sources']; need(set(refs)==set(GUARD_SHAS),'EXACT_THREE_INHERITED_GUARDS')
        io=load_source('io_guard',refs['io_guard'],GUARD_SHAS['io_guard'])
        g=load_source('inherited_guards',refs['inherited_guards'],GUARD_SHAS['inherited_guards'])
        search=load_source('search_policy',refs['search_policy'],GUARD_SHAS['search_policy'])
        self.network=g.NetworkAudit(self.counters); sys.addaudithook(self.network); self.network.install_socket_guards()
        self.denied=io.DeniedOpenAudit(self.counters,maximum_events=64)
        def guard(event,args):
            self.denied.set_stage(self.network.stage); self.denied(event,args)
        sys.addaudithook(guard); self.memory_snapshot=g.memory_snapshot
        ab=Path(self.args.admission).read_bytes(); need(hashlib.sha256(ab).hexdigest()==self.args.expected_admission_sha256,'ADMISSION_SHA256')
        self.ad=json.loads(ab); ad=self.ad
        need(ad['epoch']==57 and ad['kind']==self.kind and ad['allowed'] is True and ad['plan_sha256']==p['_sha256'] and ad['source_publication_independently_verified'] is True,'ROOT_EXACT_SOURCE_ADMISSION')
        for ref in ad['source_files']: verify(ref)
        need(p['runtime_versions']==RUNTIME_VERSIONS and p['BGE_batch_size']==32 and p['BGE_token_cap']==256,'EXACT_RUNTIME_BATCH_AND_TOKEN_CAP')
        need(p['maximum_native_batches']==1 and p['maximum_outer_seconds']==600 and p['typed_model_initializations_authorized']==0 and p['NLI_forwards_authorized']==0,'BOUNDED_WORKER_RESOURCE_SCOPE')
        baseline=g.memory_snapshot(); self.raw['pre_assets_shared_cgroup']=baseline; g.require_headroom(baseline)
        self.network.set_stage('exact_absent_interpreter_zip_search_policy')
        self.raw['metadata_search_policy']=search.apply_exact_absent_interpreter_zip_policy(p['metadata_search_policy'])
        self.progress_fd=os.open(self.out/'worker-progress.jsonl',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        self.raw['source_commit']=ad['source_commit']; self.raw['identity_namespace']=p['identity_namespace']
        self.raw['reserved_development_query_ids']=p['selected_query_ids'] if self.kind==PILOT_KIND else []
        self.progress('WORKER_ADMITTED'); return self

    def deadline(self):
        need(self.interrupt_signal is None,'DEFERRED_WORKER_SIGNAL'); left=self.args.deadline-time.monotonic(); need(left>0,'WORKER_DEADLINE'); return left

    def progress(self,stage):
        self.deadline(); b=(json.dumps({'stage':stage,'monotonic_seconds':time.monotonic(),'counters':dict(self.counters)},sort_keys=True,allow_nan=False)+'\n').encode()
        need(os.write(self.progress_fd,b)==len(b),'COMPLETE_PROGRESS_WRITE'); os.fsync(self.progress_fd)

    def finish(self,error=None):
        if error is not None:
            code=str(error); self.raw['errors'].append({'type':type(error).__name__,'code':code if code.replace('_','').isalnum() and len(code)<=120 else 'WORKER_FAILURE_PRIVATE_TRACE_RETAINED'})
            p=self.out/'worker.exception.txt'; b=''.join(traceback.format_exception(type(error),error,error.__traceback__)).encode(); fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            try: need(os.write(fd,b)==len(b),'COMPLETE_TRACE_WRITE'); os.fsync(fd)
            finally: os.close(fd)
        if self.network is not None: self.raw['network_audit']=self.network.receipt()
        if self.denied is not None: self.raw['denied_open_audit']=self.denied.receipt()
        for key in ['semantic_gold_decodes','training_calls','calibration_calls','paid_API_calls','GPU_calls','typed_model_initializations','NLI_forwards_attempted','NLI_forwards_completed','qrels_or_label_file_access_attempts','forbidden_model_or_archive_file_access_attempts']:
            self.counters.setdefault(key,0)
        self.raw['counters']=dict(self.counters)
        self.raw['deferred_signal_received']=self.interrupt_signal
        self.raw['pending_uncertainty']={'BGE_initializations':self.counters['BGE_initializations_attempted']!=self.counters['BGE_initializations_completed'],'BGE_forwards_or_postprocessing':self.counters['BGE_forward_calls_attempted']!=self.counters['BGE_forward_calls_completed'],'BGE_texts':self.counters['BGE_texts_attempted']!=self.counters['BGE_texts_completed'],'native':self.raw.get('native_state',{}).get('accounting',{}).get('pending_uncertainty',None)}
        self.raw['native_counter_accounting']=self.raw.get('native_state',{}).get('accounting',None)
        self.raw['timing']={'started_monotonic_seconds':self.started,'completed_monotonic_seconds':time.monotonic(),'worker_wall_seconds_including_init_capture':time.monotonic()-self.started}
        u=resource.getrusage(resource.RUSAGE_SELF); c=resource.getrusage(resource.RUSAGE_CHILDREN)
        self.raw['resources']={'CPU_affinity':sorted(os.sched_getaffinity(0)),'compute_threads':2,'worker_max_RSS_KiB':u.ru_maxrss,'worker_user_seconds':u.ru_utime,'worker_system_seconds':u.ru_stime,'children_max_RSS_KiB':c.ru_maxrss,'children_user_seconds':c.ru_utime,'children_system_seconds':c.ru_stime,'RSS_scope':'Separate SELF and CHILDREN highwater; never summed.','shared_cgroup':self.memory_snapshot() if self.memory_snapshot else None}
        self.raw['latency_and_resources']={'latency':self.raw.get('latency',None),'resources':self.raw['resources'],'latency_scope':self.raw.get('latency_scope',None)}
        if self.progress_fd is not None: os.close(self.progress_fd); self.progress_fd=None
        put(self.out/'raw.json',self.raw)

    def clean_io(self):
        need(self.network.receipt()['admissible'] and self.counters['Node_network_operations_attempted']==0 and self.counters['forbidden_model_or_archive_file_access_attempts']==0 and self.counters['qrels_or_label_file_access_attempts']==0,'NO_FORBIDDEN_IO_GOLD_OR_NETWORK')

def features(ctx, engineering=False):
    p=ctx.plan
    if engineering:
        obj=json.loads(verify(p['engineering_features']).read_bytes())
        need(set(obj)=={'epoch','license','corpus','queries'} and obj['epoch']==57 and obj['license']=='CC0-1.0','NEW_AUTHORED_CC0_ENGINEERING_FEATURES')
        corpus=obj['corpus']; queries=obj['queries']; expected=128
    else:
        corpus=[]
        with verify(p['corpus_features']).open(encoding='utf-8') as f:
            for line in f: corpus.append(json.loads(line))
        qb=verify(p['query_features']).read_bytes(); ctx.raw['actual_feature_query_ids_uncertain']=True
        obj=json.loads(qb); queries=obj['queries']
        if type(queries) is list: ctx.raw['actual_feature_query_ids']=list(dict.fromkeys(q.get('_id') for q in queries if type(q) is dict and type(q.get('_id')) is str))
        ctx.raw['actual_feature_query_ids_uncertain']=False
        need(p.get('prepared_feature_source_epoch')==56 and obj['epoch']==p['prepared_feature_source_epoch'] and obj['selected_ids']==p['selected_query_ids'],'FROZEN_E56_PREPARED_EXPOSED_DEV_IDENTITIES'); expected=5183
        ctx.raw['prepared_feature_source_epoch']=56; ctx.raw['all_reserved_queries_exposed_development_forever']=True
    need(type(corpus) is list and len(corpus)==p['corpus_document_count']==expected and all(type(d) is dict and set(d)=={'_id','title','text'} and all(type(d[k]) is str for k in d) and d['_id'] and d['text'].strip() for d in corpus),'EXACT_COMPLETE_CORPUS_FEATURES')
    need(len({d['_id'] for d in corpus})==expected,'UNIQUE_FULL_CORPUS_IDENTITIES')
    need(type(queries) is list and len(queries)==p['query_count']==16 and all(type(q) is dict and set(q)=={'_id','text'} and all(type(q[k]) is str for k in q) and q['_id'] and q['text'].strip() for q in queries),'EXACT_SIXTEEN_QUERY_FEATURES')
    need([q['_id'] for q in queries]==p['selected_query_ids'] and len(set(p['selected_query_ids']))==16,'EXACT_FROZEN_QUERY_ORDER')
    return corpus,queries

def embed(ctx,groups,label,model_ref,expected_sha,minimum_full_tokens):
    """One exact model initialization; all model groups finish before disposal."""
    p=ctx.plan; c=ctx.counters; need(label in {'FP32','INT8'} and model_ref['sha256']==expected_sha,'FROZEN_MODEL_ROLE_AND_SHA')
    verify(model_ref); need(p['bge_tokenizer']['sha256']==TOKENIZER_SHA,'FROZEN_TOKENIZER_SHA'); verify(p['bge_tokenizer'])
    ctx.network.set_stage('bge_runtime_import')
    import importlib.metadata
    for name,version in RUNTIME_VERSIONS.items(): need(importlib.metadata.version(name)==version,'EXACT_CPU_RUNTIME_VERSION')
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    ctx.deadline(); c['BGE_initializations_attempted']+=1; c[label+'_initializations_attempted']+=1
    need(c['BGE_initializations_attempted']<=p['BGE_initialization_cap'] and c[label+'_initializations_attempted']==1,'MODEL_INIT_CAP_BEFORE_INIT')
    ctx.progress(label+'_INITIALIZE_BEFORE'); started=time.monotonic(); session=None; tok=None; raw=None
    receipt={'label':label,'model_ref':model_ref,'provider':'CPUExecutionProvider','intra_op_threads':2,'inter_op_threads':1,'batch_size':32,'tensor_token_length':256,'embedding_only_truncation':True,'batch_measurements':[],'session_closed':False}
    outputs=[]; audit=[]
    ctx.raw[label+'_runtime']=receipt; ctx.raw[label+'_token_audit']=audit
    try:
        raw=Tokenizer.from_file(p['bge_tokenizer']['path']); raw.no_truncation(); raw.no_padding()
        tok=Tokenizer.from_file(p['bge_tokenizer']['path']); tok.enable_truncation(max_length=256,strategy='longest_first'); tok.enable_padding(length=256,pad_id=0,pad_token='[PAD]')
        opts=ort.SessionOptions(); opts.intra_op_num_threads=2; opts.inter_op_num_threads=1
        session=ort.InferenceSession(model_ref['path'],sess_options=opts,providers=['CPUExecutionProvider'])
        need(session.get_providers()==['CPUExecutionProvider'],'CPU_ONLY_PROVIDER')
        names={x.name for x in session.get_inputs()}; need({'input_ids','attention_mask'}<=names and names<={'input_ids','attention_mask','token_type_ids'},'EXACT_BGE_INPUT_NAMES')
        receipt['initialization_seconds']=time.monotonic()-started; c['BGE_initializations_completed']+=1; c[label+'_initializations_completed']+=1
        ctx.progress(label+'_INITIALIZE_COMPLETE'); ctx.network.set_stage('bge_embedding')
        for group_name,texts in groups:
            need(group_name in {'corpus','queries','full_pilot'} and texts,'NONEMPTY_KNOWN_TEXT_GROUP')
            for start in range(0,len(texts),32):
                ctx.deadline(); batch=texts[start:start+32]; full=raw.encode_batch(batch); used=tok.encode_batch(batch)
                need(len(full)==len(used)==len(batch) and all(len(f.ids)>=minimum_full_tokens for f in full),'FROZEN_FULL_TOKEN_LENGTH_MINIMUM')
                feed={'input_ids':np.asarray([e.ids for e in used],dtype=np.int64),'attention_mask':np.asarray([e.attention_mask for e in used],dtype=np.int64),'token_type_ids':np.asarray([e.type_ids for e in used],dtype=np.int64)}
                need(all(x.shape==(len(batch),256) for x in feed.values()),'EXACT_FIXED_256_INPUT_TENSORS')
                need(c['BGE_forward_calls_attempted']<p['BGE_forward_cap'] and c['BGE_texts_attempted']+len(batch)<=p['BGE_texts_cap'],'GLOBAL_EMBED_CAP_BEFORE_FORWARD')
                c['BGE_forward_calls_attempted']+=1; c['BGE_texts_attempted']+=len(batch); c[label+'_forward_calls_attempted']+=1; c[label+'_texts_attempted']+=len(batch); ctx.progress(label+'_FORWARD_BEFORE')
                before=time.monotonic(); output=session.run(None,{k:x for k,x in feed.items() if k in names})[0]
                forward_seconds=time.monotonic()-before
                c['BGE_forward_calls_returned']+=1; c[label+'_forward_calls_returned']+=1
                measurement={'group':group_name,'group_start':start,'texts':len(batch),'tensor_shape':[len(batch),256],'forward_seconds':forward_seconds,'finite_normalized_postprocessing_completed':False}
                receipt['batch_measurements'].append(measurement)
                need(output.ndim==3 and output.shape[0]==len(batch) and output.shape[1]==256 and output.shape[2]==384,'EXACT_MODEL_HIDDEN_OUTPUT_SHAPE')
                v=output[:,0,:]; need(bool(np.isfinite(v).all()),'FINITE_BGE_CLS_OUTPUT')
                norm=np.linalg.norm(v,axis=1,keepdims=True); need(bool(np.isfinite(norm).all()) and bool((norm>0).all()),'FINITE_POSITIVE_BGE_NORM')
                normalized=(v/norm).astype(np.float32); need(bool(np.isfinite(normalized).all()),'FINITE_NORMALIZED_F32')
                outputs.append(normalized.copy()); del output,v,normalized,norm
                c['BGE_forward_calls_completed']+=1; c['BGE_texts_completed']+=len(batch); c[label+'_forward_calls_completed']+=1; c[label+'_texts_completed']+=len(batch)
                measurement['finite_normalized_postprocessing_completed']=True
                for text,f,u in zip(batch,full,used):
                    count=sum(u.attention_mask); need(0<count<=256,'VALID_CONSUMED_TOKEN_COUNT')
                    audit.append({'group':group_name,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'full_token_count':len(f.ids),'full_token_ids_sha256':hashlib.sha256(json.dumps(f.ids,separators=(',',':')).encode()).hexdigest(),'consumed_token_count':count,'consumed_token_ids_sha256':hashlib.sha256(json.dumps(u.ids[:count],separators=(',',':')).encode()).hexdigest(),'tensor_token_count':256,'truncated':len(f.ids)>count})
                ctx.progress(label+'_FORWARD_COMPLETE')
        matrix=np.concatenate(outputs); need(matrix.shape==(sum(len(g[1]) for g in groups),384),'COMPLETE_EMBED_MATRIX')
    finally:
        if session is not None: del session
        del tok,raw
        gc.collect(); receipt['session_closed']=True; receipt['total_init_forward_tokenize_capture_seconds']=time.monotonic()-started
    ctx.deadline(); return matrix,audit,receipt

def block(matrix):
    need(matrix.ndim==2 and matrix.shape[1]==384 and 1<=matrix.shape[0]<=5183,'F32_BUNDLE_SHAPE')
    data=matrix.astype('<f4',copy=False).tobytes(order='C')
    return {'encoding':'base64','dtype':'float32le','shape':list(matrix.shape),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'data_base64':base64.b64encode(data).decode('ascii')}

def bm25_index(texts):
    rows=[Counter(WORD.findall(text.lower())) for text in texts]; df=Counter()
    for row in rows: df.update(row.keys())
    lengths=[sum(row.values()) for row in rows]; avg=sum(lengths)/len(rows); need(avg>0,'NONEMPTY_BM25_CORPUS')
    return rows,df,lengths,avg

def bm25_query(query,index,ids):
    rows,df,lengths,avg=index; terms=Counter(WORD.findall(query.lower())); scores=[]; n=len(rows)
    for row,dl in zip(rows,lengths):
        score=0.0
        for term,qf in terms.items():
            tf=row.get(term,0)
            if tf: score+=qf*math.log(1+(n-df[term]+.5)/(df[term]+.5))*tf*2.5/(tf+1.5*(1-.75+.75*dl/avg))
        scores.append(score)
    order=sorted(range(n),key=lambda i:(-scores[i],ids[i]))[:10]
    return order,[scores[i] for i in order]

def cosine_query(query,corpus,ids):
    import numpy as np
    docs=corpus.astype(np.float64); q=query.astype(np.float64)
    scores=(docs@q)/(np.linalg.norm(docs,axis=1)*np.linalg.norm(q)); need(bool(np.isfinite(scores).all()),'FINITE_EXACT_SAME_F32_COSINE')
    order=sorted(range(len(ids)),key=lambda i:(-float(scores[i]),ids[i]))[:10]
    return order,[float(scores[i]) for i in order]

def matched_forecasts(ctx,corpus,queries,matrix):
    p=ctx.plan; ids=[d['_id'] for d in corpus]; qids=[q['_id'] for q in queries]; count=len(corpus); vectors=matrix[:count]; qvectors=matrix[count:]
    need(matrix.shape==(count+16,384),'INT8_ONLY_MATCHED_VECTOR_GEOMETRY')
    bridge=load_source('e57_native_bridge',p['native_bridge_source'])
    bundle={'schema_version':1,'epoch':57,'kind':ctx.kind,'phase':ctx.phase,'plan_sha256':p['_sha256'],'corpus':{'ids':ids,'vectors':block(vectors)},'queries':{'ids':qids,'vectors':block(qvectors)}}
    ctx.network.set_stage('native_retrieval'); native=bridge.native(p,bundle,ctx.out,ctx.counters,ctx.deadline,ctx.progress,ctx.raw['native_state']); ctx.raw['native_result']=native
    need([row['query_id'] for row in native['queries']]==qids,'NATIVE_QUERY_ORDER')
    before=time.monotonic(); index=bm25_index([d['title']+' '+d['text'] for d in corpus]); ctx.raw['BM25_corpus_index_seconds']=time.monotonic()-before
    ranks={'native':{},'exact_cosine':{},'bm25':{}}; latencies={key:[] for key in ranks}; allowed=set(ids)
    for i,q in enumerate(queries):
        ctx.deadline(); n=native['queries'][i]; methods=[('native',n['ids'],n['scores'],n['elapsed_seconds'])]
        for arm,fn in [('exact_cosine',lambda:cosine_query(qvectors[i],vectors,ids)),('bm25',lambda:bm25_query(q['text'],index,ids))]:
            before=time.monotonic(); order,scores=fn(); elapsed=time.monotonic()-before; methods.append((arm,[ids[j] for j in order],scores,elapsed))
        for arm,docids,scores,elapsed in methods:
            need(len(docids)==len(set(docids))==len(scores)==10 and set(docids)<=allowed and all(type(v) in (float,int) and math.isfinite(v) for v in scores) and type(elapsed) in (float,int) and math.isfinite(elapsed) and elapsed>=0,'FINITE_COMPLETE_MATCHED_TOP10')
            rankcodes={docid:10-rank for rank,docid in enumerate(docids)}; ranks[arm][q['_id']]=rankcodes; latencies[arm].append(elapsed)
            ctx.raw['rows'].append({'query_id':q['_id'],'arm':arm,'ids':docids,'raw_method_scores':scores,'rank_scores':rankcodes,'query_search_seconds':elapsed,'predictions_are_retrieval_ranks_not_semantic_probabilities':True})
        ctx.counters['matched_query_forecasts_completed']+=1; ctx.progress('MATCHED_QUERY_FORECAST_COMPLETE')
    need(ctx.counters['matched_query_forecasts_completed']==16 and len(ctx.raw['rows'])==48 and ctx.raw['native_state']['accounting']['qualified'],'ALL48_FORECASTS_AND_ACTUAL_NATIVE_ACCOUNTING')
    ctx.raw['latency']={arm:{'observations':16,'p50_seconds':percentile(xs,.5),'p95_seconds':percentile(xs,.95)} for arm,xs in latencies.items()}
    ctx.raw['latency_scope']='Per-query search only; shared BGE/corpus/index/init/capture/cleanup separately recorded. Inclusive supervisor/CLI clock is batch cap.'
    return ranks

def engineering_gate(plan,fp32,int8,drift):
    gate=plan['engineering_gate']; need(set(gate)=={'minimum_speedup','maximum_projection_seconds','reserve_seconds','projection_safety_factor','numerical_cosine_floor'},'EXACT_FROZEN_ENGINEERING_GATE')
    need(gate['minimum_speedup']==1.25 and gate['maximum_projection_seconds']==420 and gate['reserve_seconds']==90 and gate['projection_safety_factor']==2 and type(gate['numerical_cosine_floor']) in (float,int) and gate['numerical_cosine_floor']==0.97,'FROZEN_GATE_LIMITS')
    fs=[x['forward_seconds'] for x in fp32['batch_measurements'] if x['group']=='corpus']; qs=[x['forward_seconds'] for x in int8['batch_measurements'] if x['group']=='corpus']
    need(len(fs)==len(qs)==4 and all(math.isfinite(x) and x>0 for x in fs+qs),'FOUR_OBSERVED_POSITIVE_BATCH_TIMES')
    speedup=sum(fs)/sum(qs); projected=2*max(qs)*math.ceil(5199/32)+int8['initialization_seconds']+90
    need(len(drift)==128 and all(math.isfinite(v) and -1<=v<=1.000001 for v in drift),'128_FINITE_PAIRED_DRIFT_OBSERVATIONS')
    flags={'observed_throughput_speedup_pass':speedup>=gate['minimum_speedup'],'conservative_runtime_projection_pass':projected<=gate['maximum_projection_seconds'],'paired_cosine_drift_pass':min(drift)>=gate['numerical_cosine_floor']}
    return {'frozen_gate':gate,'FP32_128_forward_seconds':sum(fs),'INT8_128_forward_seconds':sum(qs),'observed_FP32_over_INT8_throughput_ratio':speedup,'conservative_5199_projection_seconds':projected,'projection_formula':'2*max(four INT8 corpus32 forward seconds)*ceil(5199/32)+observed INT8 init seconds+90','paired_normalized_vector_cosines':drift,'minimum_paired_cosine':min(drift),'mean_paired_cosine':sum(drift)/128,**flags,'local_gate_passed':all(flags.values()),'pilot_execution_authorized_by_this_report':False,'limitations':'Authored stress inputs only. Projection excludes unmeasured corpus specifics; parent independent gate/admission and 600s supervisor remain required.'}
