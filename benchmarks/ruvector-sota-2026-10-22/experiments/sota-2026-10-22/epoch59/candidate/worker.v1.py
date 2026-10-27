"""E59 once-only, offline native evidence + fixed typedNLI CPU worker.

Never reads labels. Engineering and DEV are distinct conditional corpus batches.
"""
import argparse
import base64
from collections import deque
import gc
import importlib.metadata
import math
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime_common import *


def guard_sources(c):
    need(c['schema_version']==1 and c['epoch']==59 and c['kind']==KIND and c['phase'] in ('engineering','dev'),'EXACT_WORKER_KIND')
    need(os.sched_getaffinity(0)<={0,1} and len(os.sched_getaffinity(0))<=2 and not sys.flags.optimize,'CPU2_NONOPTIMIZED')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','UV_THREADPOOL_SIZE'):
        need(os.environ.get(key)=='2','FIXED_CPU_THREAD_ENV')
    need(os.environ.get('TOKENIZERS_PARALLELISM')=='false','NO_TOKENIZER_POOL')
    for r in c['source_files']:read_ref(r)
    need(typed_ref(__file__) in c['source_files'] and typed_ref(Path(__file__).parent/'runtime_common.py') in c['source_files'],'WORKER_SOURCE_CLOSURE')
    f=imported(c['fence_source'],'e59_worker_owner_fence');f.verify_owner(c['owner'])
    plan=load_ref(c['plan']);freeze=load_ref(c['source_freeze']);pub=load_ref(c['source_publication']);q=load_ref(c['independent_admission'])
    need(plan['epoch']==59 and plan['kind']==KIND and plan['train_ids']==plan['calibration_ids']==plan['fresh_confirmation_ids']==[],'FROZEN_NO_TRAIN_CAL_CF')
    need(pub['checks_passed'] is True and pub['verified_actual_remote_readback'] is True and pub['source_plan_sha256']==c['plan']['sha256'] and pub['source_freeze_sha256']==c['source_freeze']['sha256'],'ACTUAL_SOURCE_PUBLICATION_BEFORE_ANY_MODEL_NATIVE')
    need(q['checks_passed'] is True and q['scientific_execution_allowed'] is True and q['phase']==c['phase'] and q['plan_ref']==c['plan'] and q['source_freeze_ref']==c['source_freeze'] and q['owner_ref']==c['owner'],'INDEPENDENT_EXACT_SCI_ADMISSION')
    need(all(r in freeze['runtime_sources'] for r in c['source_files']),'ALL_RUNTIME_SOURCE_REFS_FROZEN')
    need(all(r in freeze['runtime_sources'] for r in [*c['assets'].values(),c['NLI_model_config']]),'ALL_MODEL_TOKENIZER_CONFIG_REFS_FROZEN')
    need(type(c['frozen_window_count']) is int and 1<=c['frozen_window_count']<=24,'FROZEN_BLIND_W_CAP24')
    if c['phase']=='dev':
        e=load_ref(c['engineering_qualification']);need(e['engineering_qualified'] is True and e['throughput_gate_passed'] is True and e['projection_seconds']<=420 and e['plan_ref']==c['plan'],'CONDITIONAL_SECOND_CORPUS_ONLY_AFTER_ENGINEERING_GATE')
    for name,version in c['runtime_versions'].items():need(importlib.metadata.version(name)==version,'EXACT_RUNTIME_DISTRIBUTION')
    return f


def block_network(events):
    def deny(*args,**kwargs):
        events.values['Python_network_operations_attempted']+=1;events.event('FORBIDDEN_NETWORK');raise RuntimeError('PYTHON_NETWORK_FORBIDDEN')
    def audit(event,args):
        if event in ('socket.connect','socket.connect_ex','socket.bind','socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr'):deny()
    sys.addaudithook(audit);socket.create_connection=deny;socket.getaddrinfo=deny


class LocalModel:
    def __init__(self,kind,c,events):
        self.kind=kind;self.events=events;self.session=None;self.tokenizer=None;self.init_seconds=0;self.forward_seconds=[]
        a=c['assets'];model=a['NLI_model' if kind=='NLI' else 'BGE_model'];tokenizer=a['NLI_tokenizer' if kind=='NLI' else 'BGE_tokenizer']
        start=time.monotonic();read_ref(model);read_ref(tokenizer)
        need(model['sha256']==(MODEL_SHA if kind=='NLI' else BGE_SHA) and model['bytes']==(82823065 if kind=='NLI' else 34014426),'EXACT_LOCAL_MODEL_WEIGHT')
        if kind=='BGE':need(tokenizer['sha256']==BGE_TOKENIZER_SHA and tokenizer['bytes']==711396,'EXACT_BGE_TOKENIZER')
        else:
            need(tokenizer['bytes']==1356048 and tokenizer['sha256']==NLI_TOKENIZER_SHA and hashlib.sha1(b'blob 1356048\0'+read_ref(tokenizer)).hexdigest()==NLI_TOKENIZER_GIT_SHA,'EXACT_NEW_NLI_TOKENIZER_SIZE')
            cfg=load_ref(c['NLI_model_config']);need(cfg['id2label']=={'0':'contradiction','1':'entailment','2':'neutral'} and cfg['max_position_embeddings']==514 and cfg['model_type']=='roberta' and cfg['num_hidden_layers']==6 and cfg['hidden_size']==768 and cfg['pad_token_id']==1,'OFFICIAL_TYPED_HEAD_LABEL_MAP')
        snap=memory_snapshot();need(snap['headroom_bytes'] is not None and snap['headroom_bytes']>=c['minimum_model_headroom_bytes'],'FROZEN_MODEL_HEADROOM')
        events.attempt(kind+'_tokenizer_initializations')
        import tokenizers
        self.tokenizer=tokenizers.Tokenizer.from_file(tokenizer['path']);self.tokenizer.no_padding();self.tokenizer.no_truncation();events.complete(kind+'_tokenizer_initializations')
        events.attempt(kind+'_model_initializations')
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena=False;options.add_session_config_entry('session.intra_op.allow_spinning','0');options.add_session_config_entry('session.inter_op.allow_spinning','0')
        self.session=ort.InferenceSession(model['path'],sess_options=options,providers=['CPUExecutionProvider'])
        need(self.session.get_providers()==['CPUExecutionProvider'],'CPU_PROVIDER_ONLY')
        self.inputs={x.name:x for x in self.session.get_inputs()};need({'input_ids','attention_mask'}<=set(self.inputs)<={'input_ids','attention_mask','token_type_ids'} and all(x.type=='tensor(int64)' for x in self.inputs.values()),'EXACT_INTEGER_TOKEN_INPUTS')
        outs=self.session.get_outputs();self.output=next((x.name for x in outs if x.name==('logits' if kind=='NLI' else 'last_hidden_state')),None);need(self.output is not None,'OFFICIAL_HEAD_OUTPUT_REQUIRED')
        events.complete(kind+'_model_initializations');self.init_seconds=time.monotonic()-start;events.event(kind+'_READY',initialization_seconds=self.init_seconds)
    def _feed(self,encodings):
        import numpy as np
        need(1<=len(encodings)<=(8 if self.kind=='NLI' else 16),'FIXED_FORWARD_BATCH_CAP')
        if self.kind=='NLI':need(all(all(t==0 for t in e['token_type_ids']) for e in encodings),'ROBERTA_SINGLE_TYPE_VOCAB')
        feed={}
        for name in self.inputs:
            values=[]
            for e in encodings:
                need(len(e['input_ids'])<=512,'NO_HIDDEN_TOKEN_TRUNCATION');pad=512-len(e['input_ids']);pad_value=1 if self.kind=='NLI' and name=='input_ids' else 0;values.append(e[name]+[pad_value]*pad)
            feed[name]=np.asarray(values,dtype=np.int64)
        need(all(x.shape==(len(encodings),512) for x in feed.values()),'PADDED512_UNTRUNCATED_INPUTS')
        return feed
    def nli(self,pairs):
        need(self.kind=='NLI','NLI_HEAD_ONLY');es=[]
        for premise,claim in pairs:
            self.events.attempt('pair_tokenizations');es.append(whole_pair(self.tokenizer,premise,claim));self.events.complete('pair_tokenizations')
        return self._forward(es)
    def _forward(self,es):
        import numpy as np
        base='NLI' if self.kind=='NLI' else 'BGE';self.events.attempt(base+'_forward_calls');self.events.attempt('NLI_pair_inferences' if base=='NLI' else 'BGE_texts',len(es))
        feed=self._feed(es);self.last_input_receipts={name:{'shape':list(a.shape),'dtype':'int64le','bytes':int(a.nbytes),'sha256':sha(a.astype('<i8',copy=False).tobytes(order='C'))} for name,a in feed.items()};started=time.monotonic();out=self.session.run([self.output],feed)[0];elapsed=time.monotonic()-started;self.forward_seconds.append(elapsed)
        need(np.isfinite(out).all(),'FINITE_MODEL_OUTPUTS');need(out.dtype==np.float32,'OFFICIAL_FLOAT32_MODEL_OUTPUT');self.last_output_receipt={'shape':list(out.shape),'dtype':'float32le','bytes':int(out.nbytes),'sha256':sha(out.astype('<f4',copy=False).tobytes(order='C'))}
        if self.kind=='NLI':
            need(out.shape==(len(es),3),'NATIVE_TYPED_THREE_LOGITS');value=[[float(x) for x in row] for row in out]
            for row in value:stable_scores(row)
        else:
            need(out.shape==(len(es),512,384),'BGE_LAST_HIDDEN_CLS_WIDTH384');v=out[:,0,:].astype(np.float32);norm=np.linalg.norm(v,axis=1,keepdims=True);need(np.isfinite(norm).all() and (norm>0).all(),'FINITE_POSITIVE_BGE_NORMS');value=(v/norm).astype(np.float32)
            need(np.isfinite(value).all() and np.max(np.abs(np.linalg.norm(value,axis=1)-1))<=1e-5,'NORMALIZED_FLOAT32_BGE_CLS')
        self.events.complete('NLI_pair_inferences' if base=='NLI' else 'BGE_texts',len(es));self.events.complete(base+'_forward_calls');self.events.event(base+'_FORWARD_OBSERVED',batch_size=len(es),sequence_length=512,forward_seconds=elapsed)
        return value
    def embed(self,texts):
        import numpy as np
        need(self.kind=='BGE','BGE_EMBED_ONLY');rows=[]
        for offset in range(0,len(texts),16):
            es=[]
            for text in texts[offset:offset+16]:
                self.events.attempt('BGE_tokenizations');es.append(whole_single(self.tokenizer,text));self.events.complete('BGE_tokenizations')
            rows.append(self._forward(es))
        return np.concatenate(rows,axis=0)
    def close(self):self.session=None;self.tokenizer=None;gc.collect()


def block(v):
    raw=v.astype('<f4',copy=False).tobytes(order='C');return {'encoding':'base64','dtype':'float32le','shape':list(v.shape),'bytes':len(raw),'sha256':sha(raw),'data_base64':base64.b64encode(raw).decode('ascii')}


def native_retrieve(c,events,documents,queries,output,state,deadline):
    bge=LocalModel('BGE',c,events);vectors=bge.embed([x['text'] for x in documents]);qvectors=bge.embed([QUERY_PREFIX+x['claim'] for x in queries]);init=bge.init_seconds;bge.close();events.event('BGE_CLOSED_BEFORE_NATIVE_AND_NLI')
    bundle={'schema_version':1,'epoch':59,'kind':KIND,'phase':c['phase'],'plan_sha256':c['plan']['sha256'],'corpus':{'ids':[x['id'] for x in documents],'vectors':block(vectors)},'queries':{'ids':[x['id'] for x in queries],'vectors':block(qvectors)}}
    adapter=imported(c['native_adapter_source'],'e59_real_native_adapter');external=output/'native';external.mkdir(mode=0o700)
    config=dict(c['native_config']);need(config['plan_sha256']==c['plan']['sha256'] and config['selected_query_ids']==[x['id'] for x in queries] and config['corpus_document_count']==len(documents),'ONE_FROZEN_NATIVE_CORPUS_GEOMETRY')
    native=adapter.run_native(config,bundle,external,events.event,deadline,counters=events.values,state=state)
    return native,init


def roundrobin_forecasts(catalog,native,nli,events):
    W=catalog['frozen_window_count'];windows=catalog['windows'];cases=catalog['cases'];source=list(range(W));ranks={r['query_id']:r for r in native['queries']};context=''.join(w['text'] for w in windows)
    states=[];diagnostics=[]
    for case in cases:
        diag=numeric_diagnostic(case['claim'],context);diagnostics.append({'case_id':case['id'],**diag})
        if diag['flags']:events.values['numerical_flagged_cases']+=1;events.event('NUMERICAL_CASE_FLAGGED',case_id=case['id'],flags=diag['flags'])
        ranked=ranks[case['id']];native_order=[int(i.split(':')[-1]) for i in ranked['ids']]
        need(len(set(native_order))==len(native_order) and set(native_order)<=set(source),'NATIVE_WITHIN_SUPPLIED_SOURCE_ONLY')
        fallback=bool(ranked['retrieval_miss'] or len(native_order)!=W)
        if fallback:native_order=source[:]
        for arm in ARMS:
            states.append({'case':case,'arm':arm,'order':native_order[:] if arm=='native_order' else source[:],'cursor':0,'logits':[],'window_ids':[],'flags':diag['flags'],'forward_calls':0,'started':None,'row':None,'force_full_coverage':arm=='native_order' and fallback})
    pending=deque(states);batch_records=[]
    while pending:
        tasks=[];planned={id(s):0 for s in pending};rotation=0
        # One next window per active case-arm per rotation. Revisit only after
        # every active state had an opportunity; fill8 without output caching.
        while len(tasks)<8 and pending:
            s=pending.popleft();available=len(s['order'])-s['cursor']-planned[id(s)]
            if available>0:
                wid=s['order'][s['cursor']+planned[id(s)]];tasks.append((s,wid));planned[id(s)]+=1
            pending.append(s);rotation+=1
            if all(len(x['order'])-x['cursor']-planned[id(x)]==0 for x in pending):break
            need(rotation<=51*8,'BOUNDED_ROUNDROBIN_PLANNING')
        need(tasks,'NONEMPTY_REAL_INFERENCE_TASKS');batch_start=time.monotonic();seen=[]
        for s,wid in tasks:
            if s['started'] is None:s['started']=batch_start
            if id(s) not in seen:s['forward_calls']+=1;seen.append(id(s))
        logits=nli.nli([(windows[wid]['text'],s['case']['claim']) for s,wid in tasks]);batch_id=len(batch_records)
        record={'batch_index':batch_id,'pairs':[{'case_id':s['case']['id'],'arm':s['arm'],'window_id':wid} for s,wid in tasks],'inference_and_tokenization_seconds':time.monotonic()-batch_start,'forward_seconds':nli.forward_seconds[-1],
            'input_tensor_receipts':nli.last_input_receipts,'output_tensor_receipt':nli.last_output_receipt,'model_weight_sha256':MODEL_SHA,'label_order':list(LABELS),
            'logit_rows':[{'batch_id':batch_id,'row_index':j,'case_id':s['case']['id'],'arm':s['arm'],'window_id':wid,'raw_logits3':row,'source_window_sha256':windows[wid]['text_sha256'],'claim_input_sha256':s['case']['input_sha256']} for j,((s,wid),row) in enumerate(zip(tasks,logits))]}
        batch_records.append(record);events.event('NLI_BATCH_OUTPUTS_OBSERVED',batch_record=record);batch_end=time.monotonic()
        for (s,wid),row in zip(tasks,logits):s['logits'].append(row);s['window_ids'].append(wid);s['cursor']+=1
        keep=deque()
        while pending:
            s=pending.popleft()
            if not s['logits']:
                keep.append(s);continue
            covered=s['cursor']==W;exit_now=s['arm']!='full_reference' and not s['force_full_coverage'] and may_exit(s['logits'],s['flags'],covered)
            if covered or exit_now:
                scores=aggregate(s['logits']);need(scores['prediction']!='NotMentioned' or covered,'NEI_REQUIRES_EXHAUSTIVE_SUPPLIED_SOURCE')
                row={'case_id':s['case']['id'],'arm':s['arm'],**scores,'elapsed_seconds':batch_end-s['started'],'actual_model_forward_calls':s['forward_calls'],'actual_pair_inferences':len(s['window_ids']),'context_window_ids':s['window_ids'],'all_source_windows_covered':covered,'early_exit':bool(exit_now),'numeric_flags':s['flags']}
                s['row']=row;events.values['forecasts_completed']+=1;events.event('FORECAST_COMPLETE',row=row)
            else:keep.append(s)
        pending=keep
    rows=[s['row'] for s in states];need(len(rows)==51 and all(r is not None for r in rows),'ALL51_ACTUAL_FORECASTS')
    need(len(batch_records)<=math.ceil(3*17*W/8),'FROZEN_GLOBAL_FORWARD_BOUND')
    return rows,batch_records,diagnostics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--expected-config-sha256',required=True);ap.add_argument('--deadline-monotonic-seconds',type=float,required=True);args=ap.parse_args();started=time.monotonic()
    cfg_ref=typed_ref(args.config);need(cfg_ref['sha256']==args.expected_config_sha256,'EXACT_WORKER_CONFIG');c=load_ref(cfg_ref);need(started<args.deadline_monotonic_seconds<=started+580,'WORKER580_INCLUSIVE_DEADLINE')
    output=safe(c['output_directory']);output.mkdir(mode=0o700,exist_ok=True);put(output/'WORKER_ATTEMPT.json',{'schema_version':1,'epoch':59,'kind':KIND,'config_ref':cfg_ref,'process_identity':process_identity(),'automatic_retry':False})
    events=Events(output/'progress.jsonl',args.deadline_monotonic_seconds);state={'process_identities':[process_identity()]};result={'schema_version':1,'epoch':59,'kind':KIND,'phase':c['phase'],'plan_sha256':c['plan']['sha256'],'source_freeze_sha256':c['source_freeze']['sha256'],'status':'FAILED_CLOSED_NO_RETRY','errors':[],'rows':[],'counters':events.values,'selected_query_ids':[],'reserved_ids':[],'owned_process_identities':state['process_identities']};nli=None
    try:
        fence=guard_sources(c);events.event('ALL_SOURCE_ADMISSIONS_VALIDATED');block_network(events)
        if c['phase']=='engineering':
            inputs=load_ref(c['engineering_inputs']);need(inputs['license']=='CC0-1.0' and inputs['authored_for']=='E59_ENGINEERING_ONLY' and len(inputs['documents'])==128 and len(inputs['queries'])==16 and len(inputs['nli_pairs'])==32,'AUTHORED_ENGINEERING_GEOMETRY')
            documents=inputs['documents'];queries=inputs['queries'];result['selected_query_ids']=[x['id'] for x in queries];result['reserved_ids']=result['selected_query_ids'][:]
            native,bge_init=native_retrieve(c,events,documents,queries,output,state,args.deadline_monotonic_seconds);fence.verify_owner(c['owner']);nli=LocalModel('NLI',c,events);records=[]
            for offset in range(0,32,8):
                pairs=inputs['nli_pairs'][offset:offset+8];logits=nli.nli([(x['premise'],x['hypothesis']) for x in pairs]);records.extend({'id':x['id'],'logits':l,**stable_scores(l)} for x,l in zip(pairs,logits))
            forwards=nli.forward_seconds;need(len(forwards)==4,'EXACT4_FULL8_ENGINEERING_FORWARDS');model_init=bge_init+nli.init_seconds;W=c['frozen_window_count'];projection=2*math.ceil(3*17*W/8)*max(forwards)+model_init+90
            result.update(native_results_ref=state['raw_result_ref'],engineering_pairs=records,throughput_measurements={'full8_NLI_forward_seconds':forwards,'model_initialization_seconds':{'nli':nli.init_seconds,'bge':bge_init},'measured_modelinit_total':model_init,'frozen_DEV_window_count':W,'projection_seconds':projection,'frozen_projection_limit_seconds':420},status='ENGINEERING_EXECUTED_PENDING_INDEPENDENT_AUDIT')
        else:
            catalog=load_ref(c['catalog_ref']);validate_catalog(catalog);need(catalog['frozen_window_count']==c['frozen_window_count'] and catalog['selected_query_ids']==list(DEV_IDS),'EXACT_BLIND_FROZEN_DEV_CATALOG');documents=[{'id':'source-window:'+str(w['window_id']),'text':w['text']} for w in catalog['windows']];queries=catalog['cases'];result['selected_query_ids']=list(DEV_IDS);result['reserved_ids']=list(DEV_IDS)
            native,bge_init=native_retrieve(c,events,documents,queries,output,state,args.deadline_monotonic_seconds);fence.verify_owner(c['owner']);nli=LocalModel('NLI',c,events);rows,batches,diagnostics=roundrobin_forecasts(catalog,native,nli,events)
            forecasts={'schema_version':1,'epoch':59,'kind':KIND,'phase':'dev','plan_sha256':c['plan']['sha256'],'source_freeze_sha256':c['source_freeze']['sha256'],'labels':list(LABELS),'arms':list(ARMS),'selected_query_ids':list(DEV_IDS),'reserved_ids':list(DEV_IDS),'rows':rows}
            result.update(rows=rows,sealed_forecasts_ref=put(output/'SEALED_FORECASTS.json',forecasts),native_results_ref=state['raw_result_ref'],NLI_batch_records=batches,numerical_diagnostics=diagnostics,catalog_ref=c['catalog_ref'],status='DEV_FORECASTS_SEALED_NO_GOLD')
        nli.close();nli=None;fence.verify_owner(c['owner']);need(events.values['paid_API_calls']==events.values['GPU_calls']==events.values['Gold_labels_decoded']==events.values['Python_network_operations_attempted']==events.values['Node_network_operations_attempted']==0,'ZERO_NETWORK_PAID_GPU_GOLD')
        need(time.monotonic()<args.deadline_monotonic_seconds,'ACTUAL_WORKER_DEADLINE');events.event('WORKER_COMPLETE')
    except Exception as e:
        result['errors'].append({'type':type(e).__name__,'code':str(e) if str(e).replace('_','').isalnum() and len(str(e))<=160 else 'WORKER_FAILED'})
    finally:
        if nli is not None:nli.close()
        events.close();result['counters']=dict(events.values);result['owned_process_identities']=state['process_identities'];result['native_adapter_state']={k:v for k,v in state.items() if k!='process_identities'};result['latency_and_resources']={'whole_worker_elapsed_seconds':time.monotonic()-started,'per_forecast_elapsed_semantics':'first submitted batch start through decision; includes interbatch queue and tokenization/forward; excludes prefirstsubmit queue; not singlequery deployment latency','per_forecast_p50_seconds':percentile([r['elapsed_seconds'] for r in result['rows']],.5) if result['rows'] else None,'per_forecast_p95_seconds':percentile([r['elapsed_seconds'] for r in result['rows']],.95) if result['rows'] else None,'resource_usage':resources()};result['progress_ref']=typed_ref(output/'progress.jsonl');put(output/'RESULT.actual.json',result)
    return 0 if not result['errors'] else 2
if __name__=='__main__':raise SystemExit(main())
