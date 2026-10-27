"""Separately admitted, label-blind tokenizer-only catalog preparation.

No ONNX runtime, model or native initialization; no Gold or confirmation path.
"""
import argparse
import importlib.metadata
import importlib.util
import os
import signal
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime_common import *


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--admission',required=True);ap.add_argument('--expected-admission-sha256',required=True);args=ap.parse_args()
    a_ref=typed_ref(args.admission);need(a_ref['sha256']==args.expected_admission_sha256,'EXACT_PREPARATION_ADMISSION');a=load_ref(a_ref)
    need(a['schema_version']==1 and a['epoch']==58 and a['kind']==KIND and a['mode']=='CATALOG' and a['phase']=='BLIND_CATALOG_PREPARATION_ONLY','ONLY_LABEL_BLIND_PREPARATION')
    need(a['allowed_ids']==list(DEV_IDS) and a['train_ids']==a['calibration_ids']==a['fresh_confirmation_ids']==[] and a['scientific_authorization_granted'] is False and a['Gold_authorization_granted'] is False,'NO_SCI_GOLD_CF')
    for ref in a['source_files']:read_ref(ref)
    need(typed_ref(__file__) in a['source_files'] and typed_ref(Path(__file__).parent/'runtime_common.py') in a['source_files'],'PREPARATION_SOURCE_CLOSURE')
    fence=imported(a['fence_source'],'e58_preparation_fence');fence.verify_owner(a['owner_ref'])
    q=load_ref(a['independent_preparation_approval_ref']);need(q['checks_passed'] is True and q['phase']==a['phase'] and q['preparation_plan_sha256']==a['plan_ref']['sha256'] and q['preparation_source_freeze_sha256']==a['source_freeze_ref']['sha256'] and q['owner_ref']==a['owner_ref'],'INDEPENDENT_FIXED_PREPARATION_APPROVAL')
    freeze=load_ref(a['source_freeze_ref']);need(all(r in freeze['files'] for r in a['source_files']) and a['plan_ref'] in freeze['files'] and a['fence_source'] in freeze['files'],'ALL_LOCAL_PREPARATION_SOURCE_FROZEN')
    plan=load_ref(a['plan_ref']);need(plan['epoch']==58 and plan['kind']==KIND and plan['allowed_ids']==list(DEV_IDS),'EXACT_SOURCE_PREPARATION_PLAN')
    need(os.sched_getaffinity(0)<={0,1} and os.environ.get('TOKENIZERS_PARALLELISM')=='false','CPU2_NO_TOKENIZER_POOL')
    output=safe(a['output_directory']);output.mkdir(mode=0o700,exist_ok=True);put(output/'CATALOG_ATTEMPT.json',{'epoch':58,'mode':a['mode'],'admission_ref':a_ref,'automatic_retry':False})
    report={'schema_version':1,'epoch':58,'kind':KIND,'mode':a['mode'],'status':'FAILED_CLOSED_NO_RETRY','errors':[],'NLI_forward_calls':0,'native_corpus_batches':0,'Gold_labels_decoded':0,'paid_API_calls':0,'GPU_calls':0,'selected_query_ids':list(DEV_IDS)}
    started=time.monotonic()
    def alarm(_s,_f):raise TimeoutError('BLIND_CATALOG_INNER100_SECONDS')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(100)
    try:
        for name,version in a['runtime_versions'].items():need(importlib.metadata.version(name)==version,'EXACT_RUNTIME_DISTRIBUTION')
        nt=a['NLI_tokenizer'];bt=a['BGE_tokenizer'];read_ref(nt);read_ref(bt)
        need(nt['bytes']==8656624 and bt['sha256']==BGE_TOKENIZER_SHA and bt['bytes']==711396,'EXACT_PINNED_TOKENIZER_ASSETS')
        # No network SDK exists; the tokenizer reads two verified local JSON files.
        import tokenizers
        nli=tokenizers.Tokenizer.from_file(nt['path']);bge=tokenizers.Tokenizer.from_file(bt['path']);nli.no_truncation();nli.no_padding();bge.no_truncation();bge.no_padding()
        d=load_ref(a['data_report_ref']);need(d['epoch']==58 and d['kind']==KIND and d['status']=='READY_LABEL_BLIND_ONE_DOC17_ALL17' and d['errors']==[] and d['features']==a['features_ref'] and d['actual_feature_case_ids']==list(DEV_IDS) and d['semantic_gold_decodes']==0,'EXACT_BLIND_DATA_CUSTODY_RECEIPT')
        bank=load_ref(a['features_ref']);doc=bank['documents'][0];text=''.join(u['sentence'] for u in doc['evidence']);need(doc['normalized_document_sha256']=='4c3e5f88c04ef038b431d74af57d26cb7a620301e020cae2f5ab334c32f494b4' and sha(' '.join(text.casefold().split()).encode())==doc['normalized_document_sha256'],'EXACT_PINNED_DOC17_NORMALIZED_IDENTITY');catalog=make_catalog(bank,nli,bge);fence.verify_owner(a['owner_ref'])
        need(time.monotonic()-started<=a['maximum_seconds']<=120,'BOUNDED_PREPARATION_WALL')
        report.update(status='BLIND_CATALOG_READY',catalog_ref=put(output/'CATALOG.json',catalog),frozen_window_count=catalog['frozen_window_count'],tokenizers_initialized=2,all17_claims_capacity_checked=True,source_features_ref=a['features_ref'])
    except Exception as e:report['errors'].append({'type':type(e).__name__,'code':str(e) if str(e).replace('_','').isalnum() else 'PREPARATION_FAILURE'})
    signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;put(output/'CATALOG_REPORT.json',report)
    return 0 if report['status']=='BLIND_CATALOG_READY' else 2
if __name__=='__main__':raise SystemExit(main())
