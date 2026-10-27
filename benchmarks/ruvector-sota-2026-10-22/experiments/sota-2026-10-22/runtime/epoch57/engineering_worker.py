"""E57 authored-stress CPU quantization qualification; no benchmark/Gold input."""
import hashlib
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runtime_common as r

KIND=r.ENGINEERING_KIND
PHASE='engineering'

def main():
    ctx=r.WorkerContext(KIND,PHASE); error=None; rc=2
    try:
        ctx.enter(); p=ctx.plan
        r.need(p['BGE_initialization_cap']==2 and p['BGE_forward_cap']==9 and p['BGE_texts_cap']==272,'EXACT_ENGINEERING_EMBEDDING_CAPS')
        r.need(p['bge_fp32_model']['sha256']==r.FP32_SHA and p['bge_model']['sha256']==r.INT8_SHA,'PAIRED_FP32_AND_INT8_SOURCE_MODELS')
        corpus,queries=r.features(ctx,engineering=True)
        texts=[d['title']+' '+d['text'] for d in corpus]; qtexts=[r.BGE_PREFIX+q['text'] for q in queries]
        ctx.raw['actual_engineering_corpus_ids']=[d['_id'] for d in corpus]; ctx.raw['actual_engineering_query_ids']=[q['_id'] for q in queries]
        ctx.raw['engineering_features_ref']=p['engineering_features']; ctx.raw['actual_engineering_attempted']=True
        fp,audit_fp,receipt_fp=r.embed(ctx,[('corpus',texts)],'FP32',p['bge_fp32_model'],r.FP32_SHA,256)
        ctx.raw['FP32_runtime']=receipt_fp; ctx.raw['FP32_token_audit']=audit_fp
        r.need(receipt_fp['session_closed'] is True,'FP32_RELEASED_BEFORE_INT8_INITIALIZATION')
        quant,audit_q,receipt_q=r.embed(ctx,[('corpus',texts),('queries',qtexts)],'INT8',p['bge_model'],r.INT8_SHA,256)
        ctx.raw['INT8_runtime']=receipt_q; ctx.raw['INT8_token_audit']=audit_q
        import numpy as np
        a=fp.astype(np.float64); b=quant[:128].astype(np.float64)
        drift=((a*b).sum(axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))).tolist()
        r.need(bool(np.isfinite(a).all()) and bool(np.isfinite(b).all()),'FINITE_PAIRED_EMBEDDING_MATRICES')
        ctx.raw['paired_corpus_token_identity_equal']=all({k:v for k,v in x.items() if k!='group'}=={k:v for k,v in y.items() if k!='group'} for x,y in zip(audit_fp,audit_q[:128]))
        r.need(ctx.raw['paired_corpus_token_identity_equal'],'SAME_AUTHORED_INPUT_TENSOR_IDENTITIES')
        ctx.raw['engineering_gate']=r.engineering_gate(p,receipt_fp,receipt_q,drift)
        ctx.raw['throughput_gate_passed']=ctx.raw['engineering_gate']['observed_throughput_speedup_pass'] and ctx.raw['engineering_gate']['conservative_runtime_projection_pass']
        ctx.raw['throughput_measurements']={'FP32':receipt_fp,'INT8':receipt_q,'engineering_gate':ctx.raw['engineering_gate']}
        ctx.raw['reason']='LOCAL_ENGINEERING_GATE_PASSED_REQUIRES_INDEPENDENT_REVIEW' if ctx.raw['engineering_gate']['local_gate_passed'] else 'LOCAL_ENGINEERING_GATE_FAILED_NO_PILOT_ADMISSION'
        # FP32 vectors are never a retrieval arm. The native/exact/BM25 arms all
        # consume exactly the same INT8 corpus/query vectors and source texts.
        del fp,a,b
        ranks=r.matched_forecasts(ctx,corpus,queries,quant)
        overlap=[]
        for q in p['selected_query_ids']:
            overlap.append(len(set(ranks['native'][q])&set(ranks['exact_cosine'][q]))/10)
        ctx.raw['native_exact_top10_overlap_on_authored_inputs']={'observations':16,'fractions':overlap,'mean':sum(overlap)/16,'scope':'Engineering ANN fidelity only; no relevance labels or benchmark accuracy.'}
        c=ctx.counters
        r.need(c['BGE_initializations_completed']==2 and c['BGE_forward_calls_completed']==9 and c['BGE_texts_completed']==272 and c['FP32_forward_calls_completed']==4 and c['FP32_texts_completed']==128 and c['INT8_forward_calls_completed']==5 and c['INT8_texts_completed']==144,'EXACT_OBSERVED_ENGINEERING_COMPLETIONS')
        ctx.clean_io()
        seal={'schema_version':1,'epoch':57,'kind':KIND,'phase':PHASE,'plan_sha256':p['_sha256'],'source_commit':ctx.ad['source_commit'],'role':'AUTHORED_CC0_ENGINEERING_INPUTS_ONLY','engineering_features_ref':p['engineering_features'],'rankings':ranks,'rows':ctx.raw['rows'],'quality_metrics':None,'semantic_gold_decodes_before_seal':0,'candidate_promoted':False,'benchmark_generalization_claim':False,'used_confirmation_identities':[],'identity_namespace':p['identity_namespace']}
        ctx.raw['forecast_seal']=r.put(ctx.out/'forecasts.json',seal)
        ctx.raw['status']='QUANTIZED_ENGINEERING_FORECASTS_SEALED_NOT_SCORED'; rc=0
    except BaseException as exc: error=exc
    finally: ctx.finish(error)
    return rc

if __name__=='__main__': raise SystemExit(main())
