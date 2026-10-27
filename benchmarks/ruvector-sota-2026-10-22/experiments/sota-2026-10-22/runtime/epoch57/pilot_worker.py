"""E57 optional frozen SciFact DEV retrieval; INT8 only and no Gold/scoring."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runtime_common as r

KIND=r.PILOT_KIND
PHASE='retrieval_development'

def main():
    ctx=r.WorkerContext(KIND,PHASE); error=None; rc=2
    try:
        ctx.enter(); p=ctx.plan; ctx.raw['actual_pilot_attempted']=True
        r.need(p['BGE_initialization_cap']==1 and p['BGE_forward_cap']==163 and p['BGE_texts_cap']==5199,'EXACT_FULL_PILOT_EMBEDDING_CAPS')
        r.need(p['bge_model']['sha256']==r.INT8_SHA and 'bge_fp32_model' not in p,'INT8_ONLY_PILOT_MODEL')
        corpus,queries=r.features(ctx)
        texts=[d['title']+' '+d['text'] for d in corpus]+[r.BGE_PREFIX+q['text'] for q in queries]
        matrix,audit,receipt=r.embed(ctx,[('full_pilot',texts)],'INT8',p['bge_model'],r.INT8_SHA,1)
        ctx.raw['INT8_runtime']=receipt; ctx.raw['embedding_audit']=audit
        ctx.raw['corpus_features_ref']=p['corpus_features']; ctx.raw['query_features_ref']=p['query_features']
        ranks=r.matched_forecasts(ctx,corpus,queries,matrix)
        c=ctx.counters
        r.need(c['BGE_initializations_completed']==1 and c['BGE_forward_calls_completed']==163 and c['BGE_texts_completed']==5199 and c['INT8_forward_calls_completed']==163 and c['INT8_texts_completed']==5199,'EXACT_OBSERVED_FULL_PILOT_EMBEDDING_COMPLETIONS')
        ctx.clean_io()
        forecasts={'schema_version':1,'epoch':57,'kind':KIND,'phase':PHASE,'role':'PROJECT_DEVELOPMENT_SOURCE_OFFICIAL_TEST_SUBSET','selected_query_ids':p['selected_query_ids'],'corpus_document_count':5183,'plan_sha256':p['_sha256'],'source_commit':ctx.ad['source_commit'],'corpus_features_ref':p['corpus_features'],'query_features_ref':p['query_features'],'rankings':ranks,'rows':ctx.raw['rows'],'metric_score_encoding':'Strict descending rankcodes10..1; not calibrated probabilities or native distance reinterpretation.','tie_policy':'Exactcosine/BM25 full-corpus equal scores doc-ID lexical; native returned adjacent equal-score groups only, global boundary ties unverified.','quality_metrics':None,'semantic_gold_decodes_before_seal':0,'used_confirmation_identities':[],'benchmark_generalization_claim':False,'candidate_promoted':False,'identity_namespace':p['identity_namespace'],'actual_feature_query_ids':ctx.raw['actual_feature_query_ids'],'actual_scored_query_ids':[]}
        ctx.raw['forecast_seal']=r.put(ctx.out/'forecasts.json',forecasts); ctx.raw['status']='RETRIEVAL_DEVELOPMENT_FORECASTS_SEALED_NOT_SCORED'; ctx.raw['pilot_completed']=True; ctx.raw['reason']='SEALED_FORECASTS_REQUIRE_PHYSICAL_CLOSURE_BEFORE_GOLD'; rc=0
    except BaseException as exc: error=exc
    finally: ctx.finish(error)
    return rc

if __name__=='__main__': raise SystemExit(main())
