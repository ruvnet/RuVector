"""E58 fixed DEV-only three-way scorer. No model/native/network imports.

The CLI opens already-joined Gold only after exact sealed-forecast, independent
actual-execution and process-closure admission. Pure evaluate() is inert and is
tested solely with authored examples; it is never a benchmark runner.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import time
from pathlib import Path

KIND = 'REAL_NATIVE_TYPED_NLI_COVERAGE_ORDER_DEV'
LABELS = ('Contradiction', 'Entailment', 'NotMentioned')
ARMS = ('native_order', 'source_order', 'full_reference')
HYPOTHESES = ('nda-1','nda-10','nda-11','nda-12','nda-13','nda-15',
              'nda-16','nda-17','nda-18','nda-19','nda-2','nda-20',
              'nda-3','nda-4','nda-5','nda-7','nda-8')
IDS = tuple('contractnli:eced6528:dev:17:' + x for x in HYPOTHESES)
TOP = {'schema_version','epoch','kind','phase','plan_sha256',
       'source_freeze_sha256','labels','arms','selected_query_ids','reserved_ids','rows'}
ROW = {'case_id','arm','probabilities','log_probabilities','elapsed_seconds',
       'actual_model_forward_calls','actual_pair_inferences','context_window_ids',
       'all_source_windows_covered','early_exit','numeric_flags','prediction'}
TOL = 1e-6


def need(value, code):
    if not value:
        raise ValueError(code)


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(rows):
    result = {}
    for k, v in rows:
        need(k not in result, 'DUPLICATE_JSON_KEY')
        result[k] = v
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def safe(path):
    p = Path(path)
    need(p.is_absolute() and p.resolve() == p and
         not any(x.is_symlink() for x in (p, *p.parents)), 'CANONICAL_NONSYMLINK_PATH')
    return p


def body(path):
    p = safe(path)
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        a = os.fstat(f.fileno())
        need(stat.S_ISREG(a.st_mode) and a.st_size <= 64*1024*1024, 'BOUNDED_REGULAR_FILE')
        raw = f.read(64*1024*1024+1)
        b = os.fstat(f.fileno())
    need(len(raw) <= 64*1024*1024 and (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns) ==
         (b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns), 'STABLE_FILE')
    return raw


def read_ref(ref):
    need(type(ref) is dict and set(ref) == {'path','bytes','sha256'} and
         type(ref['bytes']) is int and ref['bytes'] >= 0 and
         re.fullmatch('[0-9a-f]{64}', ref['sha256']) is not None, 'EXACT_TYPED_REF')
    raw = body(ref['path'])
    need(len(raw) == ref['bytes'] and sha(raw) == ref['sha256'], 'REF_BYTES_SHA')
    return raw


def file_ref(path):
    raw = body(path)
    return {'path':str(safe(path)), 'bytes':len(raw), 'sha256':sha(raw)}


def put(path, value):
    p = safe(path)
    need(p.parent.is_dir(), 'EXISTING_PRIVATE_OUTPUT_DIRECTORY')
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    d = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(d)
    finally: os.close(d)
    return file_ref(p)


def percentile(values, q):
    need(values and all(numeric(x) and x >= 0 for x in values), 'VALID_LATENCIES')
    ordered = sorted(values); at = (len(ordered)-1)*q
    lo = math.floor(at); hi = math.ceil(at)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(at-lo)


def validate_forecasts(forecasts, case_ids=IDS, arms=ARMS):
    case_ids = tuple(case_ids); arms = tuple(arms)
    need(case_ids == IDS and arms == ARMS, 'ONLY_FROZEN_DOC17_ALL17_DEV')
    need(type(forecasts) is dict and set(forecasts) == TOP and
         forecasts['schema_version'] == 1 and forecasts['epoch'] == 58 and
         forecasts['kind'] == KIND and forecasts['phase'] == 'dev' and
         forecasts['labels'] == list(LABELS) and forecasts['arms'] == list(arms) and
         forecasts['selected_query_ids'] == list(case_ids) and
         forecasts['reserved_ids'] == list(case_ids), 'SEALED_FORECAST_SCHEMA')
    for key in ('plan_sha256','source_freeze_sha256'):
        need(type(forecasts[key]) is str and re.fullmatch('[0-9a-f]{64}', forecasts[key]) is not None,
             'FORECAST_SOURCE_HASH')
    rows = forecasts['rows']
    need(type(rows) is list and len(rows) == 51 and
         [(r.get('case_id'),r.get('arm')) for r in rows] ==
         [(cid,arm) for cid in case_ids for arm in arms], 'CASE_MAJOR_ALL51_FORECASTS')
    by_arm = {a:[] for a in arms}; full_windows = None
    for r in rows:
        need(type(r) is dict and set(r) == ROW, 'EXACT_FORECAST_ROW_FIELDS')
        p, lp = r['probabilities'], r['log_probabilities']
        need(type(p) is list and type(lp) is list and len(p) == len(lp) == 3 and
             all(numeric(x) and 0 < x <= 1 for x in p) and
             all(numeric(x) and x <= 0 for x in lp) and
             abs(math.fsum(p)-1) <= TOL and
             all(abs(math.exp(l)-v) <= 1e-12 for l,v in zip(lp,p)), 'POSITIVE_STABLE_THREEWAY_SOFTMAX')
        winner = max(range(3), key=lambda j:p[j])
        need(r['prediction'] == LABELS[winner], 'EXACT_ARGMAX_LABEL')
        need(numeric(r['elapsed_seconds']) and r['elapsed_seconds'] >= 0, 'FINITE_ROW_ELAPSED')
        for k in ('actual_model_forward_calls','actual_pair_inferences'):
            need(type(r[k]) is int and r[k] > 0, 'ACTUAL_NONZERO_INFERENCE_COUNTS')
        ws = r['context_window_ids']
        need(type(ws) is list and 1 <= len(ws) <= 24 and len(set(ws)) == len(ws) and
             all(type(x) is int and 0 <= x <= 23 for x in ws) and
             r['actual_pair_inferences'] == len(ws), 'EXACT_UNIQUE_CONSUMED_WINDOW_IDS')
        need(type(r['all_source_windows_covered']) is bool and type(r['early_exit']) is bool,
             'COVERAGE_EXIT_BOOLEANS')
        flags = r['numeric_flags']
        allowed_flags=('claim_has_numerals','claim_numeral_not_in_supplied_context')
        need(type(flags) is list and flags in ([],[allowed_flags[0]],list(allowed_flags)),
             'BOUNDED_NUMERIC_DIAGNOSTICS')
        need(not (flags and r['early_exit']), 'NUMERICAL_FLAGS_DISABLE_EXIT_ONLY')
        need(not r['early_exit'] or r['prediction'] != 'NotMentioned', 'NO_NEI_EARLY_EXIT')
        need(r['prediction'] != 'NotMentioned' or r['all_source_windows_covered'],
             'NEI_REQUIRES_ALL_SUPPLIED_CONTEXT')
        if r['arm'] == 'full_reference':
            need(r['all_source_windows_covered'] and not r['early_exit'] and
                 ws == list(range(len(ws))), 'FULL_REFERENCE_EXHAUSTIVE_SOURCE_ORDER')
            if full_windows is None: full_windows = ws
            need(ws == full_windows, 'SAME_FULL_REFERENCE_WINDOW_CATALOG')
        by_arm[r['arm']].append(r)
    need(full_windows is not None, 'FULL_REFERENCE_EXISTS')
    for r in rows:
        need(set(r['context_window_ids']) <= set(full_windows) and
             r['all_source_windows_covered'] == (set(r['context_window_ids']) == set(full_windows)),
             'ALL_COVERAGE_FLAGS_MATCH_SHARED_CATALOG')
        if r['arm'] == 'source_order':
            need(r['context_window_ids'] == list(range(len(r['context_window_ids']))),
                 'SOURCE_ARM_FIXED_ORDER_PREFIX')
    return by_arm


def evaluate(forecasts, gold_rows, case_ids=IDS, arms=ARMS):
    """Score one immutable 17-case DEV evaluation; no fit/selection or mutation."""
    by_arm = validate_forecasts(forecasts, case_ids, arms)
    need(type(gold_rows) is list and len(gold_rows) == 17 and
         all(type(r) is dict and set(r) == {'case_id','label'} for r in gold_rows) and
         [r['case_id'] for r in gold_rows] == list(case_ids) and
         all(r['label'] in LABELS for r in gold_rows), 'EXACT_OFFICIAL_GOLD_ROWS')
    ys = {r['case_id']:LABELS.index(r['label']) for r in gold_rows}
    metrics = {}; per_case = {}
    for arm, rows in by_arm.items():
        scores=[]; confusion=[[0]*3 for _ in range(3)]; errors=[]
        for row in rows:
            y=ys[row['case_id']]; p=row['probabilities']; pred=LABELS.index(row['prediction'])
            brier=math.fsum((v-(1 if j==y else 0))**2 for j,v in enumerate(p))
            nll=-row['log_probabilities'][y]
            need(numeric(brier) and numeric(nll) and brier >= 0 and nll >= 0, 'FINITE_PROPER_SCORES')
            scored={'case_id':row['case_id'],'correct':pred==y,'gold':LABELS[y],
                    'prediction':row['prediction'],'Brier':brier,'NLL':nll}
            scores.append(scored); confusion[y][pred]+=1
            if pred != y: errors.append(scored)
        count=len(scores); correct=sum(s['correct'] for s in scores)
        times=[r['elapsed_seconds'] for r in rows]
        metrics[arm]={'count':count,'correct':correct,'accuracy':correct/count,
                      'Brier':math.fsum(s['Brier'] for s in scores)/count,
                      'NLL':math.fsum(s['NLL'] for s in scores)/count,
                      'confusion_matrix_gold_rows_prediction_columns':confusion,
                      'errors':errors,'error_count':len(errors),
                      'latency_seconds':{'p50':percentile(times,.5),'p95':percentile(times,.95),
                                         'total':math.fsum(times),'count':len(times)},
                      'case_forward_participations':sum(r['actual_model_forward_calls'] for r in rows),
                      'actual_pair_inferences':sum(r['actual_pair_inferences'] for r in rows),
                      'early_exit_count':sum(r['early_exit'] for r in rows),
                      'full_context_count':sum(r['all_source_windows_covered'] for r in rows)}
        per_case[arm]={s['case_id']:s for s in scores}
    paired={}
    for comparator in ('source_order','full_reference'):
        pairs=[]
        for cid in case_ids:
            n=per_case['native_order'][cid]; c=per_case[comparator][cid]
            pairs.append({'case_id':cid,'native_correct':n['correct'],
                          'comparator_correct':c['correct'],
                          'native_minus_comparator_Brier':n['Brier']-c['Brier'],
                          'native_minus_comparator_NLL':n['NLL']-c['NLL']})
        paired[comparator]={'native_wins':sum(p['native_correct'] and not p['comparator_correct'] for p in pairs),
                            'native_losses':sum(p['comparator_correct'] and not p['native_correct'] for p in pairs),
                            'both_correct':sum(p['native_correct'] and p['comparator_correct'] for p in pairs),
                            'both_wrong':sum(not p['native_correct'] and not p['comparator_correct'] for p in pairs),
                            'paired_rows':pairs,
                            'mean_native_minus_comparator_Brier':math.fsum(p['native_minus_comparator_Brier'] for p in pairs)/17,
                            'mean_native_minus_comparator_NLL':math.fsum(p['native_minus_comparator_NLL'] for p in pairs)/17}
    n,s,f=(metrics[a] for a in ARMS)
    checks={'native_correct_at_least_source_plus_one':n['correct'] >= s['correct']+1,
            'native_Brier_not_worse_source':n['Brier'] <= s['Brier']+TOL,
            'native_NLL_not_worse_source':n['NLL'] <= s['NLL']+TOL,
            'native_correct_not_worse_full':n['correct'] >= f['correct'],
            'native_Brier_not_worse_full':n['Brier'] <= f['Brier']+TOL,
            'native_NLL_not_worse_full':n['NLL'] <= f['NLL']+TOL}
    return {'schema_version':1,'epoch':58,'kind':KIND,'phase':'dev','labels':list(LABELS),
            'arms':metrics,'paired_comparisons':paired,
            'model_forward_accounting':{'row_field':'actual_model_forward_calls',
                'row_field_semantics':'Number of shared model batches in which this case-arm participated.',
                'arm_metric':'case_forward_participations',
                'global_physical_forward_source':'raw_worker.counters.NLI_forward_calls_completed',
                'global_physical_forward_audit':'Independently reconcile raw NLI_batch_records unique batch_index and durable model-forward progress.',
                'participation_sums_are_not_physical_forward_counts':True},
            'mechanism_quality_gate':{'tolerance':TOL,'checks':checks,'quality_checks_passed':all(checks.values())},
            'actual_scored_case_ids':list(case_ids),'actual_scored_forecasts':51,
            'uncertainty':{'document_clusters':1,'claims':17,'independent_document_CI':None,
                           'reason':'One NDA cluster with dependent claims; no independent-document generalization interval or rank inference.'},
            'train_ids':[],'calibration_ids':[],'fresh_confirmation_ids':[],
            'candidate_promotion_allowed':False,'official_result_or_rank':None,
            'scope':'DEV_MECHANISM_ONLY_PENDING_QUALIFIED_CONFIRMATION'}


def physical_closure(audit):
    need(audit['checks_passed'] is True and audit['all_owned_processes_closed'] is True and
         type(audit['actual_execution_independently_derived']) is dict,
         'INDEPENDENT_ACTUAL_COUNTS_AND_PHYSICAL_CLOSURE')
    need(audit['kernel_boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip() and
         audit['PID_namespace'] == os.readlink('/proc/self/ns/pid'), 'CURRENT_BOOT_NAMESPACE')
    identities=audit['owned_process_identities']; need(type(identities) is list and identities,'ALL_OWNED_IDENTITIES')
    groups=set(); sessions=set()
    for identity in identities:
        need(all(type(identity[k]) is int and identity[k]>1 for k in ('pid','startticks','pgid','sid')),
             'EXACT_PROCESS_IDENTITY')
        groups.add(identity['pgid']); sessions.add(identity['sid'])
        p=Path('/proc')/str(identity['pid'])
        try: fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        except FileNotFoundError: continue
        if int(fields[19]) != identity['startticks']: continue
        need(fields[0]=='Z' and not any(int(x) for x in (p/'statm').read_text().split()) and
             not list((p/'fd').iterdir()), 'OWNED_PROCESS_PHYSICALLY_CLOSED')
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        except (FileNotFoundError,ProcessLookupError): continue
        need(fields[0] in ('Z','X') or (int(fields[2]) not in groups and int(fields[3]) not in sessions),
             'OWNED_GROUP_OR_SESSION_STILL_LIVE')


def validate_joined_gold(joined,sealed_forecasts_ref):
    """Validate the custody-v5 joined-value artifact, separate from events."""
    need(joined['schema_version']==1 and joined['epoch']==58 and joined['kind']==KIND and joined['phase']=='dev' and
         joined['sealed_forecasts_ref']==sealed_forecasts_ref and joined['ids']==list(IDS) and
         joined['label_order']==list(LABELS) and type(joined['semantic_labels_decoded']) is int and
         joined['semantic_labels_decoded']==17, 'EXACT_OFFICIAL_LATE_GOLD_JOIN')
    return joined['rows']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--expected-config-sha256',required=True)
    args=ap.parse_args();config_ref=file_ref(args.config)
    need(config_ref['sha256']==args.expected_config_sha256,'EXACT_SCORING_CONFIG')
    c=parse(read_ref(config_ref));need(c['epoch']==58 and c['kind']==KIND and c['phase']=='dev','DEV_ONLY_SCORER_CONFIG')
    # All admission/source/forecast bodies are read BEFORE the joined Gold body.
    read_ref(c['owner']);read_ref(c['fence_source'])
    spec=importlib.util.spec_from_file_location('e58_scoring_fence',c['fence_source']['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.verify_owner(c['owner'])
    admission=parse(read_ref(c['root_admission']));audit=parse(read_ref(c['independent_actual_audit']))
    need(admission['epoch']==58 and admission['phase']=='dev' and admission['scoring_allowed'] is True and
         admission['sealed_forecasts_ref']==c['sealed_forecasts'] and admission['joined_gold_ref']==c['joined_gold'] and
         admission['independent_actual_audit_ref']==c['independent_actual_audit'] and admission['owner_ref']==c['owner'],
         'EXACT_ROOT_LATE_SCORING_ADMISSION')
    need(audit['sealed_forecasts_ref']==c['sealed_forecasts'],'AUDIT_EXACT_SEALED_FORECASTS')
    physical_closure(audit)
    forecasts=parse(read_ref(c['sealed_forecasts']));validate_forecasts(forecasts)
    need(admission['plan_ref']['sha256']==forecasts['plan_sha256'] and
         admission['source_freeze_ref']['sha256']==forecasts['source_freeze_sha256'], 'FORECAST_SOURCE_BINDING')
    for k in ('plan_ref','source_freeze_ref'):read_ref(admission[k])
    read_ref(c['scorer_source'])
    need(Path(c['scorer_source']['path'])==Path(__file__).resolve(), 'EXACT_OPERATIONAL_SCORER_SOURCE')
    freeze=parse(read_ref(admission['source_freeze_ref']))
    need(c['scorer_source'] in freeze['runtime_sources'] and
         any(r['bytes']==c['scorer_source']['bytes'] and r['sha256']==c['scorer_source']['sha256']
             for r in freeze['files']), 'SCORER_PUBLIC_BODY_AND_OPERATIONAL_REF_IN_FROZEN_SOURCE')
    out=safe(c['output']);put(str(out)+'.attempt.json',{'epoch':58,'config_ref':config_ref,'automatic_retry':False,'Gold_body_opened':False})
    m.verify_owner(c['owner']);physical_closure(audit)
    joined=parse(read_ref(c['joined_gold']))
    gold_rows=validate_joined_gold(joined,c['sealed_forecasts'])
    result=evaluate(forecasts,gold_rows)
    result.update(sealed_forecasts_ref=c['sealed_forecasts'],joined_gold_ref=c['joined_gold'],
                  independent_actual_audit_ref=c['independent_actual_audit'],config_ref=config_ref,
                  operational_acceptance_checks_passed=audit['checks_passed'],
                  mechanism_quality_gate_with_operational_checks_passed=(
                      result['mechanism_quality_gate']['quality_checks_passed'] and audit['checks_passed']))
    m.verify_owner(c['owner']);ref=put(out,result)
    print(json.dumps({'status':'DEV_SCORED_NO_PROMOTION','result':ref},sort_keys=True));return 0


if __name__=='__main__':
    raise SystemExit(main())
