"""E59 metadata-only reporting and immutable closure; no scientific entrypoint.

Root reviews the exact concrete JSON configuration before this CLI. Actual
counts come only from matched raw records and independent reconstructed audits.
No frozen expected count fills an absent or partial observation.
"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import sys

W = Path('/dev/shm/ruvector-sota-2026-10-22/epoch59-9b677a916983')
R = W / 'project'
P = 'experiments/sota-2026-10-22/epoch59/'
A = 'experiments/sota-2026-10-22/audit/epoch59/'
T = 'experiments/sota-2026-10-22/runtime/epoch59/'
ROOTS = (P, A, T)
BASE = '9b677a91698352bd563f189fee861d6ff2a1ad3a'
KIND = 'REAL_NATIVE_TYPED_READER_BOUNDED_RESEARCH'
ARMS = ('native_order', 'source_order', 'full_reference')
LABELS = ('Contradiction', 'Entailment', 'NotMentioned')
NAMESPACE = None
PAIRED_COUNTERS = (
    'NLI_tokenizer_initializations', 'BGE_tokenizer_initializations',
    'NLI_model_initializations', 'BGE_model_initializations',
    'NLI_forward_calls', 'NLI_pair_inferences', 'BGE_forward_calls', 'BGE_texts',
    'pair_tokenizations', 'BGE_tokenizations', 'native_index_initializations',
    'native_insertions', 'native_queries', 'native_len_calls',
    'native_binary_loads', 'native_version_calls', 'native_processes',
)
COUNTER_KEYS = frozenset(
    [p + '_' + suffix for p in PAIRED_COUNTERS for suffix in ('attempted', 'completed')]
    + ['native_hit_source_validations', 'Node_network_operations_attempted',
       'forecasts_completed', 'numerical_flagged_cases',
       'Python_network_operations_attempted', 'paid_API_calls', 'GPU_calls',
       'Gold_labels_decoded']
)
ACTIVE_GUARD = None

def need(ok, message):
    if not ok:
        raise ValueError(message)

def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()

def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'duplicate metadata JSON key')
        result[key] = value
    return result

def parse(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite metadata JSON')))

def body(path):
    path = Path(path)
    need(path.is_absolute() and all(not p.is_symlink() for p in (path, *path.parents)), 'absolute nonsymlink metadata')
    before = path.stat(); need(path.is_file() and before.st_size <= 8388608, 'bounded regular metadata')
    raw = path.read_bytes(); after = path.stat()
    sig = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(sig(before) == sig(after) and len(raw) == after.st_size, 'stable metadata read')
    return raw

def ref(path, project=False):
    path = Path(path); raw = body(path)
    return {'path': path.relative_to(R).as_posix() if project else str(path),
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def canonical(row, project=False):
    need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}
         and type(row['path']) is str and type(row['bytes']) is int and row['bytes'] >= 0
         and type(row['sha256']) is str and len(row['sha256']) == 64
         and all(c in '0123456789abcdef' for c in row['sha256']), 'typed3 metadata reference')
    if project:
        p = Path(row['path'])
        need(not p.is_absolute() and '\\' not in row['path'] and all(x not in ('', '.', '..') for x in row['path'].split('/')), 'canonical project reference')
        path = R / p
    else:
        path = Path(row['path'])
    raw = body(path)
    need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'metadata bytes/hash')
    return raw

def checked(row, project=False):
    return parse(canonical(row, project))

def same(a, b):
    if type(a) is not type(b): return False
    if type(a) is dict: return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if type(a) is list: return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b

def activate(config):
    global ACTIVE_GUARD, NAMESPACE
    need(ACTIVE_GUARD is None, 'one concrete reporting activation')
    need(config['epoch'] == 59 and config['kind'] == KIND and config['source_checkpoint'] == BASE, 'current reporter context')
    sf = checked(config['source_freeze'], True)
    need(sf['epoch'] == 59 and sf['kind'] == KIND and sf['source_checkpoint'] == BASE, 'source freeze identity')
    frozen = {row['path']: row for row in sf['files']}
    need(len(frozen) == len(sf['files']), 'unique source freeze rows')
    for name in ('plan', 'engineering_plan', 'DEV_plan'):
        row = config[name]; canonical(row, True)
        need(frozen.get(row['path']) == row, 'reporting study plan is source frozen: ' + name)
    plan = checked(config['plan'], True)
    ids = plan['development_ids']
    need(type(ids) is list and len(ids) == len(set(ids)) == 17
         and all(type(case) is str for case in ids), 'one frozen17-case DEV document')
    namespace = ids[0].rsplit(':', 1)[0]
    need(namespace == 'contractnli:eced6528:train:473'
         and all(case.rsplit(':', 1)[0] == namespace for case in ids)
         and plan['development_namespace'] == namespace, 'one exact source-frozen DEV namespace')
    NAMESPACE = namespace
    self_ref = config['reporter_source']
    need(self_ref['path'] == P + 'reporting/report_actual.py'
         and frozen.get(self_ref['path']) == self_ref, 'reporter is source frozen')
    canonical(self_ref, True)
    running = ref(Path(__file__).resolve())
    need((running['bytes'], running['sha256']) == (self_ref['bytes'], self_ref['sha256']), 'executed reporter source matches freeze')
    guard_ref = config['publication_guard']
    need(guard_ref['path'] == str(W / 'publication-helpers-g27/guard.py'), 'fixed operational publication guard')
    canonical(guard_ref)
    public_guard = frozen[P + 'publication/guard.py']
    canonical(public_guard, True)
    need((guard_ref['bytes'], guard_ref['sha256']) == (public_guard['bytes'], public_guard['sha256']), 'frozen operational guard exact')
    spec = importlib.util.spec_from_file_location('e59_reporting_exact_guard', guard_ref['path'])
    need(spec is not None and spec.loader is not None, 'publication guard loader')
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    guard.owner_fence(); ACTIVE_GUARD = guard

def put_bytes(path, raw):
    need(ACTIVE_GUARD is not None, 'live reviewed reporting context')
    ACTIVE_GUARD.owner_fence()
    path = Path(path)
    need(path.is_relative_to(R) and any(path.is_relative_to(R / root) for root in ROOTS), 'current public output scope')
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'safe public output path')
    need(type(raw) is bytes, 'exact public byte payload')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return ref(path, True)

def put(path, value):
    return put_bytes(path, encoded(value))

def identities(value):
    need(type(NAMESPACE) is str and type(value) is list
         and all(type(x) is str and x.startswith(NAMESPACE + ':') for x in value)
         and len(set(value)) == len(value), 'unique source-frozen DEV identities')
    return list(value)

def counters(value):
    need(type(value) is dict and set(value) == COUNTER_KEYS, 'exact 42-field actual counter dictionary')
    need(all(type(n) is int and n >= 0 for n in value.values()), 'actual counters are nonnegative integers, never bool/float')
    return copy.deepcopy(value)

def class_counts(value):
    need(type(value) is dict and set(value) == set(LABELS)
         and all(type(n) is int and n >= 0 for n in value.values()), 'actual canonical Gold class counts')
    return copy.deepcopy(value)

def counter_policy_violations(value):
    observed = counters(value); violations = []
    for key in PAIRED_COUNTERS:
        if observed[key + '_completed'] > observed[key + '_attempted']:
            violations.append('completion_exceeds_attempt:' + key)
    for key in ('paid_API_calls', 'GPU_calls', 'Gold_labels_decoded', 'Node_network_operations_attempted', 'Python_network_operations_attempted'):
        if observed[key] != 0: violations.append('forbidden_worker_activity:' + key)
    return violations

def actual(raw_ref, audit_ref, phase, plan_ref, source_freeze):
    raw = checked(raw_ref, True); audit = checked(audit_ref, True)
    need(type(raw['epoch']) is int and type(audit['epoch']) is int
         and raw['epoch'] == audit['epoch'] == 59 and raw['kind'] == KIND and raw['phase'] == phase, 'actual current phase identity')
    canonical(plan_ref, True)
    need(raw['plan_sha256'] == plan_ref['sha256'] and raw['source_freeze_sha256'] == source_freeze['sha256'], 'actual plan/source binding')
    need(audit['independent_review_complete'] is True and audit['all_owned_processes_closed'] is True
         and audit['actual_results_ref'] == raw_ref, 'independent actual observation and process closure')
    observed = counters(raw['counters']); reconstructed = counters(audit['actual_execution_independently_derived'])
    need(same(observed, reconstructed), 'raw and independently reconstructed actual counts differ')
    need(type(raw['rows']) is list and type(raw['errors']) is list and type(raw['status']) is str
         and type(audit['status']) is str and type(audit['reason']) is str, 'explicit actual rows/errors/status/reason')
    for key in ('native_counter_accounting', 'latency_and_resources'):
        need(type(audit[key]) is dict, 'independent actual ' + key)
    return raw, audit, reconstructed

def summarize(config):
    old = checked(config['initial_CONT']); source = checked(config['source_receipt'], True)
    need(source['epoch'] == 59 and source['phase'] == 'source' and source['parent'] == BASE
         and source['verified_actual_full_tree'] is True
         and source['source_plan_sha256'] == config['plan']['sha256']
         and source['source_freeze_sha256'] == config['source_freeze']['sha256'], 'actual SOURCE receipt')
    raw, eng, eng_counts = actual(config['engineering_result'], config['engineering_audit'], 'engineering',
                                  config['engineering_plan'], config['source_freeze'])
    need(type(eng['engineering_qualified']) is bool and type(eng['throughput_gate_passed']) is bool
         and type(eng['throughput_measurements']) is dict, 'engineering verdict schema')
    data = checked(config['data_preparation_result'], True); data_review = checked(config['data_preparation_review'], True)
    need(type(data['epoch']) is int and data['epoch'] == 59 and data['kind'] == KIND
         and data['source_checkpoint'] == BASE and data['identity_namespace'] == NAMESPACE
         and type(data['semantic_gold_decodes']) is int and data['semantic_gold_decodes'] == 0
         and data['used_confirmation_identities'] == [], 'blind data preparation scope')
    need(data_review['checks_passed'] is True and data_review['safe_public_projection_ref'] == config['data_preparation_result'], 'independent exact public DATA projection')
    origin = data_review['actual_data_report_ref']; origin_data = checked(origin)
    need((origin['bytes'], origin['sha256']) == (config['data_preparation_result']['bytes'], config['data_preparation_result']['sha256']) and same(origin_data, data), 'byte-identical actual private DATA metadata origin and public copy')
    prepared = identities(data['actual_feature_case_ids'])
    source_exposed = identities(data['source_text_exposed_case_ids'])
    hypotheses_exposed = identities(data['actual_hypothesis_exposed_case_ids'])
    reserved = identities(data['reserved_development_case_ids'])
    need(data['reserved_role'] == 'DEV_FOREVER' and len(reserved) == 17
         and set(prepared + source_exposed + hypotheses_exposed) <= set(reserved), 'fixed all17 DEV reservation and exposure subsets')
    pilot = config['DEV_pilot']; pa = None; pilot_raw = None; score = None; gold = None
    forwarded = []; scored = []; decoded = []; dev_counts = None; quality = None
    label_uncertainty = None; quality_evidence = None; metric_gate = None
    observed_classes = None; present_classes = None; absent_classes = None
    status = eng['status']; reason = eng['reason']
    if pilot is None:
        no = checked(config['DEV_nonexecution_review'], True)
        need(no['epoch'] == 59 and no['independent_review_complete'] is True
             and no['all_owned_processes_closed'] is True
             and type(no['DEV_CLI_invocations']) is int and no['DEV_CLI_invocations'] == 0
             and type(no['Gold_scoring_calls']) is int and no['Gold_scoring_calls'] == 0
             and type(no['semantic_gold_decodes']) is int and no['semantic_gold_decodes'] == 0, 'explicit independent DEV nonexecution')
        dev_counts = counters(no['actual_execution_independently_derived'])
        need(all(value == 0 for value in dev_counts.values()), 'nonexecuted DEV actual observations are not zero')
        label_uncertainty = {'unresolved': False, 'source': 'independent_DEV_nonexecution_review'}
    else:
        need(type(pilot) is dict and set(pilot) == {'results', 'independent_acceptance', 'frozen_plan', 'sealed_forecasts', 'scoring', 'Gold_report'}, 'complete explicit DEV actual input references')
        need(pilot['frozen_plan'] == config['DEV_plan'], 'same source-frozen optional DEV plan')
        pilot_raw, pa, dev_counts = actual(pilot['results'], pilot['independent_acceptance'], 'dev',
                                         pilot['frozen_plan'], config['source_freeze'])
        need(type(pa['pilot_completed']) is bool and type(pa['scoring_completed']) is bool, 'separate actual inference/scoring completion')
        need(pilot_raw['selected_query_ids'] == pilot_raw['reserved_ids'] == reserved,
             'all17 fixed DEV cases and no replacement')
        forwarded = identities(pa['actual_forwarded_case_ids'])
        need(set(forwarded) <= set(prepared), 'forwarded cases were physically prepared')
        decoded = identities(pa['actual_label_decoded_case_ids'])
        label_uncertainty = copy.deepcopy(pa['label_decode_uncertainty'])
        need(type(pa['Gold_not_executed_verified']) is bool and type(label_uncertainty) is dict
             and type(label_uncertainty['unresolved']) is bool
             and set(decoded) <= set(reserved), 'explicit actual Gold outcome and any decode uncertainty')
        if label_uncertainty['unresolved']:
            need(pa['scoring_completed'] is False, 'uncertain Gold cannot support completed scoring')
        if pa['pilot_completed']:
            need(len(pilot_raw['rows']) == len(reserved) * len(ARMS)
                 and dev_counts['forecasts_completed'] == len(pilot_raw['rows']), 'complete actual case-major forecasts')
            need([(row['case_id'], row['arm']) for row in pilot_raw['rows']] == [(case, arm) for case in reserved for arm in ARMS], 'actual 51 forecasts have canonical case/arm order')
        if pilot['Gold_report'] is not None:
            gold = checked(pilot['Gold_report'], True)
            need(gold['epoch'] == 59 and gold['kind'] == KIND and gold['used_confirmation_identities'] == [], 'actual DEV-only Gold report')
            need(identities(gold['actual_label_decoded_case_ids']) == decoded
                 and pa['Gold_not_executed_verified'] is False, 'actual Gold report and independent decoded lower bound match')
        else:
            need((pa['Gold_not_executed_verified'] is True and decoded == [] and label_uncertainty['unresolved'] is False)
                 or (pa['Gold_not_executed_verified'] is False and label_uncertainty['unresolved'] is True),
                 'missing Gold report cannot silently become zero decoded labels')
        if pilot['scoring'] is not None:
            score = checked(pilot['scoring'], True)
            need(score['epoch'] == 59 and score['kind'] == KIND
                 and pa['actual_scoring_ref'] == pilot['scoring'], 'independent actual scorer binding')
            need(pa['sealed_forecasts_ref'] == pilot['sealed_forecasts']
                 and score['sealed_forecasts_ref'] == pilot['sealed_forecasts'], 'late scorer consumed exact independently sealed forecasts')
            canonical(pilot['sealed_forecasts'])
            scored = identities(score['actual_scored_case_ids'])
            need(set(scored) <= set(decoded) and scored == pa['actual_scored_case_ids'], 'actual scored labels were actually decoded')
            if pa['scoring_completed']:
                need(pa['pilot_completed'] and pilot['Gold_report'] is not None
                     and label_uncertainty['unresolved'] is False and set(scored) == set(reserved)
                     and set(score['arms']) == set(ARMS)
                     and same(pa['quality_metrics'], score['arms']), 'complete independently recomputed actual quality metrics')
                quality = copy.deepcopy(score['arms'])
                observed_classes = class_counts(score['actual_gold_class_counts'])
                need(sum(observed_classes.values()) == len(scored), 'actual Gold class denominator')
                for arm in ARMS:
                    metric = quality[arm]
                    need(type(metric['count']) is int and metric['count'] == len(scored)
                         and type(metric['correct']) is int and 0 <= metric['correct'] <= metric['count'], 'actual quality denominators')
                    for key in ('accuracy', 'Brier', 'NLL'):
                        need(type(metric[key]) in (int, float) and math.isfinite(metric[key]) and metric[key] >= 0, 'finite actual quality metric')
                    per_class = metric['per_class']
                    need(type(per_class) is dict and set(per_class) == set(LABELS), 'all canonical per-class metrics')
                    for label in LABELS:
                        row = per_class[label]
                        need(type(row) is dict and set(row) == {'count', 'correct'}
                             and type(row['count']) is int and row['count'] == observed_classes[label]
                             and type(row['correct']) is int and 0 <= row['correct'] <= row['count'], 'actual per-class denominators and correct counts')
                    need(sum(row['count'] for row in per_class.values()) == metric['count']
                         and sum(row['correct'] for row in per_class.values()) == metric['correct'], 'per-class totals match actual arm totals')
        else:
            need(pa['scoring_completed'] is False and pa['actual_scored_case_ids'] == [], 'no claimed score without actual scorer')
        status = pa['status']; reason = pa['reason']
    counter_violations = {'engineering': counter_policy_violations(eng_counts), 'dev': counter_policy_violations(dev_counts)}
    quality_violations = []
    if quality is not None:
        sealed = checked(pilot['sealed_forecasts'])
        present_classes = [label for label in LABELS if observed_classes[label] > 0]
        absent_classes = [label for label in LABELS if observed_classes[label] == 0]
        quality_evidence = {
            'no_worker_errors': pilot_raw['errors'] == [],
            'exact51_actual_scored_forecasts': type(score['actual_scored_forecasts']) is int and score['actual_scored_forecasts'] == len(reserved) * len(ARMS),
            'sealed_rows_equal_actual_raw_rows': same(sealed['rows'], pilot_raw['rows']),
            'NEI_rows_have_full_source_coverage': all(row['prediction'] != 'NotMentioned' or row['all_source_windows_covered'] is True for row in pilot_raw['rows']),
            'actual_class_counts_match_late_Gold_and_independent_audit': (
                same(observed_classes, class_counts(gold['actual_decoded_label_class_counts']))
                and same(observed_classes, class_counts(pa['actual_gold_class_counts']))),
            'actual_class_presence_and_scope_are_explicit': (
                score['present_gold_classes'] == present_classes
                and score['absent_gold_classes'] == absent_classes
                and score['all_gold_classes_present'] is (not absent_classes)
                and score['all_class_competence_claim'] is False),
        }
        native, source_order, full = (quality[arm] for arm in ARMS)
        checks = {
            'native_correct_at_least_source_plus_one': native['correct'] >= source_order['correct'] + 1,
            'native_Brier_not_worse_source': native['Brier'] <= source_order['Brier'] + 1e-6,
            'native_NLL_not_worse_source': native['NLL'] <= source_order['NLL'] + 1e-6,
            'native_correct_not_worse_full': native['correct'] >= full['correct'],
            'native_Brier_not_worse_full': native['Brier'] <= full['Brier'] + 1e-6,
            'native_NLL_not_worse_full': native['NLL'] <= full['NLL'] + 1e-6,
            'native_correct_each_present_Gold_class': all(native['per_class'][label]['correct'] >= 1 for label in present_classes),
        }
        reported_gate = score['mechanism_quality_gate']
        quality_evidence['scorer_gate_matches_frozen_metric_checks'] = (
            type(reported_gate['tolerance']) in (int, float) and reported_gate['tolerance'] == 1e-6
            and same(reported_gate['checks'], checks)
            and reported_gate['quality_checks_passed'] is all(checks.values()))
        metric_gate = {'tolerance': 1e-6, 'checks': checks, 'quality_checks_passed': all(checks.values()),
                       'present_gold_classes': present_classes, 'absent_gold_classes': absent_classes,
                       'all_class_competence_claim': False}
        quality_violations = [key for key, passed in quality_evidence.items() if not passed]
    original_status = status
    if counter_violations['engineering'] or counter_violations['dev'] or quality_violations:
        status = 'REJECTED_ACTUAL_COUNTER_POLICY_OR_QUALITY_EVIDENCE'
    elif metric_gate is not None and not metric_gate['quality_checks_passed']:
        status = 'REJECTED_FROZEN_DEV_MECHANISM_QUALITY_GATE'
    effective_engineering_qualified = eng['engineering_qualified'] and not counter_violations['engineering'] and raw['errors'] == []
    label_only = sorted(set(decoded) - set(scored))
    actual_dev = sorted(set(prepared + source_exposed + hypotheses_exposed + decoded))
    common = {'schema_version': 1, 'epoch': 59, 'kind': KIND, 'source_checkpoint': BASE,
              'actual_SOURCE_commit': source['commit'], 'native104_champion_preserved': True,
              'candidate_promoted': False, 'official_goal_verified': False, 'official_rank': None,
              'benchmark_generalization_claim': False, 'new_submission': False, 'used_confirmation_identities': [],
              'original_dataset_split': 'train', 'project_data_role': 'PROJECT_DEV',
              'development_identity_namespace': NAMESPACE, 'train_fitting_performed': False}
    result = {**common, 'status': status, 'independent_reason': reason, 'independent_verdict_status': original_status,
              'engineering_qualified': effective_engineering_qualified, 'independent_engineering_qualified': eng['engineering_qualified'], 'throughput_gate_passed': eng['throughput_gate_passed'],
              'engineering_result': config['engineering_result'], 'engineering_audit': config['engineering_audit'],
              'actual_engineering_execution': eng_counts, 'engineering_latency_and_resources': eng['latency_and_resources'],
              'engineering_throughput_measurements': eng['throughput_measurements'], 'engineering_native_accounting': eng['native_counter_accounting'],
              'DEV_pilot': pilot, 'DEV_nonexecution_review': config['DEV_nonexecution_review'],
              'actual_DEV_execution': dev_counts, 'DEV_latency_and_resources': pa['latency_and_resources'] if pa else None,
              'DEV_native_accounting': pa['native_counter_accounting'] if pa else None,
              'actual_counter_policy_violations': counter_violations, 'completed_quality_evidence_checks': quality_evidence, 'quality_evidence_violations': quality_violations, 'frozen_metric_gate_recomputed': metric_gate,
              'quality_metrics': quality, 'paired_comparisons': score['paired_comparisons'] if quality is not None else None,
              'actual_gold_class_counts': observed_classes, 'present_gold_classes': present_classes,
              'absent_gold_classes': absent_classes, 'all_class_competence_claim': False,
              'data_preparation_result': config['data_preparation_result'], 'data_preparation_review': config['data_preparation_review'],
              'actual_feature_case_ids': prepared, 'source_text_exposed_or_conservatively_retired_case_ids': source_exposed,
              'source_text_exposure_semantics': 'DATA_REPORT IDs may include conservative retirement after an interrupted access journal. They do not assert an unobserved decode; the exact DATA report and independent review retain that distinction.',
              'actual_hypothesis_exposed_case_ids': hypotheses_exposed, 'actual_forwarded_case_ids': forwarded,
              'actual_scored_case_ids': scored, 'actual_label_decoded_case_ids': decoded,
              'label_only_exposed_case_ids': label_only, 'label_decode_uncertainty': label_uncertainty,
              'canonical_labels': list(LABELS), 'canonical_arms': list(ARMS),
              'forward_count_semantics': 'Raw NLI_forward_calls_completed is the global observed session.run count. Sealed row actual_model_forward_calls records case-forward participations; scorer arm case_forward_participations sums those participations and is not a physical-call count.',
              'scorer_model_forward_accounting': score['model_forward_accounting'] if quality is not None else None,
              'metric_scope': 'One source-licensed ContractNLI originalTRAIN document473 with17 fixed claims, reserved as PROJECT_DEV. No train fitting or officialDEV/test/confirmation result is measured. Paired claim differences share one NDA and are not independent-document or benchmark-wide generalization evidence. Authored engineering performance is not benchmark quality. No official rank is inferred.',
              'qualification': config['qualification'], 'official_refresh': config['official_refresh'],
              'pre_science_preparation': config['pre_science_preparation'], 'source_frozen_metadata_reporter_executed': True,
              'all_epoch59_scientific_slots_closed': True}
    used = {**common, 'roles': {'train': [], 'development': actual_dev, 'calibration': [], 'fresh_confirmation': []},
            'reserved_DEVELOPMENT_identities': reserved, 'actual_prepared_feature_identities': prepared,
            'source_text_exposed_or_conservatively_retired_identities': source_exposed, 'actual_hypothesis_exposed_identities': hypotheses_exposed,
            'actual_forwarded_DEV_identities': forwarded, 'actual_scored_DEV_identities': scored,
            'actual_label_decoded_DEV_identities': decoded, 'actual_label_only_exposed_DEV_identities': label_only,
            'label_decode_uncertainty': label_uncertainty,
            'engineering_reserved_ids': list(raw['reserved_ids']),
            'engineering_identity_scope': 'Authored CC0 engineering reservation only; actual execution rows/counts remain in the source-bound raw engineering result. Never benchmark or confirmation.',
            'inherited_used_identities': old['epoch58_final']['used_identities'],
            'inherited_used_confirmation': old['epoch58_final']['used_confirmation'],
            'all_previous_roles_and_protected_confirmation_unchanged': True,
            'existing_confirmation_status': 'UNQUALIFIED_HOLD_UNUSED', 'confirmation_payloads_features_or_labels_read': False}
    priority = ('Preserve SubmittedNative104. Retain exact E59 typed-reader coverage/order outcomes, proper scores, latency, resource limits and every failure. The17 paired claims share ONE originalTRAIN NDA reserved as PROJECT_DEV; this cannot establish officialDEV/test, independent-document or benchmark-wide generalization. No train fitting occurred. All actually exposed current PROJECT_DEV identities are DEV forever. Do not repeat an unchanged rejected runtime or exposed-question grid. Any further attempt needs a new source/identity/resource freeze. Stronger licensed typed decision models with verified training/test provenance remain the quality priority; do not repeat unchanged FinNLI author-access or ModernBERT lineage probes. Existing easy48/Legal96/SciFact16/DOC17 roles,1109query-pool caveat and unqualified confirmation holds stay unchanged. No paid API/GPU or publisher-submission authorization follows. Official success requires the matching publisher result/rank and eligibility.')
    if not eng['throughput_gate_passed']:
        priority += ' E59 did not admit its frozen DEV projection gate; do not dispatch the unchanged optional DEV plan.'
    learning = {**common, 'status': status, 'independent_reason': reason, 'next_research_priority': priority,
                'engineering_audit': config['engineering_audit'], 'DEV_pilot': pilot, 'results_scope': result['metric_scope']}
    ledger = copy.deepcopy(checked(config['initial_rejection_ledger'], True))
    need(not any(row['epoch'] == 59 for row in ledger['rejections']), 'one new epoch record only')
    rr = {'path': P + 'RESULTS.json', 'bytes': len(encoded(result)), 'sha256': hashlib.sha256(encoded(result)).hexdigest()}
    ledger['rejections'].append({**common, 'status': status, 'reason': reason, 'results': rr,
                                 'engineering_qualified': effective_engineering_qualified, 'independent_engineering_qualified': eng['engineering_qualified'], 'throughput_gate_passed': eng['throughput_gate_passed'],
                                 'independent_acceptance': config['engineering_audit'], 'DEV_pilot': pilot,
                                 'actual_engineering_execution': eng_counts, 'actual_DEV_execution': dev_counts})
    outputs = [put(R / (P + 'RESULTS.json'), result), put(R / (P + 'USED_IDENTITIES.json'), used),
               put(R / (P + 'USED_CONFIRMATION.json'), used), put(R / (P + 'LEARNINGS.json'), learning),
               put(R / (P + 'REJECTION_LEDGER.json'), ledger)]
    need(outputs[0] == rr, 'actual report body identity')
    return {'outputs': outputs, 'status': status, 'results': outputs[0], 'learning': outputs[3],
            'ledger': outputs[4], 'next_research_priority': priority}

def close(config):
    review = checked(config['independent_public_review'])
    need(review['epoch'] == 59 and review['checks_passed'] is True
         and review['public_inventory_review_passed'] is True, 'independent complete public inventory review')
    files = sorted([ref(path, True) for root in ROOTS for path in (R / root).rglob('*') if path.is_file()], key=lambda row: row['path'])
    need(files == review['files'], 'reviewed exact public inventory')
    target = R / (A + 'FINAL_PUBLIC_SUMMARY_AND_INVENTORY_INDEPENDENT_REVIEW.v1.json')
    put_ref = put_bytes(target, canonical(config['independent_public_review']))
    # Preserve the exact independently approved bytes, including JSON formatting.
    need((put_ref['bytes'], put_ref['sha256']) == (config['independent_public_review']['bytes'], config['independent_public_review']['sha256']), 'public review exact canonical bytes')
    files = sorted([*files, put_ref], key=lambda row: row['path'])
    return put(R / (P + 'EPOCH_MANIFEST.json'), {'schema_version': 1, 'epoch': 59, 'kind': KIND,
               'source_checkpoint': BASE, 'files': files, 'no_project_additions_after_manifest': True,
               'used_confirmation_identities': [], 'official_goal_verified': False, 'independent_public_review': put_ref})

def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    need(len(args) == 2 and args[0] in ('summarize', 'close'), 'explicit metadata reporting mode/config')
    config = parse(body(Path(args[1]))); activate(config)
    result = summarize(config) if args[0] == 'summarize' else close(config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))

if __name__ == '__main__': main()
