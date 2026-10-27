"""Assemble only safe E57 public facts and scoped publisher metadata; no network or messages."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
W = Path('/dev/shm/ruvector-e57-data')
O = W / 'official-refresh'
def load(p): return json.loads(p.read_bytes())
def ref(p):
    b = p.read_bytes(); return {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
def put(p, v):
    b = (json.dumps(v, indent=2, sort_keys=True, allow_nan=False) + '\n').encode(); assert b'@' not in b
    with p.open('xb') as f: f.write(b); f.flush(); os.fsync(f.fileno())
    return ref(p)
j, claim, mail, prior = [load(O / n) for n in ['JEV_STREAM_RESULT.actual.v1.json', 'claim-public-facts.json', 'MAIL_STATUS.safe.json', 'donors/benchmark-refresh.safe.json']]
assert j['status'] == 'FRESH_OFFICIAL_JEV_STREAM_COMPLETE' and j['epoch'] == claim['epoch'] == mail['epoch'] == 57
assert mail['configured_account_profile_reads'] == mail['configured_profile_identities_verified'] == mail['exact_reference_searches'] == 2
assert mail['messages_read'] == mail['snippets_inspected'] == mail['body_thread_attachment_reads'] == mail['outgoing_messages_or_submissions'] == 0
for name in ('JEV_ROOT_EXECUTION.actual.v1.json', 'CLAIM_ROOT_EXECUTION.actual.v1.json'):
    outer = load(O / name); assert outer['actual_invocations'] == 1 and outer['foreground_command_closed_observation'] is True and outer['tool_return']['exit_code'] == 0
jev = {k: j[k] for k in ['board_version', 'board_updated', 'method_revision', 'method_header', 'ranked_open_weights_systems', 'verified_capability_rows_count',
    'first_official_capability_rows', 'all_capability_rows_sha256', 'RuVector_page_text_matches', 'submission_reference_page_matches',
    'official_RuVector_rank', 'official_RuVector_score', 'official_RuVector_eligibility', 'current_questions_scorer_license_access', 'historical_access_probes_repeated']}
jev['open_weights_Capability_leader'] = j['live_open_weights_Capability_leader']
jev['leader_equals_last_verified_epoch56'] = jev['open_weights_Capability_leader'] == prior['JevBench']['open_weights_Capability_leader']
jev['source'] = {'url': j['url'], 'final_url': j['final_url'], 'status': j['HTTP_status'], 'bytes': j['bytes_streamed'], 'sha256': j['body_sha256'],
    'Content_Length': j['Content_Length'], 'retrieved_at': j['retrieved_at'], 'read_cap_bytes': j['maximum_bytes'], 'deadline_seconds': j['deadline_seconds'],
    'wall_seconds': j['wall_seconds'], 'TLS_verification_enabled': True, 'raw_body_persisted': False}
jev['stream_result_ref'] = ref(O / 'JEV_STREAM_RESULT.actual.v1.json')
data = {'schema_version': 1, 'epoch': 57, 'source_checkpoint': '5e426edd8ada8c0fdcec38ec8807054affc89e3c',
    'kind': 'FRESH_OFFICIAL_JEV_CLAIM_AND_SCOPED_PUBLISHER_METADATA_REFRESH', 'created_at': datetime.now(timezone.utc).isoformat(),
    'JevBench': jev, 'ClaimBench': claim['ClaimBench'], 'publisher_status': mail, 'direct_public_GETs': 3,
    'actual_request_breakdown': {'Jev_bounded_stream_GETs': 1, 'Claim_page_GETs': 1, 'Claim_linked_aggregate_GETs': 1, 'redirect_retry_alternate_endpoint_GETs': 0},
    'publisher_metadata_reads': {'configured_account_profiles': 2, 'exact_reference_searches': 2, 'body_thread_attachment_reads': 0, 'snippets_inspected': 0},
    'publisher_scope_expanded_before_reads': True,
    'publisher_scope_expansion_reason': 'The original private account selector was intentionally absent from public checkpoint artifacts. Root authorized both configured profiles and one exact-reference metadata query each before any calls. No personal-status inference from domain, no alternate-reference or unbounded account search.',
    'TLS_verification_enabled': True, 'inherited_proxy_preserved': True, 'raw_public_bodies_persisted': False,
    'private_mail_selectors_identifiers_addresses_headers_body_or_snippet_files_persisted': False, 'source_receipts': claim['source_receipts'],
    'official_goal_verified': False, 'official_rank': None, 'meaningful_verified_official_result_or_leader_change': not jev['leader_equals_last_verified_epoch56'],
    'new_verified_full_dataset_or_scorer_access': False, 'model_native_DATA_Gold_or_paid_GPU_calls': 0,
    'outgoing_messages_submissions_author_contacts': 0, 'project_or_Git_mutations': 0, 'champion_promoted': False,
    'Claim_pilot_authorized_or_executed': False, 'known_unchanged_access_limits': prior['known_unchanged_access_limits'],
    'unchanged_qualification_not_retested': {'FinNLI': 'Official paper data/license requires author-mediated access; no contact or repeated access probes.',
        'ModernBERT': 'Exact training/test nonuse unverified; no repeated model-card probes or weights.',
        'FinQA': 'Program task/source-rights/model-nonuse holds remain; exposed README case/page/filing identities remain DEV forever.'},
    'actual_public_command_receipts': [ref(O / n) for n in ['JEV_ROOT_EXECUTION.actual.v1.json', 'CLAIM_ROOT_EXECUTION.actual.v1.json']],
    'Claim_deadline_scope': 'urllib45s is a socket-I/O timeout. The once-executed foreground command used external100s TERM+2s KILL supervision and exited0; no retry or background process.'}
data['known_unchanged_access_limits'][1] = 'Two profile-verified configured accounts received one exact7FEA2738 metadata query each. Receipt-class and date metadata are scoped observations only; bodies and message identity were not checked. This cannot establish global result absence, an official score/rank/eligibility or current queue progress. Historical115 is an evaluation queue position, never rank.'
print(json.dumps(put(O / 'benchmark-refresh.safe.v2.json', data)))
files = [O / n for n in ['refresh_jev_stream.py', 'refresh_claim_only.py', 'JEV_STREAM_PLAN.v1.json', 'JEV_STREAM_RESULT.actual.v1.json',
    'claim-source-receipt-1.json', 'claim-source-receipt-2.json', 'claim-public-facts.json', 'MAIL_STATUS.safe.json', 'benchmark-refresh.safe.v2.json',
    'JEV_ROOT_EXECUTION.actual.v1.json', 'CLAIM_ROOT_EXECUTION.actual.v1.json']]
files += [W / 'PRIOR_CHECKPOINT_AND_REJECTIONS_REVIEW.v1.json', W / 'readback/ACTUAL_REMOTE_CHECKPOINT_AND_DONORS_INDEPENDENT.v1.json', Path(__file__)]
packet = {'schema_version': 1, 'epoch': 57, 'kind': 'PUBLIC_SAFE_OFFICIAL_REFRESH_AND_ACTUAL_PRIOR_REVIEW_PACKET', 'source_checkpoint': data['source_checkpoint'],
    'files': [ref(p) for p in files], 'raw_bodies_private_mail_fields_or_data_payloads_included': False, 'Git_mutations': 0, 'science_calls': 0,
    'new_qualification_probes': 0, 'used_confirmation_identities': []}
print(json.dumps(put(W / 'PUBLIC_METADATA_PACKET.v2.json', packet)))
