"""Root-only E59 source metadata builder. No scientific or Git operation.

Review the concrete input and source before invocation. Existing public bodies
are inventoried exactly; runtime bodies are streamed only to verify hashes and
are never copied or decoded. Master and source manifest precede the acyclic SF.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys

W = Path('/dev/shm/ruvector-sota-2026-10-22/epoch59-9b677a916983')
R = W / 'project'
P = 'experiments/sota-2026-10-22/epoch59/'
T = 'experiments/sota-2026-10-22/runtime/epoch59/'
ROOTS = (P, 'experiments/sota-2026-10-22/audit/epoch59/', T)
BASE = '9b677a91698352bd563f189fee861d6ff2a1ad3a'
KIND = 'REAL_NATIVE_TYPED_READER_BOUNDED_RESEARCH'
BUILDER_PUBLIC_PATH = P + 'preparation/build_master_and_freeze.py'

def need(value, message):
    if not value: raise ValueError(message)

def unique(rows):
    out = {}
    for key, value in rows:
        need(key not in out, 'duplicate metadata JSON key'); out[key] = value
    return out

def parse(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite metadata')))

def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()

def safe(path):
    p = Path(path)
    need(p.is_absolute() and p.resolve() == p and all(not x.is_symlink() for x in (p, *p.parents)), 'canonical absolute nonsymlink source path')
    return p

def stream_ref(path):
    p = safe(path); before = p.stat()
    need(stat.S_ISREG(before.st_mode), 'regular source file')
    digest = hashlib.sha256(); count = 0
    with p.open('rb') as source:
        for chunk in iter(lambda: source.read(1048576), b''):
            count += len(chunk); digest.update(chunk)
    after = p.stat(); sig = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(sig(before) == sig(after) and count == after.st_size, 'stable source hash')
    return {'path': str(p), 'bytes': count, 'sha256': digest.hexdigest()}

def typed(row):
    need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}
         and type(row['path']) is str and type(row['bytes']) is int and row['bytes'] >= 0
         and type(row['sha256']) is str and len(row['sha256']) == 64
         and all(c in '0123456789abcdef' for c in row['sha256']), 'exact typed3 source reference')
    return row

def absolute_checked(row):
    typed(row); need(stream_ref(row['path']) == row, 'exact actual absolute source body')
    return Path(row['path'])

def bounded_json_ref(row):
    p = absolute_checked(row); need(row['bytes'] <= 8388608, 'bounded source metadata JSON')
    return parse(p.read_bytes())

def filename_identity_qualification(config, g):
    need({'actual_filename_qualification', 'independent_filename_qualification'} <= set(config),
         'required filename qualification references')
    report_ref = config['actual_filename_qualification']; review_ref = config['independent_filename_qualification']
    for row in (report_ref, review_ref):
        typed(row); g.relative(row['path']); need(g.new_path(row['path']), 'current public filename qualification metadata')
    report_raw = g.checked(report_ref, True); report = g.parse(report_raw)
    review = g.parse(g.checked(review_ref, True))
    need(review['epoch'] == 59 and review['checks_passed'] is True
         and review['filename_alias_qualification_passed'] is True, 'independent actual filename qualification passed')
    origin_ref = review['actual_qualification_report_ref']; typed(origin_ref)
    origin = absolute_checked(origin_ref)
    need((origin_ref['bytes'], origin_ref['sha256']) == (report_ref['bytes'], report_ref['sha256'])
         and origin.read_bytes() == report_raw, 'public qualification exactly matches Q-bound actual origin')
    need(report['epoch'] == 59 and report['phase'] == 'FILENAME_IDENTITY_ONLY'
         and report['source_checkpoint'] == BASE and report['checks_passed'] is True
         and report['status'] == 'READY_TRAIN473_FILENAME_ALIAS_QUALIFIED_NO_FEATURES'
         and report['errors'] == [] and type(report['selected_document_id']) is int
         and report['selected_document_id'] == 473 and report['source_split'] == 'train'
         and report['project_role'] == 'PROJECT_DEV' and report['no_replacement'] is True,
         'fixed passed originalTRAIN473 PROJECT_DEV qualification without replacement')
    need(report['selected_normalized_document_sha256_from_frozen_metadata'] ==
         'd9c341c8757f6616004a579576c4ae29e64f7346a0bf4e98c182ad431730fd37'
         and type(report['selected_document_filename_key_sha256']) is str
         and re.fullmatch('[0-9a-f]{64}', report['selected_document_filename_key_sha256']),
         'fixed normalized identity and actual filename key hash')
    need(type(report['filename_metadata_decodes']) is int and report['filename_metadata_decodes'] == 423
         and type(report['document_IDs_seen']) is int and report['document_IDs_seen'] == 423
         and report['source_archive_sha256'] == 'e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b'
         and report['source_member'] == 'contract-nli/train.json', 'exact filename-only source member and decode counts')
    for key in ('source_plan_ref', 'source_freeze_ref'): absolute_checked(report[key])
    aliases = report['all_source_TRAIN_filename_alias_document_ids']
    need(type(aliases) is list and aliases == sorted(set(aliases))
         and all(type(doc) is int for doc in aliases) and 473 in aliases
         and report['all_filename_alias_document_ids_retired_as_PROJECT_DEV_forever'] == aliases
         and report['blocked_or_reserved_TRAIN_filename_alias_document_ids'] == []
         and report['prior_DEV_TEST_Legal96_filename_collision'] is False,
         'selected filename group collision-free and all aliases retired as project DEV')
    for key in ('source_text_decodes', 'hypothesis_value_decodes', 'semantic_Gold_decodes',
                'source_text_and_hypothesis_values_decoded', 'Gold_choice_values_decoded'):
        need(type(report[key]) is int and report[key] == 0, 'metadata-only filename qualification: ' + key)
    need(report['train_ids'] == report['calibration_ids'] == report['fresh_confirmation_ids'] == []
         and report['confirmation_qualification'] == 'UNQUALIFIED_HOLD', 'no fitting or confirmation qualification from filenames')
    return ({'filename_qualification': report_ref, 'independent_filename_qualification': review_ref},
            {'origin_ref': origin_ref, 'filename_key_sha256': report['selected_document_filename_key_sha256']})

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True); parser.add_argument('--input-sha', required=True)
    parser.add_argument('--approval', type=Path, required=True); parser.add_argument('--approval-sha', required=True)
    args = parser.parse_args(argv)
    need(not sys.flags.optimize and os.sched_getaffinity(0) == {0, 1}, 'nonoptimized CPU0,1 metadata builder')
    input_ref = stream_ref(args.input); approval_ref = stream_ref(args.approval)
    need(input_ref['sha256'] == args.input_sha and approval_ref['sha256'] == args.approval_sha, 'exact reviewed input/approval bytes')
    config = bounded_json_ref(input_ref); approval = bounded_json_ref(approval_ref)
    own = stream_ref(Path(__file__).resolve())
    need(approval['epoch'] == 59 and approval['checks_passed'] is True
         and approval['source_metadata_preparation_allowed'] is True
         and approval['phase_input_sha256'] == args.input_sha
         and approval['builder_source_sha256'] == own['sha256'], 'independent exact source metadata preparation approval')
    need(config['epoch'] == 59 and config['kind'] == KIND and config['source_checkpoint'] == BASE, 'current source metadata context')
    guard_ref = config['publication_guard']; typed(guard_ref)
    need(guard_ref['path'] == str(W / 'publication-helpers-g27/guard.py'), 'fixed operational publication guard')
    guard_path = absolute_checked(guard_ref)
    spec = importlib.util.spec_from_file_location('e59_source_metadata_exact_guard', guard_path)
    need(spec is not None and spec.loader is not None, 'exact guard loader')
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g); g.owner_fence()
    need(config['owner'] == g.OWNER and g.BASE == BASE and g.KIND == KIND and g.R == R, 'exact live owner/base/source context')
    qualification, qualification_actual = filename_identity_qualification(config, g)
    generated = {P + 'plan.json', P + 'SOURCE_MANIFEST.json', P + 'SOURCE_FREEZE.json'}
    need(all(not (R / name).exists() for name in generated), 'new master/manifest/freeze only; never overwrite')
    inventory = {}
    for row in config['public_artifacts']:
        typed(row); g.relative(row['path']); need(g.new_path(row['path']) and row['path'] not in inventory, 'unique current public inventory')
        body = g.checked(row, True); g.public_text(row['path'], body); inventory[row['path']] = row
    actual = {}
    for root in ROOTS:
        folder = R / root
        if not folder.exists(): continue
        for path in folder.rglob('*'):
            need(not path.is_symlink(), 'no public source symlink')
            if path.is_file():
                row = g.ref(path); row['path'] = path.relative_to(R).as_posix(); actual[row['path']] = row
    need(inventory == actual, 'exact existing public inventory; no omitted or extra body')
    review = bounded_json_ref(config['independent_public_source_review'])
    need(review['epoch'] == 59 and review['checks_passed'] is True
         and review['public_inventory_review_passed'] is True
         and review['files'] == sorted(inventory.values(), key=lambda x: x['path']), 'independent exact public source/privacy inventory review')
    need(inventory[BUILDER_PUBLIC_PATH]['bytes'] == own['bytes']
         and inventory[BUILDER_PUBLIC_PATH]['sha256'] == own['sha256'], 'executed builder source is public and frozen')
    need(inventory[P + 'publication/guard.py']['bytes'] == guard_ref['bytes']
         and inventory[P + 'publication/guard.py']['sha256'] == guard_ref['sha256'], 'operational guard matches public source')
    runtime = config['runtime_sources']
    need(type(runtime) is list and len({typed(row)['path'] for row in runtime}) == len(runtime), 'unique exact runtime source refs')
    for row in runtime:
        g.owner_fence(); absolute_checked(row)
    inherited = []
    seen = set(inventory)
    for row in config['inherited_public_sources']:
        typed(row); g.relative(row['path']); need(not g.new_path(row['path']) and row['path'] not in seen, 'unique inherited source outside current namespace')
        g.checked(row, True); inherited.append(row); seen.add(row['path'])
    fields = copy.deepcopy(config['master_fields'])
    required = {'schema_version': 1, 'epoch': 59, 'kind': KIND, 'source_checkpoint': BASE,
                'source_freeze_path': P + 'SOURCE_FREEZE.json', 'train_ids': [], 'calibration_ids': [],
                'fresh_confirmation_ids': [], 'candidate_promotion_allowed': False,
                'schedule_is_not_actual_execution_evidence': True}
    for key, value in required.items():
        need(key not in fields or g.same(fields[key], value), 'fixed master context/policy: ' + key)
        fields[key] = value
    need(not {'source_freeze_sha256', 'source_freeze_ref', 'source_freeze'} & set(fields), 'acyclic master: current freeze path only')
    limits = fields['limits']
    for key, ceiling in [('native_corpus_batches', 2), ('CPU_threads', 2), ('inclusive_CLI_seconds_per_batch', 600)]:
        need(type(limits[key]) is int and 1 <= limits[key] <= ceiling, 'bounded source limit: ' + key)
    for key in ('paid_API_calls', 'GPU_calls'):
        need(type(limits[key]) is int and limits[key] == 0, 'zero paid/GPU budget')
    need(type(fields['frozen_window_count']) is int and 1 <= fields['frozen_window_count'] <= 24, 'actual blind frozen window bound')
    ids = fields['development_ids']
    need(type(ids) is list and len(ids) == len(set(ids)) == 17
         and all(type(case) is str for case in ids), 'all17 fixed DEV identities')
    namespace = ids[0].rsplit(':', 1)[0]
    need(namespace == 'contractnli:eced6528:train:473'
         and all(case.rsplit(':', 1)[0] == namespace for case in ids), 'one exact frozen DEV document namespace')
    need('development_namespace' not in fields or fields['development_namespace'] == namespace,
         'configured namespace agrees with exact frozen identities')
    fields['development_namespace'] = namespace
    for key, value in {'original_dataset_split': 'train', 'project_data_role': 'PROJECT_DEV',
                       'train_fitting_performed': False, 'official_DEV_test_or_confirmation_claim': False}.items():
        need(key not in fields or g.same(fields[key], value), 'originalTRAIN PROJECT_DEV role: ' + key)
        fields[key] = value
    data = copy.deepcopy(config['actual_preparation'])
    for key, row in qualification.items():
        need(key not in data or g.same(data[key], row), 'qualification provenance must agree with actual preparation')
        data[key] = row
    need(type(data) is dict and {'DATA_report', 'catalog_report', 'local_plan', 'local_source_freeze'} <= set(data), 'actual separate local preparation provenance')
    # Public preparation metadata only; do not open a bank, vault, model or catalog as JSON here.
    for key in ('DATA_report', 'catalog_report', 'local_plan', 'local_source_freeze'):
        row = data[key]; typed(row); g.checked(row, True)
    data_actual = g.parse(g.checked(data['DATA_report'], True))
    catalog_actual = g.parse(g.checked(data['catalog_report'], True))
    need(data_actual['epoch'] == 59 and data_actual['kind'] == KIND and data_actual['source_checkpoint'] == BASE
         and data_actual['identity_namespace'] == namespace
         and data_actual['actual_feature_case_ids'] == data_actual['reserved_development_case_ids'] == ids
         and type(data_actual['semantic_gold_decodes']) is int and data_actual['semantic_gold_decodes'] == 0
         and data_actual['used_confirmation_identities'] == [], 'master identities derive from actual blind DATA')
    need(data_actual['status'] == 'READY_LABEL_BLIND_ONE_DOC473_ALL17'
         and g.same(data_actual['filename_qualification_ref'], qualification_actual['origin_ref'])
         and data_actual['selected_document_filename_key_sha256'] == qualification_actual['filename_key_sha256']
         and data_actual['selected_normalized_document_sha256'] ==
         'd9c341c8757f6616004a579576c4ae29e64f7346a0bf4e98c182ad431730fd37',
         'actual DATA separately verified selected text/filename and bound passed qualification')
    need(catalog_actual['epoch'] == 59 and catalog_actual['kind'] == KIND
         and catalog_actual['status'] == 'BLIND_CATALOG_READY' and catalog_actual['selected_query_ids'] == ids
         and catalog_actual['all17_claims_capacity_checked'] is True
         and type(catalog_actual['frozen_window_count']) is int
         and catalog_actual['frozen_window_count'] == fields['frozen_window_count'], 'master W derives from actual all17-capacity catalog')
    for key in ('NLI_forward_calls', 'native_corpus_batches', 'Gold_labels_decoded', 'paid_API_calls', 'GPU_calls'):
        need(type(catalog_actual[key]) is int and catalog_actual[key] == 0, 'blind catalog performed no scientific/Gold/paid/GPU call')
    plan_raw = g.encoded(fields); g.public_text(P + 'plan.json', plan_raw)
    plan_ref = {'path': P + 'plan.json', 'bytes': len(plan_raw), 'sha256': g.sha(plan_raw)}
    manifest = {'schema_version': 1, 'epoch': 59, 'kind': KIND, 'source_checkpoint': BASE,
                'master_plan': plan_ref, 'source_freeze_path': P + 'SOURCE_FREEZE.json',
                'public_sources_except_manifest_and_freeze_self': sorted([*inventory.values(), plan_ref], key=lambda x: x['path']),
                'inherited_public_sources': inherited, 'runtime_sources': runtime,
                'actual_separate_local_preparation': data, 'source_qualification': config['source_qualification'],
                'no_model_native_Gold_or_Git_call_by_builder': True,
                'runtime_bodies_copied_or_published_by_builder': False}
    manifest_raw = g.encoded(manifest); g.public_text(P + 'SOURCE_MANIFEST.json', manifest_raw)
    manifest_ref = {'path': P + 'SOURCE_MANIFEST.json', 'bytes': len(manifest_raw), 'sha256': g.sha(manifest_raw)}
    rows = sorted([*inventory.values(), *inherited, plan_ref, manifest_ref], key=lambda x: x['path'])
    freeze = {'schema_version': 1, 'epoch': 59, 'kind': KIND, 'source_checkpoint': BASE,
              'files': rows, 'runtime_sources': runtime, 'master_plan': plan_ref,
              'source_manifest': manifest_ref, 'owner': g.OWNER,
              'train_ids': [], 'calibration_ids': [], 'fresh_confirmation_ids': [],
              'development_ids': ids, 'development_namespace': namespace, 'frozen_window_count': fields['frozen_window_count'],
              'actual_filename_identity_qualification': qualification,
              'candidate_promotion_allowed': False, 'schedule_is_not_actual_execution_evidence': True}
    freeze_raw = g.encoded(freeze); g.public_text(P + 'SOURCE_FREEZE.json', freeze_raw)
    for name, raw in [('plan.json', plan_raw), ('SOURCE_MANIFEST.json', manifest_raw), ('SOURCE_FREEZE.json', freeze_raw)]:
        g.owner_fence(); g.write(R / (P + name), raw)
    freeze_ref = {'path': P + 'SOURCE_FREEZE.json', 'bytes': len(freeze_raw), 'sha256': g.sha(freeze_raw)}
    g.owner_fence()
    print(json.dumps({'epoch': 59, 'kind': KIND, 'master_plan': plan_ref, 'source_manifest': manifest_ref,
                      'source_freeze': freeze_ref, 'public_existing_files': len(inventory),
                      'inherited_source_files': len(inherited), 'runtime_source_refs_verified': len(runtime),
                      'actual_model_native_Gold_Git_calls': 0,
                      'scientific_authorization_granted': False}, sort_keys=True))

if __name__ == '__main__': main()
