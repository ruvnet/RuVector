#!/usr/bin/env python3
"""Create a typed Autogenous envelope from independently validated receipts."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
fw, gate_path, out = map(Path, sys.argv[1:])
sha = lambda data: hashlib.sha256(data).hexdigest()
gate_bytes = gate_path.read_bytes()
gate = json.loads(gate_bytes)
assert gate['decision']['promote'] is True and gate['lab_pass'] is True
assert gate['candidate_sha256'] == sha((fw/'tools/rig.py').read_bytes())
parent = '6fdb5f12ac7fe16752eeb60405e52567081706db'
assert gate['parent_sha'] == parent
experiment_path = fw/'build-finite/experiment.json'
experiment = json.loads(experiment_path.read_bytes())
for path, digest in experiment['frozen'].items():
    assert sha((fw/path).read_bytes()) == digest, path
invariants = [{'name': key, 'holds': True} for key in ['finite legacy retention equality',
    'frozen firmware arithmetic and default model', 'frozen 11 pair and 1.10 gain criteria',
    'integrated software acceptance', 'no merge deployment or physical claims']]
constitution = {'delivery': 'tested PR only', 'deployment_authorized': False,
                'experiment_sha256': sha(experiment_path.read_bytes())}
genome = {'hash': parent, 'identity': 'repo/ruvnet/RuVector/esp32-decision',
          'constitution': sha(json.dumps(constitution, sort_keys=True).encode()),
          'capability_ceiling': 'governed', 'hard_invariants': invariants,
          'lineage': [parent]}
record = {
    'schema': 1, 'campaign': 'esp32-overnight-20260926',
    'run': 'esp32-slot23-20260926T0301Z', 'parent_sha': parent, 'seed': 719,
    'genome': genome,
    'mutation': {'id': 'retention-finite-measurement-guard', 'parent_genome_hash': parent,
                 'scope': 'application_code', 'requested_authority': 'governed',
                 'applicability': {'workloads': ['ESP32 timing and energy offline evidence validation'],
                                   'environments': ['Python host benchmark lab'], 'jurisdictions': []},
                 'preserved_invariants': invariants, 'rollback_target': parent,
                 'expires_at': int(datetime(2026,9,26,3,47,32,tzinfo=timezone.utc).timestamp()),
                 'signature': None},
    'hypothesis': 'Reject malformed nonfinite boolean and overflowing benchmark inputs without changing finite legacy retention or fixed acceptance thresholds.',
    'scope': ['examples/esp32-decision/tools/rig.py'],
    'supporting_scope': ['examples/esp32-decision/tests/test_finite_benchmark.py',
                        'examples/esp32-decision/tests/evidence/',
                        'npm/packages/typesafe/docs/adr/ADR-007-esp32-measured-sensor-decisions.md'],
    'declared_authority': 'Governed PR proposal only. Human authorization for merging or deployment absent.',
    'input_hashes': {'parent_sha': parent, 'candidate_rig_sha256': gate['candidate_sha256'],
                     'evaluator_sha256': gate['frozen_evaluator_sha256'],
                     'experiment_sha256': gate['experiment_sha256'],
                     'frozen_invariants': experiment['frozen']},
    'artifacts': {'metaharness_gate_sha256': sha(gate_bytes),
                  'evaluator_receipt_sha256': gate['independent_receipt_sha256'],
                  'lab_sha256': gate['lab_sha256']},
    'tool_versions': {'autogenous': '905aa6cbe213392f8b3cab5d4f17bc3a48e0a509',
                      'metaharness': 'd5833dc6512ac1adeeef91a331c29055cd8a4dbb',
                      'node': 'v24.19.0', 'cargo': '1.98.1'},
    'failure_boundary': 'Any malformed fixture accepted, finite output regression, frozen invariant change, lab failure or expired authority vetoes PR retention.',
    'rollback': 'Revert only this run candidate patch, restoring tools/rig.py from parent; preserve independent evidence.',
    'fitness_vector': None, 'deployment_promoted': False,
    'limitations': ['No measured production safety or p99 values: FitnessVector hard gates not evaluated.',
                    'Genome identity uses the Git source snapshot SHA; no signed production genome artifact claimed.',
                    'ApplicationCode.auto_promotable is false. Structural admission is distinct from promotion.',
                    'Signature intentionally absent for offline structural check; no cryptographic admission claim.']
}
out.write_text(json.dumps(record, indent=2)+'\n')
