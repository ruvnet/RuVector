#!/usr/bin/env bash
# Requires the built native binary, Python build tools, npm and network access
# for the pinned TypeScript compiler. No package is published.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
validation_dir="$(mktemp -d)"
trap 'rm -rf "$validation_dir"' EXIT
export RUVECTOR_LANGUAGE_BIN="${RUVECTOR_LANGUAGE_BIN:-$project_root/crates/ruvector-language-native/target/debug/rvg}"
export RUVECTOR_LANGUAGE_FIXTURES="$project_root/crates/ruvector-language/examples"

npm pack "$project_root/npm/packages/graph-language" --pack-destination "$validation_dir" --ignore-scripts
npm install --prefix "$validation_dir" --ignore-scripts --no-audit --no-fund \
  "$validation_dir/ruvector-graph-language-0.1.0.tgz"
cat > "$validation_dir/smoke.mts" <<'TYPESCRIPT'
import { RuGraph, type RunRequest, type Receipt } from '@ruvector/graph-language';
const client = new RuGraph();
const source = 'program p { effect graph.read; effect decision.emit; budget 100000; let n = select ["api"]; let e = observe n "Failing" within 30; let d = assess e; emit d; }';
const request: RunRequest = {
  program: await client.compile(source),
  snapshot: {version:1,revision:'package_smoke',nodes:[{id:'api',label:'Service'}],edges:[],observations:[]},
  context: {now:1000,capabilities:['graph.read','decision.emit'],max_steps:100000,max_results:100},
};
const receipt: Receipt = await client.run(request);
if (receipt.output.kind !== 'decisions' || receipt.output.items[0]?.status !== 'unknown') throw new Error('Incorrect packaged SDK output');
const replay = await client.replay(request, receipt);
if (replay.receipt_hash !== receipt.receipt_hash) throw new Error('Packaged SDK replay mismatch');
request.snapshot.observations = [{id:'test',subject:'api',predicate:'Failing',value:true,source:'probe',observed_at:999,expires_at:1100}];
const audit = await client.audit(request, {interventions:[{kind:'source',id:'probe'}],max_execution_work:200000});
if (audit.trials[0]?.changes[0]?.after !== 'unknown') throw new Error('Packaged audit mismatch');
console.log('Packed npm client and TypeScript declarations: PASS');
TYPESCRIPT
npm exec --yes --package typescript@5.9.3 -- tsc \
  --strict --target ES2022 --module NodeNext --moduleResolution NodeNext \
  --outDir "$validation_dir/compiled" "$validation_dir/smoke.mts"
node "$validation_dir/compiled/smoke.mjs"

python3 -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir "$validation_dir" "$project_root/python/ruvector-language"
python3 -m pip install --no-deps --no-index --target "$validation_dir/python" \
  "$validation_dir/ruvector_language-0.1.0-py3-none-any.whl"
PYTHONPATH="$validation_dir/python" python3 - <<'PYTHON'
import json
import os
from pathlib import Path
from ruvector_language import RuGraph
fixtures = Path(os.environ["RUVECTOR_LANGUAGE_FIXTURES"])
client = RuGraph()
request = {
    "program": client.compile((fixtures / "incident.rvg").read_text()),
    "snapshot": json.loads((fixtures / "snapshot.json").read_text()),
    "context": json.loads((fixtures / "context.json").read_text()),
}
receipt = client.run(request)
assert any(item["status"] == "supported" for item in receipt["output"]["items"])
assert client.replay(request, receipt) == receipt
report = client.audit(request, {"interventions": [{"kind": "source", "id": "telemetry-a"}], "max_execution_work": 200000})
assert len(report["trials"][0]["changes"]) == 2
print("Installed Python wheel execution and replay: PASS")
PYTHON
