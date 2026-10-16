# Explicit feature integrations

The main `ruvector` SDK exposes three additional APIs. They load their underlying
packages only when called. Importing the SDK does not load KGE, RVForge or RuView,
scan devices, start a server, install software, or download a model.

| Feature | Explicit installation | API |
|---|---|---|
| Trained graph reasoning | `npm install @ruvector/kge@^0.2.0` | `createGraphReasoner`, `loadGraphReasoner` |
| RVF validation and release staging | `npm install --save-dev @ruvector/rvforge@0.2.0` | `validateRvfArtifact`, `stageRvfArtifact` |
| Device sensing/operator SDK | `npm install @ruvnet/ruview@^0.9.1` | `createRuViewIntegration` |

These are optional **peers**, not `optionalDependencies`. npm normally installs
optional dependencies; it does not automatically install an optional peer.
RVForge is a development dependency of this repository for its build/test tools.
It is not installed as a runtime dependency in applications consuming RuVector.
If your application validates artifacts at runtime, explicitly install RVForge
as a production dependency instead.

Existing RVF, SONA and MetaHarness dependencies retain their existing installation
behavior. This change does not claim the entire historical dependency tree is
now opt-in. Do not use `--omit=optional` as a blanket lightweight-install fix:
it can also remove the platform binaries required by native packages.

## Graph reasoning

```js
const { createGraphReasoner, loadGraphReasoner } = require('ruvector');

const graph = await createGraphReasoner({ scorer: 'hole', dims: 32, seed: 42 });
graph.addTriples([
  { s: 'Ada', r: 'bornIn', o: 'London' },
  { s: 'Grace', r: 'bornIn', o: 'New York' },
  { s: 'London', r: 'locatedIn', o: 'England' },
  { s: 'New York', r: 'locatedIn', o: 'USA' },
]);
await graph.train({ epochs: 20, lr: 0.05 });
const candidates = graph.predict({ s: 'Ada', r: 'bornIn', k: 2 });
const restored = await loadGraphReasoner(graph.save());
console.log(restored.predict({ s: 'Ada', r: 'bornIn', k: 2 }));
```

Predictions, relation similarity and composition reject untrained models. Adding
facts or starting retraining invalidates the old training receipt. RotatE is
required for composition; HolE's upstream unsupported-operation error is retained.

The saved adapter envelope binds the completed training report to the actual
model tables. Raw upstream KGE saves are rejected because they do not retain a
training marker. Train through this adapter and save through it for reloads.
Completion is **not** evidence of prediction quality: evaluate held-out links
before making quality claims. A checksum demonstrates consistency, not authenticity.
For files crossing a trust boundary, pass a trusted key of at least 16 bytes to
both `graph.save({ key })` and `loadGraphReasoner(saved, { key })`. Signed saves
reject absent or wrong keys and changes to either the model or training metadata.

## RVF artifact checks and staging

```js
const { validateRvfArtifact, stageRvfArtifact } = require('ruvector');

// Obtain this digest from a trusted build receipt, independently of the file.
const expectedSha256 = trustedReceipt.sha256;
const validation = await validateRvfArtifact('model.rvf', { expectedSha256 });
const result = await stageRvfArtifact({
  rvfPath: 'model.rvf', expectedSha256, outDir: './new-release-stage',
  app: { name: 'Graph Demo', version: '1.0.0', publisher: 'Example', identifier: 'org.example.graphdemo' },
  targets: ['linux-x64', 'windows-x64'],
  runtime: { profile: 'native', rvmVersion: '1.0.0', rvmCommit: 'pinned-runtime-commit' },
});
console.log(result.status, result.packaged);
```

Use actual pinned runtime identity values for your release. RVForge 0.2.0 creates
staging directories, identical embedded RVFs, manifests and provenance, with
reader placeholders. It reports `status: 'staged', packaged: false`; these are
**not finished installers**. No publishing happens. Existing output directories
are not overwritten. The adapter snapshots and verifies the input before staging.

Validation uses RVForge's real deep parser and a mandatory trusted whole-file
SHA-256. The digest catches payload changes that structural checks alone miss.
`signaturePresent` reports presence, not signature authenticity. Do not describe
these checks as verification of a signer. Unsigned executable segments need an
explicit development-only `allowUnsigned: true`.

## RuView in the core SDK

`@ruvnet/ruview` is a WiFi/radar/LiDAR device-sensing operator SDK, not a 3D viewer.
This adapter explicitly loads its ESM `/sdk` entry from RuVector's CommonJS SDK.

```js
const { createRuViewIntegration } = require('ruvector');
const sensing = await createRuViewIntegration({ strict: false });
console.log(sensing.tools().map(tool => tool.name)); // Registry only; no scan.
const report = await sensing.training.gate({
  model_score: 0.8, baseline_score: 0.5, split: 'grouped-subject',
  train_subjects: ['training-subject'], test_subjects: ['held-out-subject'],
  n_test: 200, data: 'synthetic',
});
console.log(report.evidence); // SYNTHETIC, not proof of real device accuracy.
```

`training.gate()` checks a submitted report. It does not perform training or
independently verify the measurements. `call()` exposes the real RuView registry;
`devices` exposes its typed capture methods, and `doctor()` exposes diagnostics.
Actual device access requires the relevant hardware, permissions and explicit
calls. Upstream confirmation requirements remain intact. `{ strict: true }`
preserves RuView's thrown `RuViewError` for unsuccessful non-dry-run calls.

## Errors and validation

Missing features report `ERR_INTEGRATION_MISSING` with an installation command.
Incompatible entry points report `ERR_INTEGRATION_API`; a broken installed package
reports `ERR_INTEGRATION_LOAD` with the original cause. Nothing silently falls
back to mock reasoning or synthetic sensing. Public declaration files do not
require an optional peer merely to compile an ordinary core application.

From this package directory:

```sh
npm install --workspaces=false --ignore-scripts --no-package-lock --no-save \
  @ruvector/kge@0.2.0 @ruvnet/ruview@0.9.1
npm run build
npm run test:features
KGE_BACKEND=wasm node --test test/kge-integration.js
npm run test:feature-pack
```

The tests use published KGE, RVForge and RuView packages. Hardware capture and
actual model quality are not covered. CI runs the same feature contracts on
Node 20 and Node 22; source compilation is strict, and missing dependencies do
not turn integration tests into skips.

`test:feature-pack` installs a freshly packed SDK outside the workspace, verifies
that the new feature peers remain absent while native vector operations work,
then explicitly installs them and runs the real contracts plus both downstream
native consumers. It cleans up its temporary project afterward.
