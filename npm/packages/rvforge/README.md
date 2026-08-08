# @ruvector/rvforge

**RVForge — one canonical RVF to signed platform installers.**

`forge` turns a single `.rvf` agent into installable packages for Windows,
macOS, Linux, and RVM, without the agent inside each package ever differing.
The RVF identity, contents, policies, and signatures are the same in the
`.dmg` as in the `.msi`.

## Start with the walkthrough

[![The Sandbox Is Not the Boundary — an illustrated RVForge walkthrough](https://ruvnet.github.io/RuVector/rvforge/preview.jpg)](https://ruvnet.github.io/RuVector/rvforge/)

**[The Sandbox Is Not the Boundary →](https://ruvnet.github.io/RuVector/rvforge/)**

An illustrated walkthrough at
<https://ruvnet.github.io/RuVector/rvforge/>. It explains in plain language
why process isolation does not bound an agent, then goes command by command
from authoring a signed artifact to running it under the capability gate —
ending by flipping one byte to show what the identity check is actually for.

See [ADR-283](../../../docs/adr/ADR-283-rvf-forge-canonical-installer-pipeline.md)
for the design and `docs/research/rvf-forge/requirements.md` for the full
requirements.

Requires Node.js 20 or later.

## Quick start

From an empty directory to a staged build, with nothing else installed:

```bash
npx @ruvector/rvforge init --keygen        # forge.config.json, rvforge.json, a publisher key
npx @ruvector/rvforge create               # writes a signed agent.rvf
npx @ruvector/rvforge validate agent.rvf --deep
npx @ruvector/rvforge test agent.rvf
npx @ruvector/rvforge build --mode embedded
npx @ruvector/rvforge verify forge-out
```

`create` is what makes the rest runnable: every other command reads an `.rvf`,
and until you have one there is nothing to validate, pack, or build.

## Commands

```bash
npx @ruvector/rvforge init                     # scaffold forge.config.json
npx @ruvector/rvforge init --keygen            # …plus rvforge.json and a publisher key
npx @ruvector/rvforge create                   # write a signed agent.rvf
npx @ruvector/rvforge validate agent.rvf       # structural check, no execution
npx @ruvector/rvforge build agent.rvf          # local build (--mode embedded|thin)
npx @ruvector/rvforge submit agent.rvf --yes   # hosted build
npx @ruvector/rvforge status BUILD_ID
npx @ruvector/rvforge download BUILD_ID
npx @ruvector/rvforge verify AgentSetup.exe
```

The publisher verbs work against `rvforge.json` and a registry rather than
against installers:

```bash
npx @ruvector/rvforge pack agent.rvf           # the ADR-294 validation list
npx @ruvector/rvforge test agent.rvf           # inspection-only test categories
npx @ruvector/rvforge publish agent.rvf        # sign and append a release
```

Targets are `windows-x64`, `windows-arm64`, `macos-x64`, `macos-arm64`,
`macos-universal`, `linux-x64`, `linux-arm64`, and `rvm`, with `windows`,
`macos`, and `linux` accepted as aliases. Pass them after the `.rvf`:

```bash
npx @ruvector/rvforge build agent.rvf windows macos linux
```

Omit the `.rvf` and forge uses `rvf` from the config; omit the targets and it
uses `targets`.

### `create`

Writes a valid, signed `.rvf`. It reads the project metadata and declared
capabilities from `rvforge.json` and signs with the key `init --keygen`
recorded there, so the common case takes no arguments:

```bash
forge create                       # -> agent.rvf, signed
forge create build/my-agent.rvf    # explicit output path
forge create --from ./payload      # include real payload files as segments
forge create --unsigned            # development build, no signature
forge create --force               # overwrite an existing output file
```

With no `--from`, the output is a minimal but complete agent skeleton: a `META`
segment declaring the capability classes the project requests, and a signed
root `MANIFEST`. That is enough for `validate`, `test`, `pack`, `publish`, and
`build` to run, which is the point — you can walk the whole pipeline before you
have a model to put in it.

`--from <dir>` adds every file under the directory as a segment, visited in
sorted order. A `.wasm` file becomes an executable `WASM` segment and is signed
individually; anything else becomes an opaque `VEC` payload segment. Payload
bytes are read, hashed, and stored — never parsed as code, linked, or executed.

**Refuses to overwrite.** An existing output file is an error (`FORGE_E_IO`)
unless `--force` is passed, so a second `create` cannot silently discard the
artifact you already signed and published.

**Signing.** `--key-file` overrides the path in `rvforge.json`. Without either,
`create` fails with `FORGE_E_KEY` rather than quietly writing something
unsigned; pass `--unsigned` when that is what you actually want. Only the key
*id* is ever printed. An unsigned container carrying an executable segment is
refused by `validate` unless you also pass `--allow-unsigned`.

**Deterministic.** The same project, key, and inputs produce byte-identical
output. Timestamps are zero, segment ids come from position, payload files are
sorted, and the file id is derived from the project's name, version, and
publisher — so rebuilding an agent reproduces its identity rather than minting
a new one.

The container `create` writes is the format `crates/rvf-forge-core` reads;
`scripts/rvforge-parity-check.sh` proves it by having the Rust crate inspect
and verify a container this command produced, signatures included.

### `validate`

Checks that the file is a well-formed RVF: root-manifest magic and CRC32C,
manifest field consistency, a segment-header walk, and signature presence. It
reads bytes and never executes, links, or interprets RVF content.

The default pass touches one 4 KiB page plus one 64-byte header per segment,
which keeps it under two seconds for RVFs below 1 GB. `--deep` additionally
streams the whole file through SHA256 to compute the canonical RVF identity.

```bash
forge validate agent.rvf --deep
forge validate agent.rvf --allow-unsigned   # development builds only
```

An RVF carrying executable segments (kernel, eBPF, WASM) with no root-manifest
signature fails with `FORGE_E_UNSIGNED_SEGMENT` unless `--allow-unsigned` is
passed.

### `build`

Produces the canonical build manifest, one staged bundle per target, a software
inventory, SHA256 checksums, a provenance record, and a witness receipt:

```text
forge-out/
├── build-manifest.json                canonical, deterministic
├── inventory.json                     software inventory + bundle layout
├── provenance.json                    what was built, from what, by whom
├── checksums.txt                      sha256sum-compatible
├── receipts.jsonl                     witness chain, append-only
└── bundles/
    └── <target>/
        ├── rvf/agent.rvf              embedded payload…
        ├── rvf/locator.json           …or the signed locator, in thin mode
        └── reader/reader-slot.json    where the RVF Reader goes
```

The manifest is deterministic: keys, targets, and capability grants are sorted,
and nothing time-, path-, or host-dependent appears in it. The same logical
build description always serialises to the same bytes and the same
`manifestSha256`. Everything that varies per run lives in the provenance record.

Installer generation requires the Tauri packaging layer. Until that lands the
result is labelled `staged` and forge does not claim to have produced an
installer. A failed build leaves no output directory at all, rather than a
half-written one.

### Packaging modes

`--mode` selects how the RVF reaches each bundle, overriding
`packaging.mode` in the config for one build:

```bash
forge build agent.rvf --mode embedded    # the bundle carries the whole RVF
forge build agent.rvf --mode thin        # the bundle carries a signed locator
```

**Embedded** (FR001) stages the complete RVF into every target's bundle, so the
package runs with no network access. The same bytes go into every bundle — that
is core invariant 1, *the embedded RVF hash must be identical across every
platform package* — and forge does not take the copy on trust: it re-hashes each
staged copy and fails the build with `FORGE_E_VERIFY_FAILED` if any of them
diverges, rather than shipping platform packages that disagree about what the
agent is.

**Thin** (FR002) stages a signed RVF locator instead of the payload: the
distribution URL, the RVF identity and size, the capability-policy hash, and a
signature *slot*. The reader resolves the locator, checks the digest it names,
and verifies before executing. `packaging.distributionUrl` is required — a thin
package with nowhere to fetch from is rejected at manifest generation.

The reader slot in each bundle is a JSON descriptor, never an executable.
Nothing in a staged bundle is runnable.

### Compatibility enforcement

Forge refuses any packaging-mode / target / runtime-profile combination absent
from the published RVM compatibility matrix (ADR-291 §2), before the RVF is read
and before anything is uploaded:

```console
$ forge build agent.rvf rvm
error: mode=embedded target=rvm runtime=wasm is absent from the RVM compatibility
       matrix — runtime profile "wasm" has no platform entry for target "rvm"
       (os "rvm", arch x64). Closest supported combination: mode=embedded
       target=linux-arm64 runtime=wasm.
code:  FORGE_E_UNSUPPORTED_TARGET
```

Forge never approximates, downgrades, or substitutes a runtime to make an
unsupported request succeed; it names the nearest supported combination and
lets you decide. The matrix is hash-addressed, and both `provenance.json` and
`inventory.json` record the revision that admitted the build, so a past
admission decision can be reconstructed.

`src/compatibility-matrix.json` is vendored. **The canonical copy is
`docs/research/rvf-forge/compatibility-matrix.json`** — change it there and
re-copy; `tests/compat.test.ts` fails when the two diverge.

### Witness receipts

Every build and every verification appends a receipt to `receipts.jsonl`,
following the registry data model. A receipt's id is the SHA256 of its canonical
JSON (excluding `receiptId` and `signatures`), and receipts hash-chain per
subject through `prevReceipt`:

```json
{"schemaVersion":1,"type":"witness-receipt","receiptId":"sha256:…","subject":"sha256:…",
 "event":"build","outcome":"pass","actor":{"kind":"builder","id":"@ruvector/rvforge@0.1.0"},
 "evidence":{…},"timestamp":"2026-08-03T00:00:00.000Z","prevReceipt":null,"signatures":[]}
```

`verify` checks the chain before it appends to it: an edited receipt fails
because its recomputed id no longer matches, and a removed or reordered one
fails because the link no longer resolves. Either way the result is
`FORGE_E_VERIFY_FAILED`, and forge does not append onto a chain it just refused.

`signatures` is always empty on output — forge holds signing references, never
key material, so the array is a slot for a signing worker to fill.

`receipts.jsonl` is deliberately absent from `provenance.json`: it is
append-only and grows on every verification, so recording its digest would make
a build's own provenance fail the moment the chain was extended.

### `verify`

Recomputes every digest in a provenance record, re-derives the manifest hash
from the manifest's own bytes, and checks the witness chain:

```bash
forge verify forge-out                                  # a whole build directory
forge verify forge-out/bundles/linux-x64/rvf/agent.rvf  # one artifact
forge verify Agent.dmg --provenance path/to/provenance.json
```

Re-deriving the manifest hash from content is what makes the check meaningful:
rewriting the manifest *and* patching its recorded digest still fails.

### Hosted builds

`submit`, `status`, and `download` talk to the hosted build service. The
service is not deployed yet, so these currently fail with `FORGE_E_NETWORK`
against a live endpoint; the request and response shapes are the contract it
will be built to.

```bash
export FORGE_API_URL=https://forge.example
export FORGE_API_TOKEN=...        # never pass a token as a flag
forge submit agent.rvf --yes
```

`submit` validates locally, builds the manifest, and prints the estimated build
time, output size, and cost before anything is uploaded. The upload itself
requires `--yes`, so an unattended run cannot ship a confidential RVF to a
remote worker by accident. Downloads are hashed on arrival and discarded on a
digest mismatch.

## Unattended use

Every command accepts `--json` and writes a single envelope to stdout:

```json
{"ok": true, "command": "validate", "data": { }, "exitCode": 0}
{"ok": false, "command": "verify", "error": {"code": "FORGE_E_VERIFY_FAILED", "message": "..."}, "exitCode": 9}
```

Exit codes are stable:

| Code | Exit | Meaning |
|---|---|---|
| `FORGE_E_USAGE` | 2 | Bad arguments |
| `FORGE_E_INVALID_RVF` | 3 | Missing, truncated, bad magic, failed checksum |
| `FORGE_E_UNSIGNED_SEGMENT` | 4 | Executable segment with no signature |
| `FORGE_E_UNSUPPORTED_TARGET` | 5 | Target outside the supported matrix |
| `FORGE_E_MANIFEST` | 6 | Malformed or inconsistent build manifest |
| `FORGE_E_NETWORK` | 7 | Build service unreachable |
| `FORGE_E_AUTH` | 8 | Credentials missing or rejected |
| `FORGE_E_VERIFY_FAILED` | 9 | A recomputed digest diverged |
| `FORGE_E_IO` | 10 | Filesystem failure |
| `FORGE_E_NOT_FOUND` | 11 | Unknown build, artifact, or record |
| `FORGE_E_CONFIG` | 12 | Bad `forge.config.json` |
| `FORGE_E_TOOLCHAIN` | 13 | Required local build tool unavailable |
| `FORGE_E_POLICY` | 14 | Capability policy absent, vague, or contradictory |
| `FORGE_E_LICENSE` | 15 | No license declared, or one forge cannot reconcile |
| `FORGE_E_KEY` | 16 | Signing key missing, unusable, or over-permissioned |
| `FORGE_E_REGISTRY` | 17 | Registry directory unreadable or inconsistent |
| `FORGE_E_LINEAGE` | 18 | Predecessor chain does not resolve |
| `FORGE_E_INTERNAL` | 20 | A bug in forge |

## Capability policy

`forge init` writes a default-deny policy: every capability class is present
and every allowlist is empty. Grants have to be added explicitly.

```json
{
  "capabilityPolicy": {
    "defaultDeny": true,
    "allow": {
      "network": ["https://api.example.com"],
      "filesystem": ["~/Documents/reports"],
      "devices": [], "memory": [], "models": [], "state": [], "tools": []
    }
  }
}
```

The policy is hashed into `capabilityPolicyHash` and embedded in every package,
so a reviewer can confirm the permissions an installer ships with match the
ones they approved. `defaultDeny` is forced to `true`; a policy that opts out
is rejected rather than silently accepted.

## Publishing

`pack`, `test`, and `publish` implement the ADR-294 §4 publisher CLI. They read
`rvforge.json` — the marketplace project file, scaffolded by
`rvforge init --keygen` — which holds the listing metadata, publisher identity,
license, runtime requirements, and the capability contract the install UX
renders. It is separate from `forge.config.json` on purpose: a build is
reproducible from the build config alone, and nothing in the project file can
change a build's output. Neither file ever holds credentials.

All three take an `.rvf` that `rvforge create` wrote:

```bash
rvforge init --keygen && rvforge create && rvforge pack agent.rvf
```

### `pack`

Runs every item on the ADR-294 validation list that is decidable locally — RVF
structure, publisher signature, executable segments, model provenance,
capability policy, runtime compatibility against the vendored matrix, memory
requirements, external services, software inventory, and license — then emits an
unsigned `CapabilityManifest` and a draft `Release` in the registry data model's
shapes.

```bash
rvforge pack agent.rvf --project rvforge.json
```

Every check runs before the aggregate decides, so a publisher sees all their
problems at once rather than one release at a time. A **vague capability scope**
— `all-files`, `*`, `unrestricted` — is not a failure: ADR-294 §8 makes it
*require human review*, so it is recorded in `manualReviewTriggers`, drops the
security profile to `review-required`, and the pack still succeeds. Treating it
as a failure would push publishers toward describing capabilities less precisely
to dodge the flag. A missing license, an undeclared memory ceiling, a class both
requested and denied, or a runtime absent from the matrix do fail, with
`FORGE_E_LICENSE`, `FORGE_E_MANIFEST`, `FORGE_E_POLICY`, and
`FORGE_E_UNSUPPORTED_TARGET` respectively.

### `test`

Six of the ten P4 test categories are decidable by inspection; four are not.

| Category | Outcome |
| --- | --- |
| Malformed inputs | run — truncated and bit-flipped variants must be rejected cleanly |
| Capability denials | run — all fifteen ADR-286 classes must be requested or denied |
| Filesystem escape attempts | run in reduced form — declared scopes checked for traversal and root escapes |
| State checkpoint and recovery | run — contract fields present |
| Update and rollback | run — version ordering and rollback-safety declared coherently |
| Witness verification | run when a `receipts.jsonl` exists, otherwise skipped with the reason |
| Clean installation | **skipped: requires quarantined runtime (Reader/RVM)** |
| Deterministic evaluations | **skipped: requires quarantined runtime (Reader/RVM)** |
| Network monitoring | **skipped: requires quarantined runtime (Reader/RVM)** |
| Resource exhaustion | **skipped: requires quarantined runtime (Reader/RVM)** |

The four skipped categories are **never reported as passing**. Forge does not
execute an RVF, so it has no evidence about them, and ADR-294 §7 makes a trust
level a statement about evidence actually gathered. `filesystem-escape` says in
its own detail line that runtime probing was not attempted, so a `pass` there
cannot be read as the stronger claim.

`test` exits `9` (`FORGE_E_VERIFY_FAILED`) when a category fails, after printing
the full per-category report.

### `publish`

Targets a local registry directory implementing the storage layout in
`docs/research/rvf-forge/registry-model.md`.

```bash
rvforge publish agent.rvf --registry ~/.rvforge/registry --key-file publisher.key
```

It writes the capability manifest, release, software inventory, and pack
provenance to `objects/sha256/<2-hex>/<digest>.json`; signs the release with an
Ed25519 key over its `releaseId`; appends to
`packages/<publisher>/<name>/releases.jsonl` with the predecessor resolved from
the current head; extends `log/entries.jsonl` and `log/tree-head.json`; and
chains a `WitnessReceipt` for the publish. Then it prints the P4 summary block.

An object's id is `sha256:<hex>` over its canonical JSON **excluding**
`signatures` and its own id field, so adding a countersignature later does not
change what it identifies.

**Key handling.** `rvforge init --keygen` writes the key with mode `600` and
never overwrites an existing one. `publish` refuses a key file any account other
than its owner can read (POSIX only — Windows has no comparable mode bits, and
the check is skipped rather than faked), and refuses a key the publisher record
does not already list: key rotation is a registry operation, not a side effect of
publishing. Key material is read only by `src/keys.ts` and never reaches a log
line, an error message, or a registry object.

**Deviations from `registry-model.md`**, for the parity test against
`crates/rvforge-registry` to reconcile:

- the tree head is a **hash chain**, not a Merkle tree, so it is tamper-evident
  but yields no per-entry inclusion proof;
- the tree head is unsigned — signing it is the registry's act, and a local
  registry has only the publisher's key;
- `packages/<publisher>/publisher.json` and `receipts.jsonl` are additions; the
  documented layout names neither, and the model gives no way to resolve a
  `publisherId` back to its record;
- `publish` is added to the witness event vocabulary, which the model's list
  does not cover.

## Security

- **RVF content is never executed.** Validation, packaging, and scanning are
  inspection-only operations. Embedded mode copies bytes, thin mode writes a
  locator, and the reader slot is a descriptor rather than a binary — nothing
  in a staged bundle is runnable.
- **The build path makes no network calls.** A locator records where an RVF
  will be fetched from; forge does not fetch it.
- **Signing references only, for platform installers.** Forge records which key
  to ask for and which service holds it. Platform-signing key material is never
  read, stored, logged, or transmitted; signing happens on a worker with HSM or
  KMS access.
- **The publisher key is the one exception, and it is contained.** `publish`
  needs an Ed25519 private key to sign a release. It is read only by
  `src/keys.ts`, held as a `KeyObject` rather than bytes, refused when its file
  is readable beyond its owner, and never returned, printed, or written into a
  registry object — only its derived `keyId` and public half are.
- **Credentials come from the environment.** `FORGE_API_TOKEN` is never
  accepted as a flag, so it stays out of shell history and process listings.
- **Messages are redacted.** Bearer tokens, URL userinfo, and long opaque
  strings are stripped from error messages and detail bags before they are
  printed.

## Library use

```ts
import {
  assertCompatible,
  buildManifest,
  readReceipts,
  runVerify,
  validateRvf,
  verifyReceiptChain,
} from '@ruvector/rvforge';

assertCompatible({ mode: 'embedded', targets: ['linux-x64'], runtimeProfile: 'wasm' });

const result = await validateRvf('agent.rvf', { deep: true });
const manifest = buildManifest({
  app: { name: 'My Agent', version: '1.0.0', publisher: 'Example', identifier: 'com.example.agent' },
  identity: result.identity,
  targets: ['windows', 'macos', 'linux'],
  packaging: { mode: 'embedded' },
  runtime: { profile: 'wasm', rvmVersion: '0.1.0', rvmCommit: 'abc1234' },
});

await runVerify('forge-out');
verifyReceiptChain(await readReceipts('forge-out')); // { ok: true, … }
```

The publisher verbs are available the same way:

```ts
import { loadProject, runPack, runPublish, runTestAgent, verifyLog, readLog } from '@ruvector/rvforge';

const project = await loadProject('rvforge.json');

const packed = await runPack({ rvfPath: 'agent.rvf', project });
packed.release.releaseId;        // 'sha256:…', unsigned draft
packed.manualReviewTriggers;     // ADR-294 §8 triggers, if any

const tested = await runTestAgent({ rvfPath: 'agent.rvf', project });
tested.categories.filter((c) => c.status === 'skipped'); // the quarantine-only four

const published = await runPublish({
  rvfPath: 'agent.rvf',
  project,
  registryDir: '/tmp/registry',
  keyFile: 'publisher.key',
});
verifyLog(await readLog(published.registryDir)); // { ok: true, … }
```

## Development

```bash
npm run build      # tsc → dist/
npm test           # jest
npm run typecheck
```

`tests/fixtures/minimal.rvf` is a 4.5 KB synthetic file generated by
`tests/fixtures/rvf-fixture.ts`; a test asserts the two agree, so the committed
bytes cannot drift from the builder. It contains zero-filled payloads and no
model data.
