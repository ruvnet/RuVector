# ADR-289: Desktop Host Adapters, Lifecycle CLI, and Embedding Surfaces

- **Status**: Accepted
- **Date**: 2026-08-03
- **Updated**: 2026-08-03 — desktop surface implemented: Tauri Reader with FR004 runtime-selection ladder, verify/capability/runtime screens, Agent Dock. rvm-launch/rvm-ffi/rvm-node crates are cross-repo in ruvnet/rvm.
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-283, ADR-284, ADR-285, ADR-286, ADR-287, ADR-288
- **Tags**: rvm, host-adapters, ffi, tauri, node, cli, runtime-selection, isolation

## Context

RVM already provides partitions, capabilities, witnesses, proof gates,
scheduling, memory management, measured boot, and WASM agent lifecycle. What it
does not provide is a desktop host runtime that a packaged `.exe`, `.dmg`, or
`.deb` can invoke. Without that, the Tauri-based RVF Reader has nothing to call,
and `@ruvector/forge` has no way to validate or inspect an RVF from Node.

The gap is not a single missing entry point. A desktop package needs four
distinct things: platform-specific isolation adapters that differ substantially
between Windows, macOS, and Linux; an operator-facing lifecycle CLI; a stable C
ABI that a Tauri process can link against; and Node bindings for the build
tooling. Each has a different consumer and a different stability contract, so
each is a separate crate.

Runtime selection is also a security decision, not a convenience. The reader
picks the strongest compatible runtime, and the order in which it prefers them
determines what isolation an installed agent actually gets. If application
configuration or an environment variable could reorder that list, a downgrade to
a weaker runtime would be a one-line attack. FR004 therefore states that the
order may be changed only by signed policy.

Finally, hosted RVM running as an ordinary desktop process must not claim bare
metal isolation. Overstating the boundary is the failure mode that turns a
security ADR into marketing, and the requirements explicitly gate the phrase
"hardened isolation" behind independent escape testing.

## Decision

Four crates provide the desktop and embedding surface: `rvm-host` for
platform adapters, `rvm-launch` for the lifecycle CLI, `rvm-ffi` for the stable
C ABI consumed by the Tauri RVF Reader, and `rvm-node` for the Node bindings
consumed by `@ruvector/forge`. Runtime selection follows a fixed order that only
signed policy may change.

### 1. `rvm-host` — platform adapters

`rvm-host` provides adapters for:

```text
Windows · macOS · Linux · Browser · QEMU · RVM bare metal
```

Each adapter implements the same host-capability interface, so the layers above
it — `rvm-launch`, `rvm-ffi`, `rvm-node` — are backend-agnostic. The adapters
differ in what isolation primitives they compose beneath WASM isolation:

| Backend | Isolation composed under WASM |
|---|---|
| Windows | Job Objects, restricted tokens, filesystem restrictions, outbound network controls |
| macOS | application sandboxing, hardened runtime, scoped entitlements, notarization |
| Linux | namespaces, cgroups, seccomp, restricted mounts, network namespaces |
| Browser | the browser's own origin and WASM sandbox |
| QEMU | Linux microVM boundary |
| Bare metal RVM | partition memory isolation, capability tables, device leases, measured boot, witnessed security gates |

**Hosted RVM uses operating system isolation plus WASM. It does not claim bare
metal isolation while executing as a normal desktop process.** The adapter
reports its actual isolation class, that class appears in the witness record,
and the reader surfaces it rather than flattening every backend into one
reassuring label.

**Native extensions never load directly into the RVF Reader process.** A native
acceleration segment requires a separate sandbox or an RVM partition. This is
what keeps the reader process's isolation claim honest: a dynamically loaded
native library inside the reader would have the reader's full privileges and
would silently void the WASM boundary.

### 2. Runtime selection order (FR004)

The reader selects the strongest compatible runtime in this order:

```text
1. Native RVM
2. Operating system isolation plus WASM
3. WASM
4. Linux microVM
5. Unsupported
```

Selection walks the list and takes the first entry the host can actually
provide. "Unsupported" is a terminal outcome: the reader refuses to execute
rather than falling back to something weaker than the list allows.

**The order may be changed only by signed policy.** It is not configurable
through CLI flags, environment variables, application settings, or installer
options. An unsigned attempt to reorder it is ignored and recorded. The selected
runtime and the policy hash that governed the selection are both written to the
witness record, so an auditor can tell whether an agent ran under the default
order or a signed override.

### 3. `rvm-launch` — lifecycle CLI

`rvm-launch` exposes the operator-facing commands:

```bash
rvm inspect agent.rvf        # read manifest and capability requests, no execution
rvm verify agent.rvf         # signature and hash verification, no execution
rvm run agent.rvf            # select runtime, load, execute
rvm suspend INSTANCE_ID      # halt at an instruction boundary
rvm resume INSTANCE_ID       # reconstruct and continue
rvm checkpoint INSTANCE_ID   # produce a CompressedCheckpoint
rvm witness INSTANCE_ID      # export the witness chain
rvm terminate INSTANCE_ID    # destroy the instance
```

`inspect` and `verify` **never execute RVF content**. This is the same rule that
governs build workers and package scanning: inspection and packaging must be
possible on an untrusted RVF without running it. `inspect` reports the manifest,
declared capability requests, runtime requirements, and compatibility
information; `verify` produces a witness record for the verification result
whether it passes or fails.

`run` performs verification first, refuses execution when the RVF requires
unsupported capabilities, and reports the selected runtime and its isolation
class before the agent starts.

Suspend, resume, and checkpoint operate on the state chain defined in ADR-288,
so an instance suspended under one host adapter can be resumed under another.

### 4. `rvm-ffi` — stable C interface

`rvm-ffi` exposes a stable C ABI for Tauri and other native hosts:

```text
rvm_validate
rvm_inspect
rvm_create
rvm_start
rvm_suspend
rvm_resume
rvm_checkpoint
rvm_export_witness
rvm_terminate
```

Contract properties:

- **Stability.** The symbol set and their signatures are a versioned contract.
  Additive changes get new symbols; existing symbols do not change meaning.
- **No panics across the boundary.** Every function returns a status code;
  Rust panics are caught at the boundary and converted.
- **Stable machine-readable error codes.** The same code space the CLI and the
  Node bindings report, so a Tauri reader, a shell script, and Forge all
  classify a failure identically.
- **No implicit execution.** `rvm_validate` and `rvm_inspect` do not execute RVF
  content; `rvm_create` allocates an instance without starting it; `rvm_start`
  is the only symbol that begins execution.
- **Explicit ownership.** Handles are opaque; the caller releases them through
  `rvm_terminate`, and the library does not retain caller-owned buffers past a
  call.

The Tauri RVF Reader links this surface. It does not reimplement verification,
capability reconciliation, or runtime selection — those stay in RVM, so the
desktop reader and the CLI cannot drift apart in what they enforce.

### 5. `rvm-node` — Node bindings

`rvm-node` provides the Node API used by `@ruvector/forge`, supporting Node.js
20 or later, matching the CLI's requirement.

Forge uses it for local RVF validation before upload, manifest inspection while
generating the build manifest, and artifact verification after download. All of
these are non-executing operations; the Node bindings expose the same
inspect/verify surface as the FFI and return the same stable error codes, so
`npx @ruvector/forge validate agent.rvf` and `rvm verify agent.rvf` agree.

Local validation for RVFs below 1 GB completes under two seconds excluding full
payload hashing, which is what makes pre-upload validation a usable default
rather than an opt-in.

### 6. Forge integration contract

Every generated package embeds the runtime identity block:

```json
{
  "rvfIdentity": "sha256 value",
  "rvmVersion": "semantic version",
  "rvmCommit": "source revision",
  "runtimeProfile": "wasm",
  "capabilityPolicyHash": "sha256 value",
  "stateSchemaVersion": 1,
  "witnessSchemaVersion": 1
}
```

Forge rejects combinations absent from the published RVM compatibility matrix.
The host adapters read this block at startup to confirm that the installed
runtime matches what the package was built against, and refuse to run on a
mismatch rather than attempting a best-effort interpretation.

### 7. Reader startup budget

Reader startup completes in **under 500 milliseconds before model loading**.
That budget covers process start, RVM initialization through `rvm-ffi`, manifest
verification, capability reconciliation, and runtime selection. Model loading is
excluded and uses the progressive-loading path for large segments, so a
multi-gigabyte model does not block the reader from showing that it started and
what isolation class it obtained.

Full payload hashing is likewise outside the startup budget; segment
verification happens before each segment is loaded rather than all at once up
front.

### 8. Uninstallation

Uninstallation removes the runtime and state according to policy. State removal
follows ADR-288: deleting accumulated state does not require deleting the base
RVF, and the two dispositions are separate policy decisions. Shared-reader
installations additionally deregister the `.rvf` file-type association.

## Acceptance criteria

1. One signed RVF runs unchanged through hosted Linux, Windows, macOS, QEMU,
   and bare metal RVM adapters, producing identical deterministic evaluation
   hashes.
2. Each adapter reports its actual isolation class in the witness record;
   hosted desktop execution never reports bare metal isolation.
3. Runtime selection follows Native RVM → OS isolation + WASM → WASM → Linux
   microVM → Unsupported, and an unsigned attempt to reorder it via flag,
   environment variable, or config is ignored and recorded.
4. A signed policy that reorders runtime selection takes effect, and the
   governing policy hash appears in the witness record.
5. `rvm inspect` and `rvm verify` complete on an untrusted RVF without executing
   any of its content, and `verify` emits a witness record on both pass and
   fail.
6. `rvm run` refuses execution when the RVF requires unsupported capabilities.
7. An instance suspends under one host adapter, checkpoints, and resumes under
   another with identical reconstructed state.
8. `rvm witness` exports a complete, cryptographically verifiable witness chain
   for the instance.
9. The `rvm-ffi` symbol set matches the declared contract, returns stable error
   codes, does not propagate panics across the boundary, and begins execution
   only through `rvm_start`.
10. The Tauri RVF Reader performs verification, capability reconciliation, and
    runtime selection exclusively through `rvm-ffi`, with no duplicate
    implementation in the desktop layer.
11. `@ruvector/forge validate` via `rvm-node` and `rvm verify` return the same
    verdict and the same error code for the same RVF on Node.js 20 or later.
12. Local validation of an RVF below 1 GB completes under two seconds excluding
    full payload hashing.
13. Reader startup completes under 500 milliseconds before model loading.
14. A package whose embedded runtime identity block is absent from the RVM
    compatibility matrix is rejected by Forge at build time and by the adapter
    at startup.
15. No native extension loads into the RVF Reader process; native acceleration
    runs in a separate sandbox or RVM partition.
16. Undeclared filesystem and network access is denied under every adapter, and
    each denial is witnessed.
17. Uninstallation removes runtime and state according to policy, and
    shared-reader installs deregister the `.rvf` association.

## Consequences

### Positive

- A packaged desktop installer finally has something to invoke; the reader is a
  thin shell over RVM rather than a parallel implementation.
- Keeping verification and selection inside RVM prevents the CLI, the desktop
  reader, and Forge from enforcing three different things.
- Signed-policy-only runtime ordering removes the cheapest downgrade attack.
- Reporting the real isolation class per adapter keeps the security claim
  honest and gives escape testing a concrete target.
- Shared error codes across CLI, FFI, and Node make failures diagnosable
  without reading three codebases.

### Negative

- Four crates and six adapters is a large surface to keep behaviorally
  identical; divergence between adapters is the most likely source of subtle
  bugs.
- A stable C ABI constrains internal refactoring for the life of the contract.
- macOS notarization and Windows signing are prerequisites for the adapters'
  real-world isolation properties, adding external dependencies to the release
  path.
- A 500 ms startup budget that includes verification and reconciliation is
  tight and will constrain how much work can move to startup later.
- Signed-policy-only runtime ordering makes legitimate operational overrides
  slower, since they require a signing step.

## Alternatives Considered

- **Implement verification and runtime selection in the Tauri reader**:
  rejected because the desktop reader and the CLI would drift, and the reader
  would become a second security-critical implementation.
- **One crate covering host, CLI, FFI, and Node**: rejected because the four
  surfaces have different consumers and different stability contracts; a Node
  binding change should not force a C ABI version bump.
- **Allow runtime order via configuration or environment variable**: rejected
  because it makes an isolation downgrade a one-line change.
- **Report a single uniform isolation label across backends**: rejected because
  it would let hosted desktop execution imply bare metal guarantees.
- **Load native acceleration in-process for speed**: rejected because it voids
  the reader process's isolation boundary; acceleration requires a separate
  sandbox or partition.
- **Cross-compile all targets from one host**: rejected because DMG and Debian
  bundles require native build environments, so native workers are needed
  regardless.
- **Skip the Node bindings and shell out to the CLI from Forge**: rejected
  because error codes, streaming status, and structured results would have to
  be reparsed from text, and the two paths would diverge.

## Implementation Surfaces

- `rvm-host` — Windows, macOS, Linux, Browser, QEMU, and bare metal adapters;
  isolation-class reporting
- `rvm-launch` — `inspect`, `verify`, `run`, `suspend`, `resume`, `checkpoint`,
  `witness`, `terminate`
- `rvm-ffi` — `rvm_validate` … `rvm_terminate`, versioned C ABI, stable error
  codes, panic barrier
- `rvm-node` — Node.js 20+ bindings for `@ruvector/forge`
- `rvm-rvf` — non-executing manifest reading and segment verification used by
  inspect and verify
- `rvm-policy` — signed runtime-selection order and capability policy
- `rvm-witness` — isolation class, selected runtime, governing policy hash,
  verification and denial records
- Tauri RVF Reader — links `rvm-ffi` only
- RVM compatibility matrix and the embedded runtime identity block
