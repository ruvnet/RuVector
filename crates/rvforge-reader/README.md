# RVForge Reader

Tauri v2 desktop reader for signed `.rvf` agent packages — the host described
in [ADR-289](../../docs/adr/ADR-289-desktop-host-adapters.md). RVF reading,
verification, capability derivation, runtime selection, and encrypted state
capsules are real; installing and running an agent are not, because they need
`rvm-ffi`, which does not exist yet.

## Layout

```text
src/inspect.rs      read and verify a package without executing it
src/receipts.rs     the local witness log — one record per verification result
src/capability/     derive the P6 install-time capability contract
                    mod.rs the rules, text.rs the sentences users consent to
src/runtime.rs      apply the FR004 runtime selection order
src/state/mod.rs    ADR-288 state-capsule layout and lineage rules
src/state/crypto.rs XChaCha20-Poly1305 sealing, per-install key
src/dock.rs         ADR-295 dock entry: the agent/system split, in the types
src/dock_text.rs    sanitizing and screening agent-authored strings
src/dock_state.rs   the eight dock states and who may move between them
src/dock_roster.rs  multi-agent policy and the two-interaction control path
src/dock_events.rs  D8 event thresholds — what may interrupt
src/commands.rs     Tauri command wrappers (feature `desktop`)
src/lib.rs          module root + the security invariants this crate holds
ui/                 four screens, plain HTML/CSS/JS, no build step
assets/             vendored copy of compatibility-matrix.json
capabilities/       Tauri v2 ACL: file dialog only
tests/              runtime selection, capability card, inspection, state, dock
```

## Build and test

The crate is a **standalone workspace** (its `Cargo.toml` carries an empty
`[workspace]` table) and is listed in the repo root's `workspace.exclude`. A
Tauri app pulls a large, app-specific dependency graph — webview bindings,
bundler, plugin build scripts — that every `cargo build --workspace` in the
parent repo would otherwise pay for.

Tauri is behind an optional `desktop` feature, so the core logic builds and
tests without a webview. `rvf-forge-core` is a path dependency on a member of
the *root* workspace; that works across the workspace boundary because it
inherits its own workspace's `version` and `lints`.

```bash
cd crates/rvforge-reader
cargo check          # core only — no webview packages needed
cargo test           # 113 tests, no Tauri dependency
```

This is the CI-testable path. Building the desktop shell additionally needs the
platform webview development packages (`libwebkit2gtk-4.1-dev`,
`libjavascriptcoregtk-4.1-dev`, `libsoup-3.0-dev` on Debian/Ubuntu; WebView2 on
Windows; Xcode command line tools on macOS):

```bash
cargo check --features desktop
cargo run   --features desktop     # launches the window
```

`cargo tauri dev` and `cargo tauri build` need the CLI
(`cargo install tauri-cli --version '^2'`). There is no Node toolchain: the
frontend is static files under `ui/`, referenced by `build.frontendDist`.

Before producing installers, regenerate the icon set with
`cargo tauri icon icons/icon.png` — the committed PNGs are placeholders and
there is no `.ico` or `.icns` yet, which Windows and macOS bundling require.

## Inspection, verification, and receipts

`inspect::inspect` calls `rvf_forge_core::inspect_bytes` and reports the real
identity (SHA-256 of the container), the segment inventory, and the capability
classes the container declares. It checks nothing, so its verification status is
always `unverified`.

`inspect::verify` calls `rvf_forge_core::verify_bytes` — root manifest present,
per-segment content hashes, unsigned-executable refusal — and appends one
receipt to `<state_root>/receipts.jsonl`, on a pass and on a refusal alike
(ADR-284 §1 requirement 9). Both operations run against a single read of the
file, so a package cannot be swapped between inspection and verification.

Neither executes package content.

## What is not implemented

| Gap | Current behavior | Needs |
|---|---|---|
| Publisher identity | `publisher: null`, and a present signature reports `not-checked` rather than `verified`. | A trust store mapping Ed25519 keys to publisher identities; `VerifyOptions::trusted_keys` is empty without one |
| Scoped capability grants | The container declares classes with no scopes, so each granted class says so and raises a manual-review trigger. An unsigned `<file>.rvf.manifest.json` sidecar may narrow a class to a specific scope, never widen one. | A signed `CapabilityManifest` segment inside the RVF |
| Install / Customize permissions | Disabled buttons. | The install flow |
| Emergency controls | Disabled buttons on the Runtime screen. | `rvm-ffi` lifecycle calls (pause, terminate, revoke, rollback) |
| Dock roster | In-process, seeded by `dock_add_scaffold_agent`, which reports the unknown value for every security-bearing field (unverified publisher, no confinement, no witness chain). | RVM telemetry over `rvm-ffi`; pause is `rvm_suspend`, terminate is `rvm_terminate` (ADR-295 implementation note) |
| Dock instruction field | Disabled placeholder. | The agent instruction channel, once `rvm-ffi` is wired in |
| Dock witness status | "no witness chain". | `rvm witness` export (ADR-289 §3) |

Each of these reports *absence*, never a benign default. `SignatureStatus`
distinguishes `not-checked` from `verified` for exactly this reason: a signature
nobody could check is not a signature the user can rely on.

## State capsules

`state::seal` / `unseal` use XChaCha20-Poly1305 — pure Rust, no unsafe, and
constant-time on hosts without AES hardware, which the Windows, macOS, and Linux
ARM builds cannot assume. The 192-bit nonce makes a random nonce per capsule
safe without a counter, and a counter is what a state directory the user may
copy, restore, or roll back would silently break.

One 32-byte install key lives at `<state_root>/install.key`, created `0600` on
first use from the OS CSPRNG; per-capsule keys are HKDF-SHA256 derived from it
with the base RVF identity as `info`. The base identity is written into the
capsule header in the clear — so a foreign lineage is refused without needing a
key — and is covered by the AEAD's additional data, so relabelling a capsule to
another lineage fails authentication rather than opening as that lineage's
state. Customer-held keys (ADR-288 §4) are constructible today through
`InstallKey::from_bytes`; a key-management UI is not.

## Security invariants

These are enforced in the library and covered by tests. They must survive
whatever fills the gaps above.

1. **RVF content is never executed.** No code path loads, links, or interprets
   a segment; the only bytes read from an executable segment are read to hash
   them. `inspect` and `verify` must be safe on an untrusted package
   (ADR-289 §3, ADR-284 §1 requirement 7).
2. **Verification precedes any load, and every result is witnessed.** An
   unchecked package reports `unverified` rather than being assumed good, and a
   refusal is written to the receipt log exactly as a pass is.
3. **Capability rendering is default-deny, and no card without verification.**
   Every one of the fifteen ADR-286 classes the container does not declare
   appears in the "cannot" list; a package that did not verify yields a card
   that grants nothing, as does a missing or rejected manifest.
4. **No vague permission prose.** Broad scopes (`all-files`, `*`,
   `unrestricted`) and banned phrases such as "access your computer" are
   rejected at derivation time, not filtered in the UI (requirements P6).
5. **No network calls.** Nothing here opens a socket. The packaged app's CSP
   has no remote origin in any directive, and the asset protocol is disabled.
6. **The runtime order is not configurable.** It is read from the vendored
   compatibility matrix, which is compiled in via `include_str!` so that
   swapping a file on the installed machine cannot reorder it. ADR-289 permits
   reordering only by signed policy; that path is not implemented, and
   `policy_source` always reports `embedded-default`.
7. **Hosted mode does not claim bare-metal isolation.** The card shows the
   isolation class the matrix records for the selected profile —
   `os-sandbox+wasm`, never `partition` (ADR-285).
8. **Agent input cannot reach dock chrome.** An agent supplies task text and
   progress, through `AgentProvidedStatus::from_agent(&str, i64)` and nowhere
   else. State, trust badge, network indicator, permission summary, witness
   status, resource usage, and cost live in `SystemOwnedStatus`, whose fields
   are private and whose single constructor takes no agent-derived value
   (ADR-295 §7).

Today `HostProfile::detect()` claims no OS confinement, no KVM, and no measured
boot, because the `rvm-host` adapters do not exist yet. Selection therefore
lands on plain `wasm`. That is the honest answer, and it will move up the order
as adapters land — not before.

## Screens

1. **Open** — pick or type a path, see file identity and signature status.
   Unverified renders as a warning, not a neutral state.
2. **Capabilities** — the P6 contract: "This agent requests" beside "This agent
   cannot", with Install / Customize Permissions / Cancel.
3. **Runtime** — selected profile, isolation class, mechanisms engaged, the
   selection order and where it came from, a per-profile eligibility table, and
   the emergency controls (Pause · Terminate · Disconnect Network · Revoke
   Capabilities · Rollback State) as disabled placeholders.
4. **Dock** — the ADR-295 agent dock. A collapsed pill (icon · name ·
   agent-reported task · progress · state · Pause · Stop · Expand), an expanded
   panel with the ten §2 elements and the seven-line §8 capability card, and the
   §5 multi-agent collapse: one active agent, two secondary icons, an overflow
   count, and aggregates computed over every agent including the hidden ones.

## The dock trust boundary (ADR-295 §7)

Agent-authored text is confined to `.agent-region` — an inset, dashed,
monospace block on a tinted background, preceded by a dock-drawn
"Agent-reported" label and clipped to one line. It reaches the DOM only through
`textContent`, so it cannot produce markup, a class name, or a state indicator.
System chrome uses presentation the agent has no way to emit from inside that
region.

Before it is rendered, agent text is stripped of ANSI escapes, control
characters, bidi overrides and line breaks, capped at 120 characters, and then
screened for chrome mimicry: a task string reading "Paused", "Stopped", "✓
Network disabled", or "Verified publisher" is withheld and flagged rather than
drawn next to the dock's own labels. System-authored lines are cleaned but not
screened, because "Running → Paused" written by the runtime *is* the label.

The state machine carries the same rule: an agent may assert `Idle`, `Running`,
`WaitingForApproval`, `Error`, or `Completed`, and nothing else. It cannot put
itself into `Paused`, `CapabilityDenied`, or `Quarantined`, and it cannot leave
them. Pause and terminate are legal for the user from every non-terminal state,
in one call, with no confirmation chain — which is what keeps the acceptance
test ("identify, read, inspect permissions, terminate, in two interactions")
satisfiable: expand carries the permissions and the capability card, terminate
is reachable from the collapsed pill.

Event thresholds (§9) allow only approvals, policy violations, cost limits,
failures, and milestones to interrupt. Progress, tool calls, network samples,
and state changes accrue silently into the expanded panel's recent actions.

## Keeping the matrix in sync

`assets/compatibility-matrix.json` is a vendored copy of
`docs/research/rvf-forge/compatibility-matrix.json`. A test compares the two
byte-for-byte and fails on drift; update both together.
