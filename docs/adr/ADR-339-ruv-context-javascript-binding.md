# ADR-339: A WebAssembly Binding for `ruv://` Context, and What It May Not Carry

**Status:** Accepted

**Date:** 2026-08-23
**Owners:** RVM maintainers
**Tracking:** [ruvnet/rvm#45](https://github.com/ruvnet/rvm/issues/45)

## Context

The question that prompted this was narrow — *could the Meta LLM gateway use
`ruv://` context?* — and answering it honestly required reading the manifest
rather than the prose. The manifest settles it.

`crates/rvm-context/Cargo.toml` on `main`:

```toml
[lib]
crate-type = ["rlib"]

[dependencies]
sha2  = { workspace = true }
sha3  = { workspace = true }

[dev-dependencies]
ed25519-dalek = { version = "=2.1.1", default-features = false }

[features]
default = []
std = ["rvm-types/std", "rvm-cap/std", "rvm-witness/std", ...]
```

and `src/lib.rs` opens `#![no_std]`, `#![forbid(unsafe_code)]`.

**`crate-type = ["rlib"]` is decisive.** An rlib links Rust-to-Rust. There is no
JavaScript, no WebAssembly, and no C ABI surface, and no feature flag produces
one — `cdylib` is a crate-type, not a feature. So no JavaScript consumer can
reach this crate today, and none could be made to by configuration.

Reading the API to size a binding turned up four facts that shaped the decision
more than the crate type did. Each was verified in source, because each was
plausible enough to guess wrong about — and two of them contradict what a
reasonable person would assume from the release notes.

**Capability handles are not portable.** `CapabilityHandle`
(`capability.rs:25`) is `{ index: u32, generation: u32 }` — an index and
generation into a *live local* `CapabilityManager` table. Not a bearer token,
not signed, not serializable. Its own doc says handles are safe from untrusted
callers "because authorization always resolves them through the manager," which
is precisely the point: a handle is meaningless without *that* manager. Two
integers handed from a Rust service to a JS host index a different table and
mean something else. This is the load-bearing constraint on the whole design.

**An allow decision cannot be separated from its witness record.** `authorize`
is `pub(crate)` (`capability.rs:679`), and `AuthorizedRequest` has private
fields *and* private construction, documented as reachable "only after the
runtime has appended a P1 allow witness record." Exposing a bare `authorize()`
is not possible; a binding that authorizes must carry `ContextRuntime`, and
therefore the witness log. That is more surface than expected and the right
outcome — decision-without-witness is exactly what a gateway should not be able
to perform.

**There is no entropy requirement.** `ed25519` appears only in
`crates/rvm-proof`; in `rvm-context` it is a `[dev-dependencies]` entry, tests
only. `rvm-witness` signs with **HMAC-SHA256** (`hmac` + `sha2`, feature
`crypto-sha256`, default on), which is deterministic and keyed. No `getrandom`
or `rand` exists anywhere in the workspace manifest. The real hazard is key
provisioning, not randomness: `rvm-witness` exposes `default_signer()` and
`with_default_key()`, and either reaching JavaScript would let a caller
manufacture audit records that verify.

**No host clock is needed either.** `LogicalContextClock` is a deterministic
counter from zero, and `ContextRuntime::new` uses it. Nothing in the wasm build
needs `Date.now()` or `js_sys`.

Two distribution facts as of this decision: `rvm-context` is not yet on
crates.io (the foundation crates published first and it is behind them in the
rate-limited queue), and there is no npm package. The name `rvm` on npm belongs
to an unrelated project — "Ruff Version Manager", maintainer `vilic`, from
`ruffjs/rvm` — so any JavaScript distribution here is a **new scoped package**,
never an update to that one.

## Decision

Add `crates/rvm-context-wasm`: a `cdylib` wrapper that enables
`rvm-context/std`, binds with `wasm-bindgen`, and publishes to npm as
`@ruvnet/rvm-context-wasm`. It is `publish = false` for crates.io — a
distribution artifact for a different registry, not a library another Rust
crate should depend on.

Four layers, in increasing order of what they require from the host:

1. **URI.** Canonical `ruv://` parsing into structured components, canonical
   re-formatting, and the specific `UriError` variant on rejection rather than a
   generic failure. Pure computation over a string.
2. **Scope.** `ContextScope::from_uri` plus `contains_scope` as a standalone
   predicate. This layer alone answers the shadow-mode question — *would the
   `ruv://` gate have allowed this cross-tenant reach?* — with no capability, no
   runtime, and no key. It is the cheapest useful thing in the binding and
   should stay usable on its own.
3. **Runtime.** A self-contained in-wasm `ContextRuntime`: its own authority,
   grant table, witness ring, and logical clock, surfacing the witness sequence
   on each decision.
4. **Verification.** `verify_chain`, `record_to_digest`,
   `SignedContextEpochReceipt::verify`, `verify_genesis`/`verify_successor`,
   `to_bytes`/`from_bytes`. Pure, and the highest-value exports.

**The boundary is not "no authorization" — it is that the module is its own
authority.** An earlier draft of this ADR drew the line at authorization
entirely. That was the wrong line, and the handle representation is why: the
danger was never that a JavaScript caller might mint a capability, because a
capability minted in the wasm module grants nothing outside it. The danger is
the *illusion* of authority — a gateway that provisions its own scopes, renders
a decision, and reports it as though it said something about a separate
Rust-side authority.

So the honest claim, which belongs in the README's first paragraph rather than a
footnote: **the wasm module is a faithful, deterministic policy simulator.**
Handles are not portable. A decision binds only to the scope table the host
provisioned into it. That is exactly right for shadow-mode evaluation, where
nothing is enforced and the question is whether two policies agree. It is not
evidence about another authority unless that authority provisioned the same
scopes, and anyone promoting it to enforcement must provision the grant table
from the authority that issues real capabilities.

## Consequences

### Positive

- The gateway, and any TypeScript consumer, can construct and validate
  canonical `ruv://` names and evaluate scope containment without a Rust
  toolchain or a running RVM.
- One parser, one canonical spelling. A URI validated in TypeScript and the
  same URI validated in the kernel agree by construction, because they are the
  same code compiled twice — not two implementations kept in sync by review.
- `no_std` + `forbid(unsafe_code)` + no async + no clock + no RNG is close to
  ideal for `wasm32-unknown-unknown`. The austerity that made this crate
  awkward to reach from JS is the same austerity that makes it small and clean
  once wrapped.
- Shadow mode is reachable at layer 2, well before the runtime layer is
  finished, so the motivating consumer is not blocked on the whole binding.

### Negative

- A second artifact to keep in step with the crate. A published npm package can
  drift from a published crate; the version is pinned to the workspace version
  to make drift visible rather than silent.
- The simulator framing is a documentation-strength guarantee, not a
  compiler-strength one. Nothing stops a consumer from provisioning arbitrary
  scopes and reporting the result as authoritative. The gates below are the
  mitigation and they are weaker than a type.
- Anyone who assumes handles are portable will get answers that look right and
  are meaningless. This is the single most likely misuse, which is why it leads
  the README.
- `rvm-context-service` — the durable half — remains out of reach, and
  correctly so. It is a deployment, not a library, and it depends on
  `ruvector-context` and `ruvector-core` by path into the submodule, neither of
  which is published.

## Security / Validation Gates

1. **No default-key signer crosses the boundary.** `default_signer()` and
   `with_default_key()` must not be exported. Any witness or receipt signing
   path takes a host-supplied HMAC key. Exporting a default-key signer would let
   a caller manufacture audit records that verify.
2. **No `getrandom`, no `rand`, no host clock.** Their appearance in the
   dependency tree means someone reached for a crypto path that does not belong
   here, and is a blocking review failure rather than a dependency bump.
3. **Rejection stays specific.** Each `UriError` variant reaches JavaScript as
   itself. Collapsing them into one generic error would erase the distinction
   the strict parser exists to make.
4. **Round-trip identity is tested**: parse → format → byte-identical input,
   across the rejection corpus as well as the accepted one.
5. **Cross-implementation determinism is tested.** Witness and receipt bytes,
   and `record_to_digest`, produced under wasm must be byte-identical to native
   Rust for the same inputs. This is what stops a policy scope and a cache from
   disagreeing about whether two URIs are the same URI.
6. **The cross-tenant rejection test puts the violating segment last.** A
   containment check inside a short-circuiting loop is green for position 1
   while broken for positions 2..n; a negative test that only probes the first
   segment proves nothing about the rest.
7. **The README states what the package is not** — a simulator bound to
   host-provisioned scopes, not an oracle about another authority — in its first
   paragraph.

## Affected Repos

- **ruvnet/rvm**: new `crates/rvm-context-wasm`, npm package
  `@ruvnet/rvm-context-wasm`. Implementation and issue #45 live here.
- **ruvnet/ruvector**: docs only. This ADR records the decision for the program.
- Any TypeScript consumer, including the Meta LLM gateway: consumes the npm
  package. No repo-side change required beyond a dependency.

## Alternatives Considered

**Change `rvm-context` to `crate-type = ["rlib", "cdylib"]` directly.** Rejected.
It would put `wasm-bindgen` and `std` into a `no_std` kernel crate meant to
build for bare-metal AArch64. The wrapper isolates the binding's dependencies
from the crate that must stay austere.

**Export a standalone `authorize()` without the runtime.** Not available, and
the reason is worth recording: `authorize` is `pub(crate)` and
`AuthorizedRequest` construction is private precisely so that an allow decision
cannot exist without a witness record. Working around that would mean
reimplementing the decision, which is the divergence this ADR exists to prevent.

**Make capability handles portable — sign them into bearer tokens.** A real
design, and a much larger one: it changes the trust model from live-table
resolution to token verification, needs key distribution and revocation, and
belongs in its own ADR against `rvm-cap`. Rejected as out of scope here rather
than as a bad idea.

**N-API via napi-rs instead of wasm.** Viable, and better if native performance
ever matters. Rejected for now because it needs per-platform prebuilt binaries
and a matrix build, where wasm ships one artifact everywhere. Revisit if
profiling shows the boundary is a real cost; nothing here forecloses it.

**Reimplement the URI grammar in TypeScript.** Rejected outright. Two
implementations of a canonical form is exactly the divergence the "one spelling
per name" rule exists to prevent — the failure would be silent and would surface
as a policy scope and a cache disagreeing about whether two URIs are the same
URI.

**Wait for crates.io and let consumers build from source.** Rejected as a
solution to the wrong problem. Publication makes the crate reachable from
*Rust*; it does nothing for a TypeScript consumer, because the obstacle is the
crate type, not the distribution.
