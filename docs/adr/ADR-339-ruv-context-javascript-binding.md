# ADR-339: A WebAssembly Binding for `ruv://` Context, and What It May Not Carry

**Status:** Accepted

**Date:** 2026-08-23
**Owners:** RVM maintainers

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

Three related facts, each verified rather than assumed, because each was
plausible enough to guess wrong about:

- **`crates/rvm-wasm` is not a binding.** It is a WebAssembly *guest runtime*
  that lets RVM partitions host wasm modules as an alternative to native
  guests. It faces the hypervisor, not JavaScript.
- **`ed25519-dalek` is a dev-dependency**, not a runtime one. The runtime crypto
  surface is `sha2` and `sha3` — both pure Rust and clean on `wasm32`.
- **A `std` feature already exists.** The crate is `no_std` *by default*, not
  `no_std`-only. That matters because `wasm-bindgen` requires `std`.

Two distribution facts as of this decision: `rvm-context` is not yet on
crates.io (the workspace's foundation crates published first and it is behind
them in the rate-limited queue), and there is no npm package. The name `rvm` on
npm belongs to an unrelated project — "Ruff Version Manager", maintainer
`vilic`, from `ruffjs/rvm` — so any JavaScript distribution here is a **new
scoped package**, never an update to that one.

## Decision

Add `crates/rvm-context-wasm`: a `cdylib` wrapper that enables
`rvm-context/std`, binds a deliberately narrow surface with `wasm-bindgen`, and
publishes to npm as `@ruvnet/rvm-context`. The wrapper is `publish = false` for
crates.io — it is a distribution artifact for a different registry, not a
library another Rust crate should depend on.

**What it exposes: naming and validation.** Canonical `ruv://` parsing into its
structured components, canonical re-formatting, and the specific `UriError`
variant on rejection rather than a generic failure. This is pure computation
over a string, with no I/O, no clock, and no state.

**What it deliberately does not expose: authorization.** The runtime and
resolver paths need an authenticated `PartitionId` bound at construction and a
runtime-owned `ContextClock`. Those exist precisely so a caller cannot supply
its own actor or timestamp. Projecting them into JavaScript would mean either
inventing a JS-side actor — which is the forgery the design prevents — or
shipping an API that looks like authorization and is not. Neither is
acceptable, so the binding stops at the boundary and says so in its README.

The consequence worth stating plainly: **a JavaScript consumer gets the naming
layer, not the trust layer.** That is not a limitation to be lifted later by
adding more bindings; it is the same separation the namespace is built on,
holding at one more boundary. A URI parsed in TypeScript grants exactly as much
as a URI parsed anywhere else, which is nothing.

## Consequences

### Positive

- The gateway, and any TypeScript consumer, can construct and validate
  canonical `ruv://` names without a Rust toolchain or a running RVM.
- One parser, one canonical spelling. A URI validated in TypeScript and the
  same URI validated in the kernel agree by construction, because they are the
  same code compiled twice — not two implementations kept in sync by review.
- `no_std` + `forbid(unsafe_code)` + no async is close to ideal for
  `wasm32-unknown-unknown`. The constraint that made this crate awkward to
  reach from JS is the same one that makes it small and clean once wrapped.

### Negative

- A second artifact to keep in step with the crate. A published npm package can
  drift from a published crate; the version is pinned to the workspace version
  to make drift visible rather than silent.
- The binding invites the assumption it is the whole API. The README and this
  ADR are the mitigation, and they are weaker than a compiler.
- `rvm-context-service` — the durable half — remains out of reach for any
  consumer, and correctly so. It is a deployment, not a library, and it depends
  on `ruvector-context` and `ruvector-core` by path into the submodule, neither
  of which is published.

## Security / Validation Gates

1. **No authorization surface may cross the wasm boundary.** If a future change
   binds anything from `runtime.rs` or `resolver.rs`, it must carry an
   authenticated actor from the Rust side; a JS-supplied actor or timestamp is
   a blocking review failure.
2. **Rejection must stay specific.** Each `UriError` variant reaches JavaScript
   as itself. A binding that collapses them into one generic error would remove
   the distinction the strict parser exists to make.
3. **Round-trip identity is tested**: parse → format → byte-identical input,
   across the rejection corpus as well as the accepted one.
4. **The package README states what the package is not** — the naming layer,
   not the trust layer — in its first paragraph, not a footnote.

## Affected Repos

- **ruvnet/rvm**: new `crates/rvm-context-wasm`, npm package
  `@ruvnet/rvm-context`. Implementation lands here.
- **ruvnet/ruvector**: docs only. This ADR records the decision for the program.
- Any TypeScript consumer, including the Meta LLM gateway: consumes the npm
  package. No repo-side change required of them beyond a dependency.

## Alternatives Considered

**Change `rvm-context` to `crate-type = ["rlib", "cdylib"]` directly.** Rejected.
It would put `wasm-bindgen` and `std` into a `no_std` kernel crate that is meant
to build for bare-metal AArch64. The wrapper isolates the binding's
dependencies from the crate that must stay austere.

**N-API via napi-rs instead of wasm.** Viable, and better if native performance
ever matters. Rejected for now because it needs per-platform prebuilt binaries
and a matrix build, where wasm ships one artifact everywhere. Revisit if
profiling shows the wasm boundary is a real cost; nothing about this decision
forecloses it.

**Reimplement the URI grammar in TypeScript.** Rejected outright. Two
implementations of a canonical form is exactly the divergence the "one spelling
per name" rule exists to prevent — the failure would be silent and would show up
as a policy scope and a cache disagreeing about whether two URIs are the same
URI.

**Wait for crates.io and let consumers build from source.** Rejected as a
solution to the wrong problem. Publication makes the crate reachable from
*Rust*; it does nothing for a TypeScript consumer, because the obstacle is the
crate type, not the distribution.
