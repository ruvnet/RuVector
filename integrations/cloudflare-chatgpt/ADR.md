# ADR: Tenant scoped RuVector MCP at the Cloudflare edge

Status: Implemented for review, live OAuth and target account gates pending

## Decision

Run a stateless MCP Worker with a versioned MCP Apps UI resource. Verify an external OAuth JWT and map its issuer and subject to explicit D1 tenant memberships. Use real RVF and mincut WASM for bounded synchronous operations. Keep heavier learning, Darwin evaluation, and inference behind signed internal service bindings. Do not expose a capability until its binding and acceptance suite are ready.

## Invariants

1. Every customer data query binds `tenant_id` from an authorized membership. A tool argument only selects among already authorized memberships.
2. A write needs an editor or admin role and `ruvector.write`. Every request needs `ruvector.read`.
3. Raw bearer tokens never enter D1, tool results, the UI, or internal service envelopes.
4. WASM state is request local for RVF search. Graph instances are freed after use.
5. Service bindings accept signed tenant context and may only propose or evaluate. Promotion requires a separate human reviewed gate.
6. Disabled capabilities remain visible as pending status and have no callable tool.

## Tradeoffs

| Choice | Benefit | Cost or limit |
| --- | --- | --- |
| Exact RVF WASM scan | Correct ranking within the bounded collection and small deployable binary | Linear work; 500 records per query; no HNSW claim |
| Shared D1 with composite tenant keys | Simple pilot operations and atomic quota accounting | Requires query discipline; US jurisdiction is not suitable for every tenant |
| External OAuth issuer | Reuses Cognitum identity when compatible | Issuer metadata, scope and membership provisioning are deployment dependencies |
| Internal service bindings | Clear resource and deployment isolation for Darwin, Autogenous and ruvLLM | Each service needs its own contract and acceptance test |

The D1 pilot is provisioned in the Cognitum Cloudflare account. The maintainer's staging deploy path uses a different account without access to that database. A production target decision must precede deployment: either use a credential scoped to this D1 account, or adapt storage to the SQLite Durable Object contract used by the separately claimed RuVector edge backend. The latter contract and tenant derivation are still being drafted. This adapter does not claim ownership of the backend's Rust Worker or tenant collection namespace.

## Threat model and gates

Assets include tenant vectors, membership records, usage limits, and downstream orchestration. Entry points are the public MCP endpoint, embedded UI, OAuth token, and internal binding response. A malicious client can submit another tenant ID, a malformed vector, a forged token, a large graph, or a prompt intended to trigger expensive service work. The Worker verifies signatures and scopes, checks membership before queries or service calls, bounds sizes and quotas, and uses fixed SQL and binding destinations. Residual risks are issuer misconfiguration, service replay handling, per tenant D1 data residency, and the lack of a live ChatGPT acceptance run. The deployment gate requires cross tenant negative tests and downstream signature verification.
