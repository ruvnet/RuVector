# Security boundaries

This adapter provides process-local retrieval over native RuVector and an
optional LionAGI memory interface. Host applications supply embedding code and
own authorization, credentials, network policy, tenant isolation and logging.
Use a distinct store per authorization scope. Metadata predicates are search
filters, not a substitute for access control.

## Input and resource controls

Vectors must match the configured dimension, contain finite real coordinates
within the documented magnitude limit and have nonzero cosine norm. Booleans
are rejected. The whole insertion batch is validated before native insertion.
Metadata is copied through bounded JSON serialization; unsupported objects,
nonfinite numbers, excessive nesting and cycles are rejected.

Limits cover record count, batch size, content and metadata bytes, aggregate
payload, nominal vector allocation, result count and serialized tool output.
The vector budget is not a total process RSS guarantee: HNSW graph data, Python
objects, temporary arrays and the native allocator consume additional memory.
Deleting records rebuilds the index and can temporarily increase memory usage.
Set process or container limits for deployments requiring a hard memory cap.

The search tool binds one store and exposes only query and result count. It
exposes no filesystem path, provider credential, store selector or mutation
operation. Its output marks retrieved text as untrusted data. This marker and
the tool description do not prevent an LLM from following injected instructions;
the host must enforce tool permissions independently of retrieved text.

## Failure and concurrency behavior

Store operations are serialized. A native insertion exception poisons the
store to prevent reads against potentially inconsistent native and Python
state. Recreate the store after such an exception. Deletion builds a replacement
index before swapping it in; a failed rebuild leaves the original available.

Async methods dispatch blocking native work to a worker thread. Cancelling an
await does not roll back an operation already executing in that thread. Use
stable record IDs and check committed state before retrying. Cancellation while
an embedder is still running occurs before insertion. The host supplied
embedder is trusted executable code and may perform network I/O; the adapter
neither restricts nor audits its behavior.

## Assurance and exclusions

The acceptance tests include metadata cycles, malformed numeric inputs,
resource boundaries, copied data, fault injection and a real LionAGI tool call.
The dependency audit and Ruflo scan reported zero findings on 2026-10-11;
see [validation evidence](VALIDATION.md) for exact scope. Neither scan proves
native memory safety, absence of new advisory vulnerabilities or resistance to
all prompt injection. No fuzzing campaign, penetration test or independent
third-party audit has been completed for this adapter.

There is no persistence, remote service, authentication layer or encrypted
storage supplied here. Avoid storing secrets unnecessarily. Report suspected
vulnerabilities privately using the parent repository's security reporting
process rather than publishing exploit details in a public issue.
