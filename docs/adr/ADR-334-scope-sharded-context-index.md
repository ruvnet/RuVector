# ADR-334: Scope-Sharded Context Vector Index

**Status:** Proposed

**Date:** 2026-08-22
**Owners:** RuVector maintainers

## Context

RVM now defines capability-governed `ruv://` names for resources, memories,
and skills. Its authorization boundary runs before a retrieval backend is
called. RuVector must preserve that boundary once the authorized request
reaches vector retrieval.

The existing `VectorDB::search` traverses one ANN graph and applies optional
metadata filters after candidates are returned. That behavior is useful for
ordinary filtering but cannot provide tenant isolation. Traversing a shared
graph can make latency, result quality, and resource use depend on objects the
caller was not authorized to enumerate.

## Decision

Add `ruvector-context`, a small persistent crate that physically separates
every exact structured context scope into its own `VectorDB` shard.

A scope contains exact authority, tenant, subject kind, subject ID,
collection, and path segments. Scope components are validated as a strict
ASCII contract. The on-disk filename is SHA-256 over a domain-separated,
length-delimited encoding; tenant and path names never enter filesystem paths.

Search accepts an already authorized root scope. It performs an exact
namespace lookup, selects only lexicographic descendant path shards, checks a
fanout ceiling, and only then invokes ANN search. Results from selected shards
are merged into one globally ranked bounded top K.

Each shard stores its complete scope manifest inside the same persistent
vector database. Recovery verifies that the manifest hashes to the shard
filename before the shard becomes queryable. Missing, malformed, renamed, or
configuration-incompatible shards fail closed.

RVM remains responsible for authenticating actors and capabilities. RuVector
does not parse capability handles or infer authority from a URI.

## Invariants

1. A search for namespace A never calls an ANN index belonging to namespace B.
2. A path-prefix search calls only exact shards at or below that structured
   prefix; text-prefix comparison is forbidden.
3. Fanout and top K limits are validated before the first ANN call.
4. Point IDs are immutable. An identical replay is idempotent; different
   vector bytes under the same ID are rejected.
5. Scope strings never become filesystem path components.
6. Recovery authenticates the scope-to-filename binding before serving data.
7. Exact-scope erasure closes and removes the persistent shard while holding
   the catalog write lock. The unlink is ordered before the catalog mutation,
   so a failed erasure never reports in-memory success over a surviving file.
8. Shard creation never adopts a pre-existing file. A file already present at
   the computed filename is refused rather than opened, because
   `VectorDB::new` on an existing path silently inherits that file's stored
   config and stored vectors.
9. One process holds at most one index handle per root directory, enforced by
   an exclusive advisory lock taken in `open`. Two handles over one root would
   otherwise share the process-global database pool while maintaining
   independent in-memory indexes.
10. The root directory is private to the index's uid: created `0700`, shards
    and the lock created `0600`, and `open` refuses a group- or
    other-accessible root. Mode enforcement is unix-only.

### The root directory is a private precondition, enforced

**The index root must be private to the uid running the index.** It is created
`0700`, shard files and the lock are created `0600`, and `open` refuses a root
that is group- or other-accessible. This is a precondition the crate checks,
not an assumption it documents and hopes for.

That enforcement is load-bearing, and the reason is worth recording because
four rounds of security review converged on it the hard way. Every earlier
attempt defended a **name** in the root: refuse an existing file at the shard
name; then refuse a symlink at the shard name; then build under a scratch name
and publish atomically; then move the scratch name inside a private staging
directory. Each round the review swapped the *next* name — and the staging
directory itself is a name in the root, so it was swappable too. There is no
fixed point, because every path component under the root is re-resolved on
every syscall and this crate holds none of them by descriptor.

Two things actually terminate that regress. The complete answer is
descriptor-anchored I/O: hold file descriptors on the root and staging and do
all work with `openat`/`mkdirat`-style relative syscalls, so a swapped name
cannot redirect anything. That is blocked here by the vector engine's
path-based API, which would need a descriptor- or directory-relative open —
an engine-level change, not a change to this crate, and the long-term fix.
The answer taken instead is to remove the hostile directory from the threat
model altogether by making the root private and refusing to run when it is
not.

Everything else the shard layer does — the private staging directory, the
inode identity check after publication, the lone-regular-file requirement
before the engine sees a path, the reserved-name sweep — is retained as
**defence in depth against operator error**, which is the honest description
of its role. None of it is what stands between a tenant and an attacker.

An attacker running as the **same uid** remains conceded and always did: they
can read and rewrite the shard files directly, and no permission bit or path
check helps. Detecting that requires authenticating shard contents, which
brings us to the next section.

### What the manifest binding does and does not prove

Invariant 6 authenticates the scope-to-**filename** binding. It does not
authenticate shard **contents**: there is no MAC over the stored vectors, and
`shard_id` is an unkeyed digest over non-secret identifiers, so a party that
can write the root directory can compute a victim's filename offline and
author a well-formed shard for it. Invariant 8 is what closes the resulting
laundering path — without it, a planted file adopted through the create path
would have its manifest rewritten to the victim scope and would then pass
invariant 6 forever after.

Under invariant 10 the only principal who can write the root is the index's
own uid, so this is the same-uid adversary already conceded above rather than
an open door for anyone on the host. It is nonetheless the residual worth
naming: a same-uid process can author a valid shard for any scope and this
layer will serve it. Closing that needs content authentication — a MAC over
stored vectors, and `shard_id` keyed with a per-deployment secret (HMAC rather
than bare SHA-256) so filenames are unguessable as well as unforgeable. Both
are out of scope here.

### Explicit non-goals

- **Availability is not isolated.** `max_scopes` is a process-global ceiling
  with no per-namespace quota, so one tenant exhausting it makes another
  tenant's first insert fail. Per-tenant quotas are deferred.
- **Startup cost is proportional to the whole corpus.** `open` materializes
  every shard before the first query rather than opening shards lazily, so
  memory and startup scale with total stored vectors across all tenants,
  bounded only by `max_scopes`.
- **No authorization is performed here.** Every public method, not only
  `search`, assumes the caller has already authorized the scope. `scope_stats`
  in particular answers existence and exact vector count for any nameable
  scope.

## Pseudocode

```text
search(authorized_root, vector, k):
    validate vector dimensions and k
    namespace = catalog.exact_lookup(authorized_root.namespace)
    shards = namespace.lexicographic_descendants(authorized_root.path)
    require shards.count <= max_search_shards
    for shard in shards:
        candidates = shard.ann_search(vector, k)
        bounded_global_top_k.merge(candidates)
    return bounded_global_top_k.sorted()
```

Failure case: if two descendant shards exceed a configured fanout of one, the
method returns `ScopeFanout` and every shard search counter remains zero.

## Alternatives

### Filter a shared HNSW result set

Rejected. It traverses unauthorized candidates before filtering and can lose
authorized results when the initial K is dominated by another tenant.

### Prefix point IDs inside one index

Rejected. Identifier prefixes do not constrain ANN graph traversal.

### One index per tenant only

Rejected. A caller authorized for one subject or path could still influence
and observe traversal over sibling scopes in the same tenant.

### Let RuVector validate RVM capabilities

Rejected. It would duplicate kernel policy and couple the vector engine to one
runtime. The typed RVM adapter authorizes first; this crate preserves physical
retrieval isolation afterward.

## Security and Operations

The configured root directory is trusted deployment state. Untrusted scope
components are validated and represented only by a digest on disk. Resource
ceilings bound scopes, descendant fanout, and returned results. Non-finite
vectors are rejected at ingress.

The reference persistence is process-local REDB. Multi-host replication,
distributed admission control, encrypted disks, and KMS-backed erasure remain
deployment responsibilities. RVM separately owns immutable RVF objects,
linearizable aliases, signed receipt draining, and execution capabilities.

## Acceptance Evidence

The upstream crate must demonstrate:

1. Cross-tenant search returns only the authorized tenant and leaves every
   other tenant shard search counter at zero.
2. Segment-aware path selection excludes textual siblings such as `a` and
   `alpha`.
3. Restart reconstructs scope manifests and vectors.
4. Fanout refusal occurs before any ANN call.
5. Immutable replay, conflict rejection, and exact-scope erasure behave
   deterministically.
6. Formatting, unit tests, Clippy, dependency audit, and changed-file secret
   scan pass before merge.
7. The isolation suite runs against **both** index kinds. A suite configured
   with `hnsw_config: None` routes every shard to `FlatIndex` and therefore
   cannot establish claim 1, whose subject is the ANN index.
8. The per-shard search counter is incremented on a path that cannot be
   bypassed by construction, rather than by a call site that a future edit
   could omit while leaving the suite green.
9. A shard file planted at another scope's computed filename while the index
   is open is refused, and no restart launders it into the victim scope.
10. A second index handle on an already-open root fails loudly.
11. A single unreadable shard-shaped file does not deny `open` to unrelated
    tenants.
