//! Error contract for scoped context retrieval.

/// Result returned by the scoped context index.
pub type Result<T> = core::result::Result<T, ContextIndexError>;

/// Fail-closed errors from scope validation, persistence, or retrieval.
#[derive(Debug, thiserror::Error)]
pub enum ContextIndexError {
    /// A namespace or path component violates the canonical scope contract.
    #[error("invalid context scope: {0}")]
    InvalidScope(&'static str),
    /// A point identifier is empty, too large, or contains unsafe bytes.
    #[error("invalid context point identifier")]
    InvalidPointId,
    /// An embedding contains NaN or infinity.
    #[error("context vectors must contain only finite values")]
    NonFiniteVector,
    /// A configured resource limit is zero or otherwise inconsistent.
    #[error("invalid context index configuration: {0}")]
    InvalidConfiguration(&'static str),
    /// A query requested too many results.
    #[error("requested {requested} results, maximum is {maximum}")]
    ResultLimit {
        /// Requested top K value.
        requested: usize,
        /// Configured maximum top K value.
        maximum: usize,
    },
    /// A search would cross too many authorized descendant shards.
    #[error("scope fanout {actual} exceeds maximum {maximum}")]
    ScopeFanout {
        /// Number of matching shards.
        actual: usize,
        /// Configured maximum shard fanout.
        maximum: usize,
    },
    /// Creating another exact-scope shard would exceed the configured bound.
    #[error("context scope capacity {maximum} exhausted")]
    ScopeCapacity {
        /// Configured maximum shard count.
        maximum: usize,
    },
    /// An immutable point ID was reused with different vector bytes.
    #[error("context point ID already exists with different bytes")]
    ImmutableConflict,
    /// A discovered shard lacks a valid, hash-bound scope manifest.
    ///
    /// Reports the shard filename only; the absolute path is never disclosed.
    #[error("corrupt context shard: {0}")]
    CorruptShard(String),
    /// A file already occupies the deterministic path of a scope being created.
    ///
    /// The vector engine would adopt that file's stored vectors and relabel
    /// them as the new scope's data, so creation fails and the file is left
    /// untouched for quarantine at the next open.
    #[error("refusing to adopt pre-existing context shard: {0}")]
    ShardAdoption(String),
    /// Another live handle already holds the exclusive lock on this index root.
    #[error("context index root is already locked by another handle")]
    RootLocked,
    /// A shard path is not a regular file that this index alone names.
    ///
    /// Raised for a symlink, a directory, or a hard-link alias to a file
    /// elsewhere on the filesystem. The vector engine opens read-write and
    /// initialises whatever a path resolves to, so such a path is never handed
    /// to it: aliasing a foreign file into a shard name would otherwise
    /// overwrite that file.
    #[error("context shard {0} is not an exclusively named regular file")]
    AliasedShardFile(String),
    /// The index root is reachable by users other than the one running it.
    ///
    /// A root that other users can write is the precondition for every
    /// name-substitution attack this crate defends against, and a root they
    /// can read exposes every tenant's vectors without any attack at all.
    /// Opening fails rather than running with that exposure. On macOS, any
    /// extended root ACL is rejected because its grants bypass mode bits.
    /// Unix only: no mode enforcement is possible elsewhere.
    #[error("context index root is not private to this user: {0}")]
    InsecureRoot(String),
    /// The root lock path is not a regular file, for example a symlink.
    ///
    /// Following it would let whoever planted it choose which file this index
    /// opens for writing, and which lock the single-handle guarantee rests on.
    #[error("context index root lock is not a regular file")]
    UnsafeRootLock,
    /// This scope's shard name is occupied by a quarantined file.
    ///
    /// Reported instead of "no such scope" so a corrupted or planted file is
    /// never mistaken for an absent tenant. Clear it with
    /// `ScopedContextIndex::discard_quarantined`.
    #[error("context scope shard {0} is quarantined")]
    ScopeQuarantined(String),
    /// A quarantined file turned out to be a valid shard for the named scope.
    ///
    /// Raised when the file changed between open time and the discard, so a
    /// live shard is never deleted by a remediation call.
    #[error("quarantined file {0} is a valid shard for this scope")]
    ShardNotDiscardable(String),
    /// A shared index lock was poisoned by a panicking caller.
    #[error("context index lock poisoned")]
    LockPoisoned,
    /// Filesystem persistence failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Scope manifest encoding or decoding failed.
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
    /// The underlying vector engine refused an operation.
    #[error(transparent)]
    Vector(#[from] ruvector_core::RuvectorError),
}
