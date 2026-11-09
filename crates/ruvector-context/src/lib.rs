//! Physically isolated vector retrieval for governed context namespaces.
//!
//! Metadata filtering after approximate-nearest-neighbor traversal is not a
//! tenant boundary. This crate places each exact context scope in a distinct
//! vector index and selects authorized descendant shards before any ANN call.
//!
//! # This crate is not the tenant boundary
//!
//! [`ScopedContextIndex`] enforces *physical* isolation between scopes; it does
//! not authenticate or authorize anyone. Every method — [`ScopedContextIndex::insert`],
//! [`ScopedContextIndex::search`], [`ScopedContextIndex::delete_point`],
//! [`ScopedContextIndex::erase_scope`], and [`ScopedContextIndex::scope_stats`]
//! — treats the [`ContextScope`] it is handed as already authorized for the
//! requesting principal. Deciding which principal may name which scope is the
//! caller's responsibility, and it must happen before the call. In particular
//! [`ScopedContextIndex::scope_stats`] is an existence oracle that reports
//! exact vector counts, so it needs the same authorization as a read.
//!
//! The scope manifest stored in each shard binds a scope to a *filename*, not
//! to shard contents; there is no message authentication code over the stored
//! vectors.
//!
//! # The root directory is a private precondition
//!
//! The root must be private to the uid running the index.
//! [`ScopedContextIndex::open`] creates it `0700`, creates shard files and the
//! lock `0600`, and refuses a root that is group- or other-accessible with
//! [`ContextIndexError::InsecureRoot`]. On macOS it also refuses any extended
//! ACL on the root, because ACL grants are not reflected in the mode bits.
//!
//! This is the boundary the crate actually rests on. Every name under the root
//! — shard files, the lock, the staging directory — is re-resolved by the
//! kernel on each syscall, and this crate holds none of them by descriptor, so
//! a user who can write the root can substitute any of them between one
//! syscall and the next. No in-process check closes that; taking away their
//! write access does. The staging directory, the post-publication inode
//! identity check, the lone-regular-file requirement, and the reserved-name
//! sweep are retained as defence in depth against operator error, not as the
//! boundary itself.
//!
//! An attacker running as the *same* uid is conceded: they can read and
//! rewrite shard files directly, and no permission bit or path check helps.
//!
//! # Lock portability
//!
//! The exclusive root lock is `flock`-based, which is per open file
//! description: two handles in one process conflict, which is the case it
//! exists to catch. Two platforms degrade that to per-process locking, where a
//! second handle in the same process would silently succeed:
//!
//! - Solaris and illumos, where the backing crate falls back to `fcntl`.
//! - Linux over NFS, where `flock` is emulated with POSIX record locks.
//!
//! Do not rely on the single-handle guarantee there.
//!
//! None of the mode enforcement above applies off Unix either: the root, the
//! shards, and the lock are created with whatever the platform defaults to,
//! and the privacy check is skipped. The `O_NOFOLLOW` that keeps the lock path
//! from being redirected through a symlink is Unix-only too, so on Windows a
//! symlink planted at the lock name is unmitigated.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod index;
mod layout;
mod options;
mod quarantine;
mod scope;
mod shard;

pub use error::{ContextIndexError, Result};
pub use index::{ContextMatch, ContextPoint, ScopeStats, ScopedContextIndex};
pub use options::{ContextIndexOptions, MAX_RESULT_LIMIT};
pub use quarantine::QuarantinedShard;
pub use scope::{ContextNamespace, ContextScope};
