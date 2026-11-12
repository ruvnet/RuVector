//! Integration tests for persistent, scope-isolated context indexes.
//!
//! Every behavioural test runs against both vector engines. Isolation claims
//! about approximate-nearest-neighbor traversal are meaningless when the
//! process contains no ANN index at all, so `hnsw_config: None` alone cannot
//! establish them.

use ruvector_context::{
    ContextIndexError, ContextIndexOptions, ContextNamespace, ContextPoint, ContextScope,
    ScopedContextIndex,
};
use ruvector_core::types::{DbOptions, HnswConfig};
use ruvector_core::DistanceMetric;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
enum Engine {
    Flat,
    Hnsw,
}

impl Engine {
    fn hnsw_config(self) -> Option<HnswConfig> {
        match self {
            Self::Flat => None,
            Self::Hnsw => Some(HnswConfig {
                max_elements: 1_024,
                ..HnswConfig::default()
            }),
        }
    }
}

/// Generate one `flat` and one `hnsw` test per declared body.
macro_rules! engine_tests {
    ($(fn $name:ident($engine:ident: Engine) $body:block)*) => {
        $(
            mod $name {
                use super::*;

                fn run($engine: Engine) $body

                #[test]
                fn flat() {
                    run(Engine::Flat);
                }

                #[test]
                fn hnsw() {
                    run(Engine::Hnsw);
                }
            }
        )*
    };
}

/// A private index root.
///
/// `tempfile::tempdir()` creates the directory with the ambient umask applied
/// — `0755` under the common `022` — and `open` refuses a root other users can
/// reach, so tests cannot use one directly. This creates a private directory
/// inside it instead. Tests that need the root not to exist yet, so that
/// `open` creates it, build their own path.
struct IndexRoot {
    _base: tempfile::TempDir,
    path: PathBuf,
}

impl IndexRoot {
    fn path(&self) -> &Path {
        &self.path
    }
}

fn index_root() -> IndexRoot {
    let base = tempfile::tempdir().unwrap();
    let path = base.path().join("index");
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        builder.mode(0o700);
    }
    builder.create(&path).unwrap();
    IndexRoot { _base: base, path }
}

fn namespace(tenant: &str) -> ContextNamespace {
    ContextNamespace::new("context.example", tenant, "agent", "reader", "memory").unwrap()
}

fn scope(tenant: &str, path: &[&str]) -> ContextScope {
    ContextScope::new(
        namespace(tenant),
        path.iter().map(|value| (*value).to_string()).collect(),
    )
    .unwrap()
}

fn options(engine: Engine, max_search_shards: usize) -> ContextIndexOptions {
    ContextIndexOptions {
        vector: DbOptions {
            dimensions: 3,
            distance_metric: DistanceMetric::Euclidean,
            storage_path: String::new(),
            hnsw_config: engine.hnsw_config(),
            quantization: None,
        },
        max_scopes: 16,
        max_search_shards,
        max_results: 8,
    }
}

fn point(id: &str, vector: [f32; 3]) -> ContextPoint {
    ContextPoint {
        id: id.to_string(),
        vector: vector.to_vec(),
    }
}

/// Shard filenames are opaque hashes, so tests recover them from the directory
/// rather than recomputing the private hash construction.
fn shard_files(root: &Path) -> Vec<PathBuf> {
    let mut found = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "redb")
        })
        .collect::<Vec<_>>();
    found.sort();
    found
}

/// The opaque filename `target` hashes to, learned from a throwaway root.
fn shard_filename_of(engine: Engine, root: &Path, target: &ContextScope) -> String {
    isolated_shard(engine, root, target, "probe")
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
}

/// Build a shard for `target` in a throwaway root and return its file path.
fn isolated_shard(engine: Engine, root: &Path, target: &ContextScope, id: &str) -> PathBuf {
    let index = ScopedContextIndex::open(root, options(engine, 8)).unwrap();
    index.insert(target, point(id, [1.0, 0.0, 0.0])).unwrap();
    drop(index);
    let files = shard_files(root);
    assert_eq!(files.len(), 1);
    files.into_iter().next().unwrap()
}

engine_tests! {
    fn cross_tenant_search_never_touches_other_tenant_index(engine: Engine) {
        let temp = index_root();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        let acme = scope("acme", &["project"]);
        let other = scope("other", &["project"]);
        index
            .insert(&acme, point("acme:one", [1.0, 0.0, 0.0]))
            .unwrap();
        index
            .insert(&other, point("other:one", [1.0, 0.0, 0.0]))
            .unwrap();

        let matches = index.search(&acme, &[1.0, 0.0, 0.0], 4).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, "acme:one");
        assert_eq!(index.scope_stats(&acme).unwrap().unwrap().searches, 1);
        assert_eq!(index.scope_stats(&other).unwrap().unwrap().searches, 0);
    }

    fn path_prefix_selects_descendants_without_touching_siblings(engine: Engine) {
        let temp = index_root();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        let root = scope("acme", &["project"]);
        let child = scope("acme", &["project", "doc"]);
        let sibling = scope("acme", &["project-archive"]);
        index
            .insert(&child, point("child", [0.0, 1.0, 0.0]))
            .unwrap();
        index
            .insert(&sibling, point("sibling", [0.0, 1.0, 0.0]))
            .unwrap();

        let matches = index.search(&root, &[0.0, 1.0, 0.0], 4).unwrap();
        assert_eq!(
            matches
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            vec!["child"]
        );
        assert_eq!(index.scope_stats(&child).unwrap().unwrap().searches, 1);
        assert_eq!(index.scope_stats(&sibling).unwrap().unwrap().searches, 0);
    }

    fn restart_recovers_hash_bound_scope_and_vectors(engine: Engine) {
        let temp = index_root();
        let target = scope("acme", &["durable"]);
        {
            let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
            index
                .insert(&target, point("revision:view", [0.0, 0.0, 1.0]))
                .unwrap();
        }
        let recovered = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        let matches = recovered.search(&target, &[0.0, 0.0, 1.0], 1).unwrap();
        assert_eq!(matches[0].id, "revision:view");
    }

    fn immutable_replay_is_idempotent_but_conflict_is_refused(engine: Engine) {
        let temp = index_root();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        let target = scope("acme", &["immutable"]);
        index
            .insert(&target, point("same", [1.0, 0.0, 0.0]))
            .unwrap();
        index
            .insert(&target, point("same", [1.0, 0.0, 0.0]))
            .unwrap();
        assert!(matches!(
            index.insert(&target, point("same", [0.0, 1.0, 0.0])),
            Err(ContextIndexError::ImmutableConflict)
        ));
    }

    fn deleted_point_ids_are_reusable_with_different_bytes(engine: Engine) {
        let temp = index_root();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        let target = scope("acme", &["reuse"]);
        index
            .insert(&target, point("id", [1.0, 0.0, 0.0]))
            .unwrap();
        assert!(index.delete_point(&target, "id").unwrap());
        index
            .insert(&target, point("id", [0.0, 1.0, 0.0]))
            .unwrap();
        assert_eq!(index.scope_stats(&target).unwrap().unwrap().vectors, 1);
    }

    fn fanout_failure_occurs_before_any_shard_search(engine: Engine) {
        let temp = index_root();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 1)).unwrap();
        let root = ContextScope::root(namespace("acme"));
        let one = scope("acme", &["one"]);
        let two = scope("acme", &["two"]);
        index.insert(&one, point("one", [1.0, 0.0, 0.0])).unwrap();
        index.insert(&two, point("two", [0.0, 1.0, 0.0])).unwrap();
        assert!(matches!(
            index.search(&root, &[1.0, 0.0, 0.0], 1),
            Err(ContextIndexError::ScopeFanout {
                actual: 2,
                maximum: 1
            })
        ));
        assert_eq!(index.scope_stats(&one).unwrap().unwrap().searches, 0);
        assert_eq!(index.scope_stats(&two).unwrap().unwrap().searches, 0);
    }

    fn exact_scope_erasure_removes_persistent_shard(engine: Engine) {
        let temp = index_root();
        let target = scope("acme", &["erase"]);
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        index
            .insert(&target, point("erase-me", [1.0, 0.0, 0.0]))
            .unwrap();
        assert!(index.erase_scope(&target).unwrap());
        assert_eq!(index.scope_count().unwrap(), 0);
        assert!(shard_files(temp.path()).is_empty());
        drop(index);
        assert_eq!(
            ScopedContextIndex::open(temp.path(), options(engine, 8))
                .unwrap()
                .scope_count()
                .unwrap(),
            0
        );
    }

    fn erased_scope_recreation_never_reuses_the_unlinked_database(engine: Engine) {
        let temp = index_root();
        let target = scope("acme", &["erase-recreate"]);
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        index
            .insert(&target, point("old", [1.0, 0.0, 0.0]))
            .unwrap();
        assert!(index.erase_scope(&target).unwrap());
        index
            .insert(&target, point("new", [0.0, 1.0, 0.0]))
            .unwrap();

        let matches = index.search(&target, &[1.0, 0.0, 0.0], 8).unwrap();
        assert_eq!(
            matches
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            vec!["new"]
        );
    }

    fn non_finite_vectors_fail_before_index_access(engine: Engine) {
        let temp = index_root();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        let target = scope("acme", &["finite"]);
        assert!(matches!(
            index.insert(&target, point("nan", [f32::NAN, 0.0, 0.0])),
            Err(ContextIndexError::NonFiniteVector)
        ));
        assert!(matches!(
            index.search(&target, &[f32::INFINITY, 0.0, 0.0], 1),
            Err(ContextIndexError::NonFiniteVector)
        ));
        assert_eq!(index.scope_count().unwrap(), 0);
    }

    fn planted_foreign_shard_is_refused_and_never_laundered(engine: Engine) {
        let attacker = scope("attacker", &["docs"]);
        let victim = scope("victim", &["docs"]);

        // A real, well-formed shard belonging to the attacker's scope.
        let attacker_root = index_root();
        let planted =
            isolated_shard(engine, attacker_root.path(), &attacker, "attacker-planted-row");

        // The victim scope's opaque filename, learned from a throwaway root.
        let probe_root = index_root();
        let victim_file = isolated_shard(engine, probe_root.path(), &victim, "probe");
        let victim_filename = victim_file.file_name().unwrap().to_owned();

        // Plant it while the victim's index is already open, so the binding
        // check performed at open time never sees the file.
        let temp = index_root();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        std::fs::copy(&planted, temp.path().join(&victim_filename)).unwrap();

        let refused = index.insert(&victim, point("victim-own-row", [1.0, 0.0, 0.0]));
        let Err(ContextIndexError::ShardAdoption(reported)) = refused else {
            panic!("create adopted a pre-existing shard: {refused:?}");
        };
        assert_eq!(Some(reported.as_str()), victim_filename.to_str());
        assert!(!reported.contains(std::path::MAIN_SEPARATOR));
        assert!(index.search(&victim, &[1.0, 0.0, 0.0], 8).unwrap().is_empty());
        assert_eq!(index.scope_count().unwrap(), 0);

        // Nothing was laundered: after a restart the planted file is still
        // bound to the attacker's scope and is quarantined, not adopted.
        drop(index);
        let restarted = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        let quarantined = restarted.quarantined_shards().unwrap();
        assert_eq!(quarantined.len(), 1);
        assert_eq!(Some(quarantined[0].filename.as_str()), victim_filename.to_str());
        assert_eq!(restarted.scope_count().unwrap(), 0);
        assert!(restarted
            .search(&victim, &[1.0, 0.0, 0.0], 8)
            .unwrap()
            .is_empty());
    }

    fn second_handle_on_one_root_fails_loudly(engine: Engine) {
        let temp = index_root();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        assert!(matches!(
            ScopedContextIndex::open(temp.path(), options(engine, 8)),
            Err(ContextIndexError::RootLocked)
        ));

        // Dropping the handle releases the advisory lock.
        drop(index);
        ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
    }

    fn lock_file_left_by_a_crashed_process_does_not_brick_the_root(engine: Engine) {
        let temp = index_root();
        // A process that exits without dropping its handle leaves the lock file
        // behind, but the kernel released the advisory lock on exit. The
        // leftover file must be reused, and never mistaken for a shard.
        std::fs::write(temp.path().join(".lock"), b"").unwrap();

        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        assert!(index.quarantined_shards().unwrap().is_empty());
        let target = scope("acme", &["after-crash"]);
        index
            .insert(&target, point("row", [1.0, 0.0, 0.0]))
            .unwrap();
        assert_eq!(index.scope_count().unwrap(), 1);
    }

    fn unloadable_shard_is_quarantined_without_denying_open(engine: Engine) {
        let temp = index_root();
        let good = scope("acme", &["healthy"]);
        {
            let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
            index
                .insert(&good, point("healthy-row", [1.0, 0.0, 0.0]))
                .unwrap();
        }
        let junk_name = format!("{}.redb", "ab".repeat(32));
        std::fs::write(temp.path().join(&junk_name), b"not a vector database").unwrap();

        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();
        let quarantined = index.quarantined_shards().unwrap();
        assert_eq!(quarantined.len(), 1);
        assert_eq!(quarantined[0].filename, junk_name);
        assert!(!quarantined[0].reason.is_empty());
        assert_eq!(index.scope_count().unwrap(), 1);
        let matches = index.search(&good, &[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(matches[0].id, "healthy-row");
    }

    fn quarantined_scope_fails_loudly_and_can_be_discarded(engine: Engine) {
        let target = scope("acme", &["quarantined"]);
        let probe = index_root();
        let filename = shard_filename_of(engine, probe.path(), &target);

        let temp = index_root();
        std::fs::write(temp.path().join(&filename), b"not a vector database").unwrap();
        let index = ScopedContextIndex::open(temp.path(), options(engine, 8)).unwrap();

        let listed = index.quarantined_shards().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].filename, filename);
        assert!(!listed[0].reason.is_empty(), "reason must be diagnosable");

        // Every per-scope operation says "quarantined", never "absent". In
        // particular erase must not report a false negative.
        assert!(matches!(
            index.insert(&target, point("blocked", [1.0, 0.0, 0.0])),
            Err(ContextIndexError::ScopeQuarantined(_))
        ));
        assert!(matches!(
            index.scope_stats(&target),
            Err(ContextIndexError::ScopeQuarantined(_))
        ));
        assert!(matches!(
            index.erase_scope(&target),
            Err(ContextIndexError::ScopeQuarantined(_))
        ));
        assert!(matches!(
            index.delete_point(&target, "blocked"),
            Err(ContextIndexError::ScopeQuarantined(_))
        ));

        assert!(index.discard_quarantined(&target).unwrap());
        assert!(index.quarantined_shards().unwrap().is_empty());
        assert!(!index.discard_quarantined(&target).unwrap());

        index
            .insert(&target, point("fresh", [1.0, 0.0, 0.0]))
            .unwrap();
        let matches = index.search(&target, &[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(matches[0].id, "fresh");
    }

    fn unbounded_max_results_is_refused_instead_of_panicking(engine: Engine) {
        let temp = index_root();
        let mut unbounded = options(engine, 8);
        unbounded.max_results = usize::MAX;
        assert!(matches!(
            ScopedContextIndex::open(temp.path(), unbounded),
            Err(ContextIndexError::InvalidConfiguration(_))
        ));
    }
}

/// A failed unlink must not be reported as an erase.
///
/// Unix only: the failure is produced by revoking write permission on the
/// index root, which a process running as root would bypass.
#[cfg(unix)]
#[test]
fn failed_shard_unlink_does_not_report_a_phantom_erase() {
    use std::os::unix::fs::PermissionsExt;

    let temp = index_root();
    let target = scope("acme", &["erase-fails"]);
    let index = ScopedContextIndex::open(temp.path(), options(Engine::Flat, 8)).unwrap();
    index
        .insert(&target, point("gdpr-erase-me", [1.0, 0.0, 0.0]))
        .unwrap();

    let original = std::fs::metadata(temp.path()).unwrap().permissions();
    let mut readonly = original.clone();
    readonly.set_mode(0o555);
    std::fs::set_permissions(temp.path(), readonly).unwrap();
    let probe = temp.path().join("probe");
    let enforced = std::fs::File::create(&probe).is_err();
    let erased = enforced.then(|| index.erase_scope(&target));
    std::fs::set_permissions(temp.path(), original).unwrap();
    let _ = std::fs::remove_file(&probe);

    let Some(erased) = erased else {
        return; // running as root; the permission bits are not enforced
    };
    assert!(erased.is_err(), "unlink failure reported as success");
    assert_eq!(index.scope_count().unwrap(), 1);
    assert_eq!(index.scope_stats(&target).unwrap().unwrap().vectors, 1);
    let matches = index.search(&target, &[1.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(matches[0].id, "gdpr-erase-me");
    assert_eq!(shard_files(temp.path()).len(), 1);
}

/// A shard built with an HNSW graph must not be adopted by an index that asked
/// for a flat engine. It binds correctly to its own filename, so only a full
/// engine-configuration comparison catches it.
#[test]
fn adopted_shard_cannot_dictate_engine_parameters() {
    let target = scope("acme", &["engine"]);
    let probe = index_root();
    let decoy = isolated_shard(Engine::Hnsw, probe.path(), &target, "probe");
    let filename = decoy.file_name().unwrap().to_str().unwrap().to_string();

    let temp = index_root();
    std::fs::copy(&decoy, temp.path().join(&filename)).unwrap();
    let index = ScopedContextIndex::open(temp.path(), options(Engine::Flat, 8)).unwrap();

    let quarantined = index.quarantined_shards().unwrap();
    assert_eq!(quarantined.len(), 1);
    assert_eq!(quarantined[0].filename, filename);
    assert!(matches!(
        index.scope_stats(&target),
        Err(ContextIndexError::ScopeQuarantined(_))
    ));
}

/// Symlink attacks are Unix-shaped: on Windows creating one needs a privilege
/// the attacker model here does not assume.
#[cfg(unix)]
mod symlink {
    use super::*;

    fn plant(link: &Path, target: &Path) {
        std::os::unix::fs::symlink(target, link).unwrap();
    }

    /// `try_exists` follows symlinks, so a DANGLING one reports "nothing here"
    /// and a stat-then-create would build the victim's shard at the link
    /// target, outside the index root.
    #[test]
    fn dangling_symlink_at_a_shard_path_cannot_redirect_a_tenants_vectors() {
        let victim = scope("victim", &["docs"]);
        let probe = index_root();
        let filename = shard_filename_of(Engine::Flat, probe.path(), &victim);

        let outside = tempfile::tempdir().unwrap();
        let exfiltrated = outside.path().join("exfiltrated.redb");
        let temp = index_root();
        let index = ScopedContextIndex::open(temp.path(), options(Engine::Flat, 8)).unwrap();
        plant(&temp.path().join(&filename), &exfiltrated);

        let refused = index.insert(&victim, point("victim-secret", [1.0, 0.0, 0.0]));
        assert!(
            matches!(refused, Err(ContextIndexError::ShardAdoption(_))),
            "insert walked past the planted symlink: {refused:?}"
        );
        assert!(
            !exfiltrated.exists(),
            "vectors were written outside the index root"
        );
        assert_eq!(index.scope_count().unwrap(), 0);
        assert!(index
            .search(&victim, &[1.0, 0.0, 0.0], 8)
            .unwrap()
            .is_empty());

        // And the phantom erase that followed from it: nothing to erase, and
        // the report says quarantined rather than "no such scope".
        drop(index);
        let restarted = ScopedContextIndex::open(temp.path(), options(Engine::Flat, 8)).unwrap();
        assert!(matches!(
            restarted.erase_scope(&victim),
            Err(ContextIndexError::ScopeQuarantined(_))
        ));
    }

    /// `open()` must not hand a symlinked entry to the engine either: doing so
    /// creates the backing database at the link target before the entry is
    /// ever judged.
    #[test]
    fn open_never_opens_a_symlinked_shard_entry() {
        let victim = scope("victim", &["docs"]);
        let probe = index_root();
        let filename = shard_filename_of(Engine::Flat, probe.path(), &victim);

        let outside = tempfile::tempdir().unwrap();
        let exfiltrated = outside.path().join("exfiltrated.redb");
        let temp = index_root();
        plant(&temp.path().join(&filename), &exfiltrated);

        let index = ScopedContextIndex::open(temp.path(), options(Engine::Flat, 8)).unwrap();
        // Assert the harm before the bookkeeping: quarantining the entry after
        // the engine already materialised a database at the link target is not
        // a fix, it is the exfiltration plus a log line.
        assert!(
            !exfiltrated.exists(),
            "open() created a database through a planted symlink"
        );
        let quarantined = index.quarantined_shards().unwrap();
        assert_eq!(quarantined.len(), 1);
        assert_eq!(quarantined[0].filename, filename);
        assert!(
            quarantined[0].reason.contains("symlink"),
            "reason should name the cause: {}",
            quarantined[0].reason
        );
    }

    #[test]
    fn symlinked_root_lock_is_refused() {
        let outside = tempfile::tempdir().unwrap();
        let hijacked = outside.path().join("hijacked.lock");
        let temp = index_root();
        plant(&temp.path().join(".lock"), &hijacked);

        let opened = ScopedContextIndex::open(temp.path(), options(Engine::Flat, 8)).err();
        assert!(
            matches!(opened, Some(ContextIndexError::UnsafeRootLock)),
            "opened through a symlinked lock: {opened:?}"
        );
        assert!(!hijacked.exists(), "lock file created outside the root");
    }

    /// Discard must never delete a file that IS the named scope's shard, even
    /// though quarantine said otherwise when the index was opened.
    #[test]
    fn discard_refuses_a_file_that_is_a_valid_shard_for_the_scope() {
        let target = scope("acme", &["revived"]);
        let probe = index_root();
        let valid = isolated_shard(Engine::Flat, probe.path(), &target, "still-here");
        let filename = valid.file_name().unwrap().to_str().unwrap().to_string();

        let outside = tempfile::tempdir().unwrap();
        let temp = index_root();
        plant(
            &temp.path().join(&filename),
            &outside.path().join("missing.redb"),
        );
        let index = ScopedContextIndex::open(temp.path(), options(Engine::Flat, 8)).unwrap();
        assert_eq!(index.quarantined_shards().unwrap().len(), 1);

        // The name now holds this scope's real shard, changed after open.
        std::fs::remove_file(temp.path().join(&filename)).unwrap();
        std::fs::copy(&valid, temp.path().join(&filename)).unwrap();

        let discarded = index.discard_quarantined(&target);
        assert!(
            matches!(discarded, Err(ContextIndexError::ShardNotDiscardable(_))),
            "a live shard was discardable: {discarded:?}"
        );
        assert!(
            temp.path().join(&filename).exists(),
            "a valid shard was deleted by remediation"
        );
    }
}

/// Attacks that swap what a name points at, rather than planting a name.
///
/// Unix only: both primitives (renaming over a live name, aliasing a foreign
/// file into the root with a hard link) need filesystem semantics the Windows
/// attacker model here does not assume.
#[cfg(unix)]
mod inode {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// Two directories guaranteed to share a filesystem, so `hard_link` works.
    ///
    /// The index root is created private, because `open` now refuses one that
    /// is not.
    fn same_filesystem_dirs() -> (tempfile::TempDir, PathBuf, PathBuf) {
        use std::os::unix::fs::DirBuilderExt as _;

        let base = tempfile::tempdir().unwrap();
        let root = base.path().join("index");
        let outside = base.path().join("outside");
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&root)
            .unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        (base, root, outside)
    }

    /// The scratch name is claimed with `O_EXCL`, but an attacker can rename
    /// something over it while the shard is being built. Publishing must then
    /// fail rather than link a foreign inode into a tenant's shard name.
    ///
    /// The thief races, so a round may simply not be hijacked. The assertions
    /// never require the hijack to land: they require that whenever `insert`
    /// reports success, the vector really is there — including after a reopen,
    /// which is where the orphaned-inode variant loses the data.
    #[test]
    fn scratch_name_hijack_never_publishes_a_foreign_inode() {
        let (_base, root, outside) = same_filesystem_dirs();
        let stolen = outside.join("stolen.redb");
        let stop = Arc::new(AtomicBool::new(false));
        let thief = {
            let root = root.clone();
            let stolen = stolen.clone();
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let Ok(entries) = std::fs::read_dir(&root) else {
                        continue;
                    };
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        if name.to_str().is_some_and(|n| n.starts_with(".create-")) {
                            let path = entry.path();
                            let _ = std::fs::remove_file(&path);
                            let _ = std::os::unix::fs::symlink(&stolen, &path);
                        }
                    }
                }
            })
        };

        let index = ScopedContextIndex::open(&root, options(Engine::Flat, 8)).unwrap();
        let mut stored = Vec::new();
        for round in 0..40 {
            let target = scope("acme", &[&format!("round-{round}")]);
            if index.insert(&target, point("row", [1.0, 0.0, 0.0])).is_ok() {
                stored.push(target);
            }
        }
        stop.store(true, Ordering::Relaxed);
        thief.join().unwrap();

        // A reported success must be a real, readable vector.
        for target in &stored {
            let matches = index.search(target, &[1.0, 0.0, 0.0], 1).unwrap();
            assert_eq!(matches.len(), 1, "reported stored but absent in memory");
        }
        // Every published shard is a regular file this index alone names.
        for path in shard_files(&root) {
            let metadata = std::fs::symlink_metadata(&path).unwrap();
            assert!(
                metadata.is_file(),
                "published shard is not a regular file: {path:?}"
            );
        }
        assert!(!stolen.exists(), "the engine wrote outside the index root");

        // The decisive one: an insert that returned Ok while the engine held an
        // orphaned inode looks fine until the handle is dropped.
        drop(index);
        let reopened = ScopedContextIndex::open(&root, options(Engine::Flat, 8)).unwrap();
        for target in &stored {
            let matches = reopened.search(target, &[1.0, 0.0, 0.0], 1).unwrap();
            assert_eq!(
                matches.len(),
                1,
                "insert reported success but the vector did not survive a reopen"
            );
        }
    }

    fn reserved_entries(root: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with(".staging-") || name.starts_with(".create-")
                    })
            })
            .collect()
    }

    /// The hijack test above passes because the scratch name is no longer
    /// somewhere an attacker can reach, so the mode on that directory is the
    /// control doing the work. Assert it directly, or the other test could
    /// pass merely because the thief lost track of the name.
    ///
    /// This bounds the threat model too: the directory keeps out other users,
    /// not the user running the index. An attacker with this uid can reach the
    /// shard files themselves and no permission bit changes that.
    #[test]
    fn staging_directory_is_private_and_cleaned_up() {
        use std::os::unix::fs::PermissionsExt;

        let temp = index_root();
        let index = ScopedContextIndex::open(temp.path(), options(Engine::Flat, 8)).unwrap();

        let staging = reserved_entries(temp.path());
        assert_eq!(staging.len(), 1, "expected exactly one staging directory");
        let metadata = std::fs::symlink_metadata(&staging[0]).unwrap();
        assert!(metadata.is_dir());
        assert_eq!(
            metadata.permissions().mode() & 0o777,
            0o700,
            "staging directory is writable by someone other than this user"
        );

        drop(index);
        assert!(
            reserved_entries(temp.path()).is_empty(),
            "staging directory outlived its handle"
        );
    }

    /// Scratch state left by a crashed handle must not accumulate, and must
    /// never be mistaken for index content.
    #[test]
    fn reserved_leftovers_are_swept_at_open() {
        let temp = index_root();
        let stale_dir = temp.path().join(".staging-deadbeefdeadbeef");
        std::fs::create_dir(&stale_dir).unwrap();
        std::fs::write(stale_dir.join(".create-0123"), b"half-built").unwrap();
        std::fs::write(temp.path().join(".create-orphan"), b"scratch").unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("nowhere"),
            temp.path().join(".create-symlink"),
        )
        .unwrap();

        let index = ScopedContextIndex::open(temp.path(), options(Engine::Flat, 8)).unwrap();
        assert!(index.quarantined_shards().unwrap().is_empty());
        assert!(!stale_dir.exists(), "stale staging directory survived open");
        assert!(!temp.path().join(".create-orphan").exists());
        assert!(!temp.path().join(".create-symlink").exists());
        // Only this handle's own staging directory remains.
        assert_eq!(reserved_entries(temp.path()).len(), 1);
    }

    /// A hard-link alias is indistinguishable from a regular file by type, so
    /// only the link count keeps the engine from opening — and initialising —
    /// a file that belongs to something else.
    #[test]
    fn hard_link_alias_is_quarantined_without_clobbering_the_aliased_file() {
        let victim = scope("victim", &["docs"]);
        let probe = index_root();
        let filename = shard_filename_of(Engine::Flat, probe.path(), &victim);

        let (_base, root, outside) = same_filesystem_dirs();
        let foreign = outside.join("someone-elses.dat");
        std::fs::write(&foreign, b"").unwrap();
        std::fs::hard_link(&foreign, root.join(&filename)).unwrap();

        let index = ScopedContextIndex::open(&root, options(Engine::Flat, 8)).unwrap();
        assert_eq!(
            std::fs::metadata(&foreign).unwrap().len(),
            0,
            "open() initialised a database over an aliased foreign file"
        );

        let quarantined = index.quarantined_shards().unwrap();
        assert_eq!(quarantined.len(), 1);
        assert_eq!(quarantined[0].filename, filename);
        assert!(matches!(
            index.scope_stats(&victim),
            Err(ContextIndexError::ScopeQuarantined(_))
        ));

        // Discarding removes only the alias; the foreign file keeps its data.
        assert!(index.discard_quarantined(&victim).unwrap());
        assert!(foreign.exists(), "discard deleted the aliased file itself");
        assert_eq!(std::fs::metadata(&foreign).unwrap().len(), 0);
    }

    /// The same alias aimed at a populated file must also be left untouched.
    #[test]
    fn hard_link_alias_never_mutates_a_populated_foreign_file() {
        let victim = scope("victim", &["docs"]);
        let probe = index_root();
        let filename = shard_filename_of(Engine::Flat, probe.path(), &victim);

        let (_base, root, outside) = same_filesystem_dirs();
        let foreign = outside.join("ledger.db");
        let original = b"important bytes that are not a vector database".to_vec();
        std::fs::write(&foreign, &original).unwrap();
        std::fs::hard_link(&foreign, root.join(&filename)).unwrap();

        let index = ScopedContextIndex::open(&root, options(Engine::Flat, 8)).unwrap();
        assert_eq!(std::fs::read(&foreign).unwrap(), original);
        assert_eq!(index.quarantined_shards().unwrap().len(), 1);
    }
}

/// The root directory is the boundary the crate rests on, so its permissions
/// and those of everything inside it are asserted directly.
///
/// Unix only: no mode enforcement is possible elsewhere.
#[cfg(unix)]
mod private_root {
    use super::*;
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

    fn mode_of(path: &Path) -> u32 {
        std::fs::symlink_metadata(path)
            .unwrap()
            .permissions()
            .mode()
            & 0o777
    }

    fn reachable_by_others(path: &Path) -> bool {
        mode_of(path) & 0o077 != 0
    }

    /// A `0600` file and a `0700` directory survive any umask, because a umask
    /// only clears bits. So this holds whatever the ambient umask is — which
    /// matters, since under the common `022` the previous defaults published
    /// shards `0644` inside a `0755` root.
    #[test]
    fn nothing_under_the_root_is_reachable_by_other_users() {
        let base = tempfile::tempdir().unwrap();
        let root = base.path().join("index");
        let index = ScopedContextIndex::open(&root, options(Engine::Flat, 8)).unwrap();
        let target = scope("acme", &["private"]);
        index
            .insert(&target, point("row", [1.0, 0.0, 0.0]))
            .unwrap();

        assert_eq!(mode_of(&root), 0o700, "root is not private");
        assert_eq!(mode_of(&root.join(".lock")), 0o600, "lock is not private");
        for shard in shard_files(&root) {
            assert_eq!(mode_of(&shard), 0o600, "shard {shard:?} is not private");
        }
        for entry in std::fs::read_dir(&root).unwrap() {
            let path = entry.unwrap().path();
            assert!(
                !reachable_by_others(&path),
                "{path:?} is reachable by other users: {:04o}",
                mode_of(&path)
            );
        }

        // And the shard is still private after a reopen, which is where the
        // engine rewrites the file.
        drop(index);
        let reopened = ScopedContextIndex::open(&root, options(Engine::Flat, 8)).unwrap();
        assert_eq!(reopened.scope_count().unwrap(), 1);
        for shard in shard_files(&root) {
            assert_eq!(mode_of(&shard), 0o600);
        }
    }

    #[test]
    fn a_root_other_users_can_reach_is_refused() {
        for mode in [0o777, 0o755, 0o750, 0o720, 0o701] {
            let base = tempfile::tempdir().unwrap();
            let root = base.path().join("index");
            std::fs::DirBuilder::new().mode(mode).create(&root).unwrap();
            // `mkdir` applies the umask, so confirm the case is really testable
            // before asserting on it.
            if !reachable_by_others(&root) {
                continue;
            }
            let opened = ScopedContextIndex::open(&root, options(Engine::Flat, 8)).err();
            assert!(
                matches!(opened, Some(ContextIndexError::InsecureRoot(_))),
                "mode {mode:04o} was accepted: {opened:?}"
            );
        }
    }

    #[test]
    fn a_private_root_created_by_an_operator_is_accepted_unchanged() {
        let base = tempfile::tempdir().unwrap();
        let root = base.path().join("index");
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&root)
            .unwrap();
        let index = ScopedContextIndex::open(&root, options(Engine::Flat, 8)).unwrap();
        assert_eq!(mode_of(&root), 0o700);
        drop(index);
    }
}
