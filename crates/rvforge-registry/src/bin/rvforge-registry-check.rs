//! `rvforge-registry-check <registry-dir>` — audit a registry directory written
//! by anyone.
//!
//! The publisher CLI (`@ruvector/rvforge`) and this crate both write the layout
//! in `docs/research/rvf-forge/registry-model.md`, and they were built
//! independently. This binary is the arbiter: it opens a directory, reads it
//! with nothing but the crate's public API, and reports every place the bytes on
//! disk disagree with the model.
//!
//! It answers six questions, in the order a reader would have to answer them:
//!
//! ```text
//! objects      does every file hash to the name it is filed under?
//! log          is the Merkle log self-consistent, and does the head match?
//! releases     does every release satisfy the publication rules?
//! packages     is each package index a chain, and does it agree with the log?
//! witness      does every receipt chain link and recompute?
//! revocations  does every revocation name something that exists?
//! ```
//!
//! Exit codes: `0` clean, `1` violations found (each printed with the rule it
//! broke), `2` the directory could not be read at all.
//!
//! Nothing here mutates the registry — it opens read paths only, so running it
//! against a live directory is safe.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::ExitCode;

use rvf_forge_core::canonical::canonical_sha256_hex;
use rvforge_registry::crypto::{verify_over_id, verifying_key_from_b64};
use rvforge_registry::objects::{
    CapabilityManifest, LogObjectType, PublisherRecord, Release, Revocation, SignatureRole,
    SubjectType, TransparencyLogEntry, TreeHead, TrustLevel, TrustUpdate, WitnessReceipt,
    TYPE_CAPABILITY_MANIFEST, TYPE_PUBLISHER, TYPE_RELEASE, TYPE_REVOCATION, TYPE_TRUST_UPDATE,
    TYPE_WITNESS_RECEIPT,
};
use rvforge_registry::{Registry, RegistryError};
use serde_json::Value;

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let Some(dir) = args.next() else {
        eprintln!("usage: rvforge-registry-check <registry-dir>");
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("usage: rvforge-registry-check <registry-dir>");
        return ExitCode::from(2);
    }

    let dir = Path::new(&dir);
    let registry = match Registry::open(dir) {
        Ok(registry) => registry,
        Err(e) => {
            eprintln!("cannot open {}: {e}", dir.display());
            return ExitCode::from(2);
        }
    };

    let mut report = Report::default();
    let objects = check_objects(&registry, &mut report);
    check_log(&registry, &objects, &mut report);
    check_releases(&registry, &objects, &mut report);
    check_packages(&registry, &objects, &mut report);
    check_witness(&registry, &objects, &mut report);
    check_revocations(&registry, &mut report);

    println!("registry {}", dir.display());
    for (kind, count) in objects.counts() {
        println!("  {count:>4} {kind}");
    }
    if report.violations.is_empty() {
        println!("  registry is consistent with registry-model.md");
        ExitCode::SUCCESS
    } else {
        println!("  {} violation(s)", report.violations.len());
        for violation in &report.violations {
            println!("violation [{}] {}", violation.area, violation.detail);
        }
        ExitCode::FAILURE
    }
}

/// One rule that the directory broke, named by the area that owns it.
struct Violation {
    area: &'static str,
    detail: String,
}

#[derive(Default)]
struct Report {
    violations: Vec<Violation>,
}

impl Report {
    fn fail(&mut self, area: &'static str, detail: impl Into<String>) {
        self.violations.push(Violation {
            area,
            detail: detail.into(),
        });
    }

    /// Record `result`'s error, if it has one, and report whether it was clean.
    fn absorb<T>(
        &mut self,
        area: &'static str,
        context: &str,
        result: Result<T, RegistryError>,
    ) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(e) => {
                self.fail(area, format!("{context}: {e}"));
                None
            }
        }
    }
}

/// What the object store holds, indexed by id.
#[derive(Default)]
struct Objects {
    by_type: BTreeMap<String, Vec<String>>,
    types: BTreeMap<String, String>,
}

impl Objects {
    fn ids_of(&self, object_type: &str) -> &[String] {
        self.by_type
            .get(object_type)
            .map_or(&[][..], |ids| ids.as_slice())
    }

    fn type_of(&self, id: &str) -> Option<&str> {
        self.types.get(id).map(String::as_str)
    }

    fn counts(&self) -> Vec<(&str, usize)> {
        self.by_type
            .iter()
            .map(|(kind, ids)| (kind.as_str(), ids.len()))
            .collect()
    }
}

/// Every object hashes to the name it is filed under, and parses as its type.
///
/// Typed objects go through [`rvforge_registry::ObjectStore::get`], which
/// re-derives the content address and applies the type's own field rules. An
/// object of a type this crate does not model — the CLI also stores a software
/// inventory and a pack provenance record — still has to be content-addressed,
/// so its id is recomputed here from the same exclusion rule: drop
/// `signatures`, and drop whichever field holds the object's own id.
fn check_objects(registry: &Registry, report: &mut Report) -> Objects {
    let mut objects = Objects::default();
    let store = registry.store();

    let Some(ids) = report.absorb("objects", "enumerating objects", store.ids()) else {
        return objects;
    };

    for id in ids {
        let Some(bytes) = report.absorb("objects", &format!("reading {id}"), store.get_bytes(&id))
        else {
            continue;
        };
        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(e) => {
                report.fail("objects", format!("{id} is not JSON: {e}"));
                continue;
            }
        };
        let Some(map) = value.as_object() else {
            report.fail("objects", format!("{id} is not a JSON object"));
            continue;
        };
        let object_type = map
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("(untyped)")
            .to_string();

        let typed = match object_type.as_str() {
            TYPE_PUBLISHER => store.get::<PublisherRecord>(&id).map(|_| ()),
            TYPE_RELEASE => store.get::<Release>(&id).map(|_| ()),
            TYPE_CAPABILITY_MANIFEST => store.get::<CapabilityManifest>(&id).map(|_| ()),
            TYPE_WITNESS_RECEIPT => store.get::<WitnessReceipt>(&id).map(|_| ()),
            TYPE_REVOCATION => store.get::<Revocation>(&id).map(|_| ()),
            TYPE_TRUST_UPDATE => store.get::<TrustUpdate>(&id).map(|_| ()),
            _ => {
                check_untyped_address(&id, &value, report);
                Ok(())
            }
        };
        if let Err(e) = typed {
            report.fail("objects", format!("{object_type} {id}: {e}"));
        }

        objects.types.insert(id.clone(), object_type.clone());
        objects.by_type.entry(object_type).or_default().push(id);
    }
    objects
}

/// Content-address an object whose type this crate does not model.
fn check_untyped_address(id: &str, value: &Value, report: &mut Report) {
    let mut content = value.clone();
    let Some(map) = content.as_object_mut() else {
        return;
    };
    map.remove("signatures");
    // A field cannot contain its own hash, so whichever field holds this
    // object's id is excluded — identified by its value, since the field's
    // name is only known for the types this crate models.
    let self_id: Vec<String> = map
        .iter()
        .filter(|(_, v)| v.as_str() == Some(id))
        .map(|(k, _)| k.clone())
        .collect();
    for key in self_id {
        map.remove(&key);
    }

    match canonical_sha256_hex(&content) {
        Ok(hex) => {
            let computed = format!("sha256:{hex}");
            if computed != id {
                report.fail(
                    "objects",
                    format!(
                        "{id} is filed under an address its content does not produce ({computed})"
                    ),
                );
            }
        }
        Err(e) => report.fail("objects", format!("{id} is not canonicalizable: {e}")),
    }
}

/// The log recomputes, every entry proves inclusion, and the head file agrees.
fn check_log(registry: &Registry, objects: &Objects, report: &mut Report) {
    let log = registry.log();
    report.absorb("log", "recomputing the log", log.verify_consistency());

    let Some(entries) = report.absorb("log", "reading entries.jsonl", log.entries()) else {
        return;
    };
    let Some(head) = report.absorb("log", "computing the tree head", log.tree_head()) else {
        return;
    };

    if head.tree_size as usize != entries.len() {
        report.fail(
            "log",
            format!(
                "tree head covers {} leaves but the log has {} entries",
                head.tree_size,
                entries.len()
            ),
        );
    }

    let mut logged: BTreeSet<&str> = BTreeSet::new();
    for entry in &entries {
        if !logged.insert(entry.object.as_str()) {
            report.fail(
                "log",
                format!(
                    "{} is logged twice; the log is append-once per object",
                    entry.object
                ),
            );
        }
        check_entry_type(entry, objects, report);

        match log.inclusion_proof(&entry.object) {
            Ok(proof) => {
                if !proof.verify(&head.root_hash) {
                    report.fail(
                        "log",
                        format!(
                            "entry {} ({}) does not prove inclusion in tree head {}",
                            entry.index, entry.object, head.root_hash
                        ),
                    );
                }
            }
            Err(e) => report.fail(
                "log",
                format!("entry {} has no inclusion proof: {e}", entry.index),
            ),
        }
    }

    check_head_file(registry, &head, report);
}

/// The logged `objectType` matches the stored object's `type`.
fn check_entry_type(entry: &TransparencyLogEntry, objects: &Objects, report: &mut Report) {
    let Some(stored) = objects.type_of(&entry.object) else {
        report.fail(
            "log",
            format!(
                "entry {} logs {}, which is not in the object store",
                entry.index, entry.object
            ),
        );
        return;
    };
    let expected = match entry.object_type_logged {
        LogObjectType::Release => TYPE_RELEASE,
        LogObjectType::Revocation => TYPE_REVOCATION,
        LogObjectType::Publisher => TYPE_PUBLISHER,
        LogObjectType::TrustUpdate => TYPE_TRUST_UPDATE,
    };
    if stored != expected {
        report.fail(
            "log",
            format!(
                "entry {} logs {} as {:?}, but the stored object is a {stored:?}",
                entry.index, entry.object, entry.object_type_logged
            ),
        );
    }
}

/// `log/tree-head.json` parses as a `TreeHead` and matches the recomputed one.
fn check_head_file(registry: &Registry, computed: &TreeHead, report: &mut Report) {
    let path = registry.root().join("log").join("tree-head.json");
    if !path.exists() {
        if computed.tree_size > 0 {
            report.fail(
                "log",
                "log/tree-head.json is missing but the log is not empty",
            );
        }
        return;
    }
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            report.fail("log", format!("cannot read {}: {e}", path.display()));
            return;
        }
    };
    let stored: TreeHead = match serde_json::from_slice(&bytes) {
        Ok(head) => head,
        Err(e) => {
            report.fail("log", format!("{} is not a TreeHead: {e}", path.display()));
            return;
        }
    };
    if stored.root_hash != computed.root_hash || stored.tree_size != computed.tree_size {
        report.fail(
            "log",
            format!(
                "log/tree-head.json claims {} over {} leaves; the entries produce {} over {}",
                stored.root_hash, stored.tree_size, computed.root_hash, computed.tree_size
            ),
        );
    }
}

/// Every stored release satisfies the publication rules of the model.
fn check_releases(registry: &Registry, objects: &Objects, report: &mut Report) {
    for id in objects.ids_of(TYPE_RELEASE) {
        let Some(release) = report.absorb("releases", id, registry.store().get::<Release>(id))
        else {
            continue;
        };

        // Absence from the log means unpublished, however complete the record.
        match registry.is_published(id) {
            Ok(true) => {}
            Ok(false) => report.fail(
                "releases",
                format!("{id} is in the object store but absent from the transparency log"),
            ),
            Err(e) => report.fail("releases", format!("{id}: {e}")),
        }

        if release.trust_level != TrustLevel::INITIAL {
            report.fail(
                "releases",
                format!(
                    "{id} was published at trustLevel {:?}; a publisher may only assert {:?}",
                    release.trust_level,
                    TrustLevel::INITIAL
                ),
            );
        }

        check_publisher_signature(registry, &release, report);

        // (c) the capability manifest resolves.
        match registry
            .store()
            .get_optional::<CapabilityManifest>(&release.capability_manifest)
        {
            Ok(Some(_)) => {}
            Ok(None) => report.fail(
                "releases",
                format!(
                    "{id} names capabilityManifest {}, which is not in the object store",
                    release.capability_manifest
                ),
            ),
            Err(e) => report.fail("releases", format!("{id} capabilityManifest: {e}")),
        }

        check_predecessor(registry, &release, report);
        report.absorb(
            "releases",
            &format!("{id} execution status"),
            registry.execution_status(id),
        );
    }
}

/// (a) the publisher signature verifies against a live key in the record.
fn check_publisher_signature(registry: &Registry, release: &Release, report: &mut Report) {
    let id = &release.release_id;
    let Some(sig) = release.publisher_signature() else {
        report.fail(
            "releases",
            format!("{id} carries no publisher-role signature"),
        );
        return;
    };

    let publisher = match registry
        .store()
        .get_optional::<PublisherRecord>(&release.package.publisher_id)
    {
        Ok(Some(publisher)) => publisher,
        Ok(None) => {
            report.fail(
                "releases",
                format!(
                    "{id} names package.publisherId {}, which is not in the object store",
                    release.package.publisher_id
                ),
            );
            return;
        }
        Err(e) => {
            report.fail("releases", format!("{id} package.publisherId: {e}"));
            return;
        }
    };

    let Some(entry) = publisher.live_key(&sig.key_id) else {
        report.fail(
            "releases",
            format!(
                "{id} is signed by {}, which is {} publisher {}",
                sig.key_id,
                if publisher.knows_key(&sig.key_id) {
                    "revoked in"
                } else {
                    "not listed in"
                },
                publisher.publisher_id
            ),
        );
        return;
    };

    match verifying_key_from_b64(&entry.public_key) {
        Ok(key) => {
            if let Err(e) = verify_over_id(id, &sig.sig, &key) {
                report.fail("releases", format!("{id} publisher signature: {e}"));
            }
        }
        Err(e) => report.fail(
            "releases",
            format!("{id} signing key {} is unusable: {e}", sig.key_id),
        ),
    }

    for extra in &release.signatures {
        if extra.role == SignatureRole::Registry {
            if let Some(key) = registry.registry_key() {
                if extra.key_id != key.key_id {
                    report.fail(
                        "releases",
                        format!(
                            "{id} is countersigned by {}, not the registry key",
                            extra.key_id
                        ),
                    );
                }
            }
        }
    }
}

/// (d) `predecessor` is null, or a published release of the same package.
fn check_predecessor(registry: &Registry, release: &Release, report: &mut Report) {
    let Some(predecessor) = &release.predecessor else {
        return;
    };
    let id = &release.release_id;
    let previous = match registry.store().get_optional::<Release>(predecessor) {
        Ok(Some(previous)) => previous,
        Ok(None) => {
            report.fail(
                "releases",
                format!("{id} names predecessor {predecessor}, which is not a stored release"),
            );
            return;
        }
        Err(e) => {
            report.fail("releases", format!("{id} predecessor {predecessor}: {e}"));
            return;
        }
    };
    if !previous.same_package_as(release) {
        report.fail(
            "releases",
            format!(
                "{id} names predecessor {predecessor}, which belongs to {}/{}",
                previous.package.publisher_id, previous.package.name
            ),
        );
    }
    match registry.is_published(predecessor) {
        Ok(true) => {}
        Ok(false) => report.fail(
            "releases",
            format!("{id} names predecessor {predecessor}, which was never published"),
        ),
        Err(e) => report.fail("releases", format!("{id} predecessor {predecessor}: {e}")),
    }
}

/// Each package index is a lineage chain that agrees with the stored releases.
fn check_packages(registry: &Registry, objects: &Objects, report: &mut Report) {
    let Some(packages) = report.absorb("packages", "enumerating packages", registry.packages())
    else {
        return;
    };

    let mut indexed: BTreeSet<String> = BTreeSet::new();
    for (publisher_id, name) in packages {
        let label = format!("{publisher_id}/{name}");
        let Some(entries) = report.absorb(
            "packages",
            &label,
            registry.releases_for(&publisher_id, &name),
        ) else {
            continue;
        };

        let mut expected_predecessor: Option<String> = None;
        let mut versions: BTreeSet<&str> = BTreeSet::new();
        for entry in &entries {
            indexed.insert(entry.release_id.clone());

            let release = match registry.store().get_optional::<Release>(&entry.release_id) {
                Ok(Some(release)) => release,
                Ok(None) => {
                    report.fail(
                        "packages",
                        format!("{label} indexes {}, which is not stored", entry.release_id),
                    );
                    continue;
                }
                Err(e) => {
                    report.fail(
                        "packages",
                        format!("{label} entry {}: {e}", entry.release_id),
                    );
                    continue;
                }
            };

            if release.package.name != name || release.package.publisher_id != publisher_id {
                report.fail(
                    "packages",
                    format!(
                        "{label} indexes {}, which belongs to {}/{}",
                        entry.release_id, release.package.publisher_id, release.package.name
                    ),
                );
            }
            if entry.version != release.version
                || entry.predecessor != release.predecessor
                || entry.published_at != release.published_at
            {
                report.fail(
                    "packages",
                    format!(
                        "{label} entry for {} disagrees with the release it names",
                        entry.release_id
                    ),
                );
            }
            if !versions.insert(entry.version.as_str()) {
                report.fail(
                    "packages",
                    format!("{label} lists version {} twice", entry.version),
                );
            }
            if entry.predecessor != expected_predecessor {
                report.fail(
                    "packages",
                    format!(
                        "{label} entry {} names predecessor {:?}, but the head before it was {:?}",
                        entry.release_id, entry.predecessor, expected_predecessor
                    ),
                );
            }
            expected_predecessor = Some(entry.release_id.clone());
        }
    }

    for id in objects.ids_of(TYPE_RELEASE) {
        if !indexed.contains(id) {
            report.fail(
                "packages",
                format!("release {id} appears in no packages/<publisher>/<name>/releases.jsonl"),
            );
        }
    }
}

/// Every receipt chain recomputes, links, and covers every stored receipt.
fn check_witness(registry: &Registry, objects: &Objects, report: &mut Report) {
    let Some(subjects) = report.absorb(
        "witness",
        "enumerating witness chains",
        registry.witness_subjects(),
    ) else {
        return;
    };

    let mut chained: BTreeSet<String> = BTreeSet::new();
    for subject in subjects {
        report.absorb(
            "witness",
            &format!("chain for {subject}"),
            registry.verify_witness_chain(&subject),
        );
        if let Some(receipts) = report.absorb(
            "witness",
            &format!("receipts for {subject}"),
            registry.receipts_for(&subject),
        ) {
            for receipt in receipts {
                if receipt.subject != subject {
                    report.fail(
                        "witness",
                        format!(
                            "receipt {} is filed under {subject} but names subject {}",
                            receipt.receipt_id, receipt.subject
                        ),
                    );
                }
                if !chained.insert(receipt.receipt_id.clone()) {
                    report.fail(
                        "witness",
                        format!(
                            "receipt {} appears in more than one chain",
                            receipt.receipt_id
                        ),
                    );
                }
            }
        }
    }

    for id in objects.ids_of(TYPE_WITNESS_RECEIPT) {
        if !chained.contains(id) {
            report.fail(
                "witness",
                format!("receipt {id} is stored but indexed by no witness chain"),
            );
        }
    }
}

/// Every revocation names a subject the registry can resolve.
fn check_revocations(registry: &Registry, report: &mut Report) {
    let Some(revocations) =
        report.absorb("revocations", "reading revocations", registry.revocations())
    else {
        return;
    };
    for revocation in revocations {
        match revocation.subject_type {
            SubjectType::Release | SubjectType::Publisher => {
                match registry.store().contains(&revocation.subject) {
                    Ok(true) => {}
                    Ok(false) => report.fail(
                        "revocations",
                        format!(
                            "{} revokes {}, which is not in the object store",
                            revocation.revocation_id, revocation.subject
                        ),
                    ),
                    Err(e) => {
                        report.fail("revocations", format!("{}: {e}", revocation.revocation_id))
                    }
                }
            }
            // A key id is not a content address, so there is nothing to resolve.
            SubjectType::PublisherKey => {}
        }
    }
}
