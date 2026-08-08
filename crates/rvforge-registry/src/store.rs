//! Content-addressed object store: `objects/sha256/<2-hex>/<digest>.json`.
//!
//! Writes are idempotent by construction — the path is derived from the
//! content — with one wrinkle. Content addressing excludes `signatures`, so two
//! puts of the same id can carry different signature sets: a publisher's
//! release, and the same release once the registry has countersigned it. A put
//! therefore *unions* the signature arrays rather than overwriting, so a
//! re-publish by the publisher cannot silently drop the countersignature that
//! makes the release executable (ADR-294 §9).

use crate::error::{RegistryError, Result};
use crate::objects::{compute_id, digest_hex, seal, validate, RegistryObject, Signature};
use rvf_forge_core::canonical::to_canonical_json;
use std::fs;
use std::path::{Path, PathBuf};

/// The directory holding all content-addressed objects.
pub const OBJECTS_DIR: &str = "objects";

/// A file-backed content-addressed object store.
#[derive(Clone, Debug)]
pub struct ObjectStore {
    root: PathBuf,
}

impl ObjectStore {
    /// Open (creating if needed) the store rooted at `registry_root`.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Io`] if the directory cannot be created.
    pub fn open(registry_root: impl AsRef<Path>) -> Result<Self> {
        let root = registry_root.as_ref().join(OBJECTS_DIR).join("sha256");
        fs::create_dir_all(&root).map_err(|e| RegistryError::io(&root, &e))?;
        Ok(Self { root })
    }

    /// The on-disk path an id maps to, whether or not it exists.
    ///
    /// # Errors
    ///
    /// [`RegistryError::InvalidObject`] if `id` is not a `sha256:` digest.
    pub fn path_for(&self, id: &str) -> Result<PathBuf> {
        let hex = digest_hex("object id", id)?;
        Ok(self.root.join(&hex[..2]).join(format!("{hex}.json")))
    }

    /// Whether an object is present.
    ///
    /// # Errors
    ///
    /// As [`ObjectStore::path_for`].
    pub fn contains(&self, id: &str) -> Result<bool> {
        Ok(self.path_for(id)?.exists())
    }

    /// Validate, address, and write an object, returning it with its id set.
    ///
    /// # Errors
    ///
    /// [`RegistryError::InvalidObject`] for a malformed object,
    /// [`RegistryError::IdMismatch`] if a declared id disagrees with the
    /// content, [`RegistryError::Io`] on write failure.
    pub fn put<T: RegistryObject>(&self, object: T) -> Result<T> {
        let mut sealed = seal(object)?;
        let path = self.path_for(sealed.id())?;

        if path.exists() {
            let existing: T = self.read_at(&path, sealed.id())?;
            sealed.set_signatures(union_signatures(existing.signatures(), sealed.signatures()));
        }

        let parent = path
            .parent()
            .ok_or_else(|| RegistryError::invalid("object path has no parent"))?;
        fs::create_dir_all(parent).map_err(|e| RegistryError::io(parent, &e))?;
        let body = to_canonical_json(&sealed)?;
        fs::write(&path, body.as_bytes()).map_err(|e| RegistryError::io(&path, &e))?;
        Ok(sealed)
    }

    /// Read an object by id.
    ///
    /// The read re-derives the content address and refuses anything whose
    /// content no longer hashes to the name it is filed under, so an edit made
    /// directly on disk surfaces as [`RegistryError::IdMismatch`] rather than
    /// being served as authentic.
    ///
    /// # Errors
    ///
    /// [`RegistryError::NotFound`] if absent, [`RegistryError::InvalidObject`]
    /// if the stored bytes do not parse or validate,
    /// [`RegistryError::IdMismatch`] if the content has been tampered with.
    pub fn get<T: RegistryObject>(&self, id: &str) -> Result<T> {
        let path = self.path_for(id)?;
        if !path.exists() {
            return Err(RegistryError::NotFound {
                kind: T::TYPE.into(),
                id: id.into(),
            });
        }
        self.read_at(&path, id)
    }

    /// Read an object by id, returning `None` rather than erroring when absent.
    ///
    /// # Errors
    ///
    /// As [`ObjectStore::get`], minus [`RegistryError::NotFound`].
    pub fn get_optional<T: RegistryObject>(&self, id: &str) -> Result<Option<T>> {
        if self.contains(id)? {
            self.get(id).map(Some)
        } else {
            Ok(None)
        }
    }

    /// The stored bytes for an id, exactly as written.
    ///
    /// Revocation blocks execution but never removes an object, so this stays
    /// available for export and forensics after a revocation (ADR-294 §9).
    ///
    /// # Errors
    ///
    /// [`RegistryError::NotFound`] if absent, [`RegistryError::Io`] on read
    /// failure.
    pub fn get_bytes(&self, id: &str) -> Result<Vec<u8>> {
        let path = self.path_for(id)?;
        if !path.exists() {
            return Err(RegistryError::NotFound {
                kind: "object".into(),
                id: id.into(),
            });
        }
        fs::read(&path).map_err(|e| RegistryError::io(&path, &e))
    }

    /// Every object id present in the store, in no particular order.
    ///
    /// An auditor has to be able to enumerate what is on disk: the log names
    /// only what was published, so an object written out of band is invisible
    /// to every other reader on this type.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Io`] on read failure, [`RegistryError::InvalidObject`]
    /// if a file sits at a path that is not a content address.
    pub fn ids(&self) -> Result<Vec<String>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let mut ids = Vec::new();
        for shard in read_dir_sorted(&self.root)? {
            if !shard.is_dir() {
                continue;
            }
            for file in read_dir_sorted(&shard)? {
                let Some(name) = file.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                let Some(hex) = name.strip_suffix(".json") else {
                    continue;
                };
                let id = format!("{}{hex}", crate::objects::DIGEST_PREFIX);
                let shard_name = shard
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                digest_hex("object file name", &id)?;
                if hex[..2] != *shard_name {
                    return Err(RegistryError::invalid(format!(
                        "{} is filed under shard {shard_name:?}, but its digest starts {:?}",
                        file.display(),
                        &hex[..2]
                    )));
                }
                ids.push(id);
            }
        }
        Ok(ids)
    }

    fn read_at<T: RegistryObject>(&self, path: &Path, id: &str) -> Result<T> {
        let bytes = fs::read(path).map_err(|e| RegistryError::io(path, &e))?;
        let object: T = serde_json::from_slice(&bytes).map_err(|e| {
            RegistryError::invalid(format!(
                "{} at {} is not readable: {e}",
                T::TYPE,
                path.display()
            ))
        })?;
        validate(&object)?;
        let computed = compute_id(&object)?;
        if computed != id {
            return Err(RegistryError::IdMismatch {
                declared: id.into(),
                computed,
            });
        }
        Ok(object)
    }
}

/// Directory entries in name order, so enumeration is reproducible.
fn read_dir_sorted(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| RegistryError::io(dir, &e))? {
        paths.push(entry.map_err(|e| RegistryError::io(dir, &e))?.path());
    }
    paths.sort();
    Ok(paths)
}

/// Existing signatures first, then any new ones not already present.
fn union_signatures(existing: &[Signature], incoming: &[Signature]) -> Vec<Signature> {
    let mut merged = existing.to_vec();
    for sig in incoming {
        if !merged.contains(sig) {
            merged.push(sig.clone());
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::{CapabilityManifest, SignatureRole};
    use crate::testkit;

    fn store() -> (tempfile::TempDir, ObjectStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = ObjectStore::open(dir.path()).unwrap();
        (dir, store)
    }

    #[test]
    fn put_then_get_round_trips() {
        let (_dir, store) = store();
        let put = store.put(testkit::capability_manifest()).unwrap();
        let got: CapabilityManifest = store.get(&put.manifest_id).unwrap();
        assert_eq!(got, put);
    }

    #[test]
    fn objects_are_sharded_by_the_first_two_hex_digits() {
        let (_dir, store) = store();
        let put = store.put(testkit::capability_manifest()).unwrap();
        let hex = digest_hex("id", &put.manifest_id).unwrap();
        let path = store.path_for(&put.manifest_id).unwrap();
        assert!(path.ends_with(format!("{}/{hex}.json", &hex[..2])));
        assert!(path.exists());
    }

    #[test]
    fn a_declared_id_that_does_not_match_the_content_is_refused() {
        let (_dir, store) = store();
        let mut m = testkit::capability_manifest();
        m.manifest_id = format!("sha256:{}", "0".repeat(64));
        let err = store.put(m).unwrap_err();
        assert_eq!(err.code(), crate::RegistryErrorCode::IdMismatch);
    }

    #[test]
    fn a_malformed_object_is_refused() {
        let (_dir, store) = store();
        let mut m = testkit::capability_manifest();
        m.requests[0].scope = String::new();
        let err = store.put(m).unwrap_err();
        assert_eq!(err.code(), crate::RegistryErrorCode::InvalidObject);
    }

    #[test]
    fn get_of_an_absent_object_is_not_found() {
        let (_dir, store) = store();
        let err = store
            .get::<CapabilityManifest>(&format!("sha256:{}", "0".repeat(64)))
            .unwrap_err();
        assert_eq!(err.code(), crate::RegistryErrorCode::NotFound);
    }

    #[test]
    fn tampering_with_a_stored_object_is_detected_on_read() {
        let (_dir, store) = store();
        let put = store.put(testkit::capability_manifest()).unwrap();
        let path = store.path_for(&put.manifest_id).unwrap();

        let mut tampered = put.clone();
        tampered.denials.push("network".into());
        fs::write(&path, to_canonical_json(&tampered).unwrap()).unwrap();

        let err = store
            .get::<CapabilityManifest>(&put.manifest_id)
            .unwrap_err();
        assert_eq!(err.code(), crate::RegistryErrorCode::IdMismatch);
    }

    #[test]
    fn re_put_unions_signatures_rather_than_dropping_them() {
        let (_dir, store) = store();
        let base = testkit::capability_manifest();

        let mut publisher_signed = base.clone();
        publisher_signed.signatures.push(crate::objects::Signature {
            key_id: "ed25519:aa".into(),
            role: SignatureRole::Publisher,
            sig: "cHViCg==".into(),
        });
        let first = store.put(publisher_signed).unwrap();

        let mut registry_signed = base;
        registry_signed.signatures.push(crate::objects::Signature {
            key_id: "ed25519:bb".into(),
            role: SignatureRole::Registry,
            sig: "cmVnCg==".into(),
        });
        let second = store.put(registry_signed).unwrap();

        assert_eq!(first.manifest_id, second.manifest_id);
        assert_eq!(second.signatures.len(), 2);
        let stored: CapabilityManifest = store.get(&second.manifest_id).unwrap();
        assert_eq!(stored.signatures.len(), 2);
    }

    #[test]
    fn ids_enumerates_what_was_put_and_nothing_else() {
        let (_dir, store) = store();
        assert!(store.ids().unwrap().is_empty());

        let manifest = store.put(testkit::capability_manifest()).unwrap();
        let mut other = testkit::capability_manifest();
        other.denials.push("network".into());
        let other = store.put(other).unwrap();

        let mut ids = store.ids().unwrap();
        ids.sort();
        let mut want = vec![manifest.manifest_id, other.manifest_id];
        want.sort();
        assert_eq!(ids, want);
    }

    #[test]
    fn ids_rejects_an_object_filed_under_the_wrong_shard() {
        let (dir, store) = store();
        let put = store.put(testkit::capability_manifest()).unwrap();
        let hex = digest_hex("id", &put.manifest_id).unwrap();
        let misfiled = dir.path().join("objects/sha256/ff");
        fs::create_dir_all(&misfiled).unwrap();
        fs::rename(
            store.path_for(&put.manifest_id).unwrap(),
            misfiled.join(format!("{hex}.json")),
        )
        .unwrap();
        assert_eq!(
            store.ids().unwrap_err().code(),
            crate::RegistryErrorCode::InvalidObject
        );
    }

    #[test]
    fn bytes_stay_readable_for_export() {
        let (_dir, store) = store();
        let put = store.put(testkit::capability_manifest()).unwrap();
        let bytes = store.get_bytes(&put.manifest_id).unwrap();
        assert_eq!(bytes, to_canonical_json(&put).unwrap().as_bytes());
    }
}
