//! Optional CDC-backed [`SnapshotStorage`] implementation (`cdc` feature).
//!
//! `LocalStorage` re-serializes and re-writes the entire collection on
//! every `save()`. This backend chunks the same bincode-encoded bytes with
//! `ruvector_cdc_checkpoint`'s content-defined chunker, persists chunks
//! content-addressed on disk (so identical chunks across successive
//! snapshots of the same collection are written once), and verifies every
//! `load()` through the crate's own witness-chain `verify()` — see
//! ADR-350 and `docs/research/nightly/2026-08-27-cdc-witness-checkpoint/`
//! for the measured comparison against `LocalStorage`'s real save path.
//!
//! Known limitations, stated plainly:
//! - Chunk garbage collection is not implemented: `delete()` removes a
//!   snapshot's manifest/metadata but never its chunks, since a sibling
//!   snapshot of the same collection may still reference them. Storage
//!   only grows; see ADR-350's Open Questions.
//! - `round`/`chain_root` bookkeeping reads/writes plain files with no
//!   locking beyond the process-wide mutex below; concurrent `save()`
//!   calls for the *same* collection from *different* processes can race.
//!   Acceptable for this feature-gated, non-default, research-tier
//!   backend; not a claim of production-grade concurrency safety.

use std::path::PathBuf;
use tokio::sync::Mutex;

use async_trait::async_trait;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use ruvector_cdc_checkpoint::chunker::{cdc_boundaries, CdcParams};
use ruvector_cdc_checkpoint::store::{hash_bytes, ChunkHash, ChunkStore};
use ruvector_cdc_checkpoint::witness::{self, CheckpointManifest, WitnessChain};
use sha2::{Digest, Sha256};
use tokio::fs;

use crate::error::{Result, SnapshotError};
use crate::snapshot::{Snapshot, SnapshotData};
use crate::storage::SnapshotStorage;

fn cdc_params() -> CdcParams {
    CdcParams::new(512, 2048, 8192)
}

fn gzip(data: &[u8]) -> Result<Vec<u8>> {
    use std::io::Write;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(data)
        .map_err(|e| SnapshotError::compression(e.to_string()))?;
    encoder
        .finish()
        .map_err(|e| SnapshotError::compression(e.to_string()))
}

fn gunzip(data: &[u8]) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut decoder = GzDecoder::new(data);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| SnapshotError::compression(e.to_string()))?;
    Ok(out)
}

fn hex(hash: &ChunkHash) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

/// On-disk manifest for one snapshot: enough to reconstruct and
/// witness-verify it without depending on any in-memory state from the
/// process that wrote it.
struct PersistedManifest {
    prev_root: [u8; 32],
    round: u64,
    chunk_hashes: Vec<ChunkHash>,
    content_hash: [u8; 32],
    chain_root: [u8; 32],
}

impl PersistedManifest {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + 8 + 4 + self.chunk_hashes.len() * 32 + 32 + 32);
        out.extend_from_slice(&self.prev_root);
        out.extend_from_slice(&self.round.to_le_bytes());
        out.extend_from_slice(&(self.chunk_hashes.len() as u32).to_le_bytes());
        for h in &self.chunk_hashes {
            out.extend_from_slice(h);
        }
        out.extend_from_slice(&self.content_hash);
        out.extend_from_slice(&self.chain_root);
        out
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let err = || SnapshotError::corrupted("truncated CDC manifest file");
        let mut cursor = 0usize;
        let take = |cursor: &mut usize, n: usize| -> Result<&[u8]> {
            let slice = bytes.get(*cursor..*cursor + n).ok_or_else(err)?;
            *cursor += n;
            Ok(slice)
        };
        let prev_root: [u8; 32] = take(&mut cursor, 32)?.try_into().unwrap();
        let round = u64::from_le_bytes(take(&mut cursor, 8)?.try_into().unwrap());
        let n_chunks = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().unwrap()) as usize;
        let mut chunk_hashes = Vec::with_capacity(n_chunks);
        for _ in 0..n_chunks {
            chunk_hashes.push(<[u8; 32]>::try_from(take(&mut cursor, 32)?).unwrap());
        }
        let content_hash: [u8; 32] = take(&mut cursor, 32)?.try_into().unwrap();
        let chain_root: [u8; 32] = take(&mut cursor, 32)?.try_into().unwrap();
        Ok(Self {
            prev_root,
            round,
            chunk_hashes,
            content_hash,
            chain_root,
        })
    }

    fn as_checkpoint_manifest(&self) -> CheckpointManifest {
        // `CheckpointManifest`'s fields are all `pub`, but it has no public
        // constructor (by design — `WitnessChain::append` is the only
        // producer within the crate). Rebuilding one here from persisted,
        // already-committed fields is not forging a manifest: every field
        // was produced by a real `WitnessChain::append` call at save time
        // and is being handed back to the crate's own `verify()`, which
        // independently recomputes the root and rejects any mismatch.
        CheckpointManifest {
            round: self.round,
            chunk_hashes: self.chunk_hashes.clone(),
            content_hash: self.content_hash,
            chain_root: self.chain_root,
        }
    }
}

/// Content-defined-chunking storage backend for [`SnapshotManager`](crate::SnapshotManager).
pub struct CdcLocalStorage {
    base_path: PathBuf,
    /// Guards the read-modify-write of a collection's `chain_root.bin` +
    /// round count against concurrent `save()` calls within this process.
    lock: Mutex<()>,
}

impl CdcLocalStorage {
    pub fn new(base_path: PathBuf) -> Self {
        Self {
            base_path,
            lock: Mutex::new(()),
        }
    }

    fn collection_dir(&self, collection: &str) -> PathBuf {
        self.base_path.join(collection)
    }
    fn chunks_dir(&self, collection: &str) -> PathBuf {
        self.collection_dir(collection).join("chunks")
    }
    fn manifests_dir(&self, collection: &str) -> PathBuf {
        self.collection_dir(collection).join("manifests")
    }
    fn meta_dir(&self, collection: &str) -> PathBuf {
        self.collection_dir(collection).join("meta")
    }
    fn chain_root_path(&self, collection: &str) -> PathBuf {
        self.collection_dir(collection).join("chain_root.bin")
    }
    fn chunk_path(&self, collection: &str, hash: &ChunkHash) -> PathBuf {
        self.chunks_dir(collection)
            .join(format!("{}.chunk.gz", hex(hash)))
    }
    fn manifest_path(&self, collection: &str, id: &str) -> PathBuf {
        self.manifests_dir(collection)
            .join(format!("{id}.manifest"))
    }
    fn meta_path(&self, collection: &str, id: &str) -> PathBuf {
        self.meta_dir(collection).join(format!("{id}.json"))
    }

    async fn ensure_dirs(&self, collection: &str) -> Result<()> {
        fs::create_dir_all(self.chunks_dir(collection)).await?;
        fs::create_dir_all(self.manifests_dir(collection)).await?;
        fs::create_dir_all(self.meta_dir(collection)).await?;
        Ok(())
    }

    async fn read_chain_root(&self, collection: &str) -> Result<[u8; 32]> {
        let path = self.chain_root_path(collection);
        match fs::read(&path).await {
            Ok(bytes) if bytes.len() == 32 => Ok(bytes.try_into().unwrap()),
            Ok(_) => Err(SnapshotError::corrupted(
                "chain_root.bin has the wrong length",
            )),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(witness::GENESIS_ROOT),
            Err(e) => Err(e.into()),
        }
    }

    async fn write_chain_root(&self, collection: &str, root: [u8; 32]) -> Result<()> {
        fs::write(self.chain_root_path(collection), root).await?;
        Ok(())
    }

    async fn next_round(&self, collection: &str) -> Result<u64> {
        let mut n = 0u64;
        let dir = self.manifests_dir(collection);
        if dir.exists() {
            let mut entries = fs::read_dir(&dir).await?;
            while entries.next_entry().await?.is_some() {
                n += 1;
            }
        }
        Ok(n)
    }
}

#[async_trait]
impl SnapshotStorage for CdcLocalStorage {
    async fn save(&self, snapshot_data: &SnapshotData) -> Result<Snapshot> {
        let collection = snapshot_data.collection_name().to_string();
        let id = snapshot_data.id().to_string();
        self.ensure_dirs(&collection).await?;

        let _guard = self.lock.lock().await;

        let config = bincode::config::standard();
        let serialized = bincode::encode_to_vec(snapshot_data, config)
            .map_err(|e| SnapshotError::SerializationError(e.to_string()))?;
        let content_hash: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(&serialized);
            h.finalize().into()
        };

        let ranges = cdc_boundaries(&serialized, &cdc_params());
        let mut chunk_hashes = Vec::with_capacity(ranges.len());
        let mut total_bytes = 0u64; // bytes needed to reconstruct THIS snapshot (may include pre-existing chunks)
        for (start, end) in ranges {
            let chunk = &serialized[start..end];
            let hash = hash_bytes(chunk);
            let path = self.chunk_path(&collection, &hash);
            let compressed_len = if fs::try_exists(&path).await? {
                fs::metadata(&path).await?.len()
            } else {
                let compressed = gzip(chunk)?;
                let len = compressed.len() as u64;
                fs::write(&path, compressed).await?;
                len
            };
            total_bytes += compressed_len;
            chunk_hashes.push(hash);
        }

        let prev_root = self.read_chain_root(&collection).await?;
        let round = self.next_round(&collection).await?;
        let mut chain = WitnessChain::resume(prev_root);
        let manifest = chain.append(round, chunk_hashes.clone(), &serialized);

        let persisted = PersistedManifest {
            prev_root,
            round,
            chunk_hashes,
            content_hash: manifest.content_hash,
            chain_root: manifest.chain_root,
        };
        fs::write(self.manifest_path(&collection, &id), persisted.encode()).await?;
        self.write_chain_root(&collection, manifest.chain_root)
            .await?;

        let created_at = chrono::DateTime::parse_from_rfc3339(&snapshot_data.metadata.created_at)
            .map_err(|e| SnapshotError::storage(format!("Invalid timestamp: {e}")))?
            .with_timezone(&chrono::Utc);
        let snapshot = Snapshot {
            id: id.clone(),
            collection_name: collection.clone(),
            created_at,
            vectors_count: snapshot_data.vectors_count(),
            checksum: hex(&content_hash),
            size_bytes: total_bytes,
        };
        let metadata_json = serde_json::to_string_pretty(&snapshot)?;
        fs::write(self.meta_path(&collection, &id), metadata_json).await?;

        Ok(snapshot)
    }

    async fn load(&self, id: &str) -> Result<SnapshotData> {
        // The manifest/meta layout is per-collection but `load` is keyed
        // only by snapshot id (matching `SnapshotStorage`'s contract), so
        // scan collections for the one holding this id — acceptable for a
        // research-tier backend; a production version would index id ->
        // collection directly instead of scanning.
        let mut collections = fs::read_dir(&self.base_path).await?;
        while let Some(entry) = collections.next_entry().await? {
            if !entry.file_type().await?.is_dir() {
                continue;
            }
            let collection = entry.file_name().to_string_lossy().to_string();
            let manifest_path = self.manifest_path(&collection, id);
            if !fs::try_exists(&manifest_path).await? {
                continue;
            }
            let manifest_bytes = fs::read(&manifest_path).await?;
            let persisted = PersistedManifest::decode(&manifest_bytes)?;

            let mut store = ChunkStore::new();
            for hash in &persisted.chunk_hashes {
                let path = self.chunk_path(&collection, hash);
                let compressed = fs::read(&path).await.map_err(|_| {
                    SnapshotError::corrupted(format!("missing chunk {}", hex(hash)))
                })?;
                let bytes = gunzip(&compressed)?;
                store.put(&bytes);
            }

            let manifest = persisted.as_checkpoint_manifest();
            let reconstructed =
                witness::verify(&persisted.prev_root, &manifest, &store).map_err(|e| {
                    SnapshotError::corrupted(format!("CDC checkpoint failed verification: {e:?}"))
                })?;

            let config = bincode::config::standard();
            let (snapshot_data, _): (SnapshotData, usize) =
                bincode::decode_from_slice(&reconstructed, config)
                    .map_err(|e| SnapshotError::SerializationError(e.to_string()))?;
            return Ok(snapshot_data);
        }
        Err(SnapshotError::SnapshotNotFound(id.to_string()))
    }

    async fn list(&self) -> Result<Vec<Snapshot>> {
        let mut out = Vec::new();
        if !self.base_path.exists() {
            return Ok(out);
        }
        let mut collections = fs::read_dir(&self.base_path).await?;
        while let Some(entry) = collections.next_entry().await? {
            if !entry.file_type().await?.is_dir() {
                continue;
            }
            let meta_dir = entry.path().join("meta");
            if !meta_dir.exists() {
                continue;
            }
            let mut files = fs::read_dir(&meta_dir).await?;
            while let Some(f) = files.next_entry().await? {
                if f.path().extension().and_then(|e| e.to_str()) == Some("json") {
                    let contents = fs::read_to_string(f.path()).await?;
                    if let Ok(snapshot) = serde_json::from_str::<Snapshot>(&contents) {
                        out.push(snapshot);
                    }
                }
            }
        }
        out.sort_by(|a: &Snapshot, b: &Snapshot| b.created_at.cmp(&a.created_at));
        Ok(out)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let mut collections = fs::read_dir(&self.base_path).await?;
        while let Some(entry) = collections.next_entry().await? {
            if !entry.file_type().await?.is_dir() {
                continue;
            }
            let collection = entry.file_name().to_string_lossy().to_string();
            let manifest_path = self.manifest_path(&collection, id);
            if fs::try_exists(&manifest_path).await? {
                fs::remove_file(&manifest_path).await?;
                let meta_path = self.meta_path(&collection, id);
                if meta_path.exists() {
                    fs::remove_file(&meta_path).await?;
                }
                // Chunks are deliberately not removed here — see module
                // docs on garbage collection.
                return Ok(());
            }
        }
        Err(SnapshotError::SnapshotNotFound(id.to_string()))
    }
}
