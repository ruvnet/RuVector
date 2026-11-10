//! Policy-aware constructors and I/O hooks of [`RvfStore`] (ADR-033 section 4,
//! ADR-354 item 5). State and checks live in the sibling `security` module.

use super::security::{
    check_signed_root, read_trailing_page, CommitMismatch, Committed, LastManifest, SecurityState,
};
use super::*;
use rvf_types::security::{SecurityError, SecurityPolicy};
use rvf_wire::hash::compute_shake256_128;
use std::io;

use crate::root_page;

impl RvfStore {
    /// Create a store whose every commit ends in a signed Level-0 root page.
    ///
    /// All segments are hashed with SHAKE-256-128 (never the fast XXH3 hash,
    /// which is integrity only), and each manifest carries a per-segment hash
    /// trailer that the signed page commits to. `signing_key` is an Ed25519
    /// secret key (ADR-033 names ML-DSA-65; only Ed25519 exists in this
    /// codebase today). Fails closed without the `ed25519` feature.
    ///
    /// Crash behaviour: a signed commit is "segments, manifest, then page".
    /// A crash before the page is written leaves a file that opens under
    /// `Permissive` and is refused (`UnsignedManifest`) under `Strict`.
    pub fn create_signed(
        path: &Path,
        options: RvfOptions,
        signing_key: &[u8; 32],
    ) -> Result<Self, RvfError> {
        if !cfg!(feature = "ed25519") {
            return Err(err(ErrorCode::InvalidSignature));
        }
        Self::create_inner(path, options, Some(*signing_key))
    }

    /// Open under an explicit security policy (ADR-033 section 4.5).
    ///
    /// Trust anchors come from `options.trusted_signers`. `Permissive` is
    /// exactly [`RvfStore::open`]: read-write, takes the writer lock, and
    /// ignores `options` entirely. Every other policy opens **read-only**
    /// (no lock; the asymmetry is deliberate): a
    /// writer without the signing key would break the signed chain; use
    /// [`RvfStore::open_signed`] to write.
    ///
    /// Failures are `RvfError::Security(..)` with the structured fields
    /// (offset, phase, hashes) and are logged at WARN.
    pub fn open_with_policy(
        path: &Path,
        options: RvfOptions,
        policy: SecurityPolicy,
    ) -> Result<Self, RvfError> {
        if policy == SecurityPolicy::Permissive {
            return Self::open(path);
        }
        let mut state = SecurityState::with(policy, options.trusted_signers.clone(), None);
        state.freshness = options.freshness;
        state.serve_signed_prefix = options.serve_signed_prefix;
        Self::open_readonly_inner(path, &mut HashSet::new(), state)
    }

    /// [`RvfStore::open_with_policy`] using `options.security_policy` (which
    /// defaults to `Strict`).
    pub fn open_with_options(path: &Path, options: RvfOptions) -> Result<Self, RvfError> {
        let policy = options.security_policy;
        Self::open_with_policy(path, options, policy)
    }

    /// Open a signed store for writing. Requires a policy that checks
    /// signatures; every later commit re-signs with `signing_key`.
    pub fn open_signed(
        path: &Path,
        options: RvfOptions,
        policy: SecurityPolicy,
        signing_key: &[u8; 32],
    ) -> Result<Self, RvfError> {
        if !cfg!(feature = "ed25519") || !policy.requires_signature() {
            return Err(err(ErrorCode::InvalidSignature));
        }
        let mut state =
            SecurityState::with(policy, options.trusted_signers.clone(), Some(*signing_key));
        state.freshness = options.freshness;
        state.serve_signed_prefix = options.serve_signed_prefix;
        Self::open_inner(path, state)
    }

    /// True if any legacy-dialect (pre ADR-354) layout was encountered:
    /// unpadded segments, a CRC-rotation hash labelled `checksum_algo = 0`, or
    /// a MANIFEST segment stamped `checksum_algo = 0`. Under `Strict` a legacy
    /// file never opens, so this is mainly an audit/migration signal.
    pub fn legacy_dialect(&self) -> bool {
        self.security.legacy_dialect.load(Ordering::Relaxed)
    }

    /// True if the file ends in a signed Level-0 root page that names the
    /// manifest in use (CRC-valid and self-consistent; **not** verified
    /// against any key unless the store was opened with a signature-checking
    /// policy). Lets a `Permissive` consumer notice it is handling a signed
    /// artifact it is not verifying.
    pub fn has_signed_root_page(&self) -> bool {
        self.security.root_page_present.load(Ordering::Acquire)
    }

    /// Under a signature-checking policy opened with
    /// `RvfOptions::serve_signed_prefix`: the newest parseable manifest found
    /// beyond the signed page, `(offset, epoch)`. It was NOT used; the store
    /// serves the signed state. `None` when there is no such manifest.
    /// True if unsigned bytes (of any shape) follow the signed manifest and
    /// were served past under `serve_signed_prefix`.
    pub fn unsigned_tail_present(&self) -> bool {
        self.security.unsigned_tail.is_some()
    }

    pub fn newer_manifest_ignored(&self) -> Option<(u64, u32)> {
        self.security.newer_ignored
    }

    /// The policy this store was opened under.
    pub fn security_policy(&self) -> SecurityPolicy {
        self.security.policy
    }

    /// True when this handle must not write: read-only, or it holds no signing
    /// key yet the file ends in a valid signed root page that a write would
    /// have to cut (see [`RvfStore::allow_unsigning`]).
    pub(super) fn is_write_blocked(&self) -> bool {
        self.read_only
            || (self.security.root_page_present.load(Ordering::Acquire)
                && self.security.signing_key.is_none()
                && !self.security.allow_unsigning)
    }

    /// Consent for a handle that cannot re-sign (opened with `open()`, i.e.
    /// `Permissive`) to write to a store whose file ends in a signed root
    /// page. The write cuts the page, so the file is **unsigned from then
    /// on** and `Strict` readers refuse it. Without this call such a handle
    /// refuses to write (`ReadOnly`); use [`RvfStore::open_signed`] to keep
    /// the signature.
    pub fn allow_unsigning(&mut self, allow: bool) {
        self.security.allow_unsigning = allow;
    }

    /// Writer cursor positioned at EOF. A trailing root page (ours) is cut
    /// first so new segments never land behind it; the next commit re-seals.
    pub(super) fn append_cursor<'a>(
        file: &'a File,
        security: &SecurityState,
    ) -> Result<BufWriter<&'a File>, RvfError> {
        if security.root_page_present.swap(false, Ordering::AcqRel) {
            let len = file
                .metadata()
                .map_err(|_| err(ErrorCode::FsyncFailed))?
                .len();
            file.set_len(len.saturating_sub(root_page::PAGE_SIZE as u64))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
        }
        let mut w = BufWriter::new(file);
        w.seek(SeekFrom::End(0))
            .map_err(|_| err(ErrorCode::FsyncFailed))?;
        Ok(w)
    }

    /// Read a segment payload, applying the policy's committed-hash rule.
    pub(super) fn read_seg<R: Read + Seek>(
        &self,
        reader: &mut R,
        offset: u64,
    ) -> io::Result<(rvf_types::SegmentHeader, Vec<u8>)> {
        let (header, payload) = read_path::read_segment_payload(reader, offset)?;
        if header.checksum_algo == 0 {
            self.security.legacy_dialect.store(true, Ordering::Relaxed);
        }
        if let Some(committed) = &self.security.committed {
            let expected = committed.get(&offset).copied();
            if !expected.is_some_and(|c| c.header_ok(&header)) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    CommitMismatch {
                        offset,
                        expected: expected.map(|c| c.hash),
                        actual: header.content_hash,
                    },
                ));
            }
        }
        Ok((header, payload))
    }

    /// Convert a failed [`Self::read_seg`] into the right `RvfError`: a
    /// committed-hash mismatch becomes the structured security error.
    pub(super) fn map_read_error(e: &io::Error, fallback: ErrorCode) -> RvfError {
        if let Some(m) = e.get_ref().and_then(|i| i.downcast_ref::<CommitMismatch>()) {
            return RvfError::Security(SecurityError::ContentHashMismatch {
                pointer_name: "segment_content_hash",
                expected_hash: m.expected.unwrap_or([0; 16]),
                actual_hash: m.actual,
                seg_offset: m.offset,
            });
        }
        err(fallback)
    }

    /// Policy check run by `boot` once the manifest is located. Sets up
    /// committed-hash enforcement.
    pub(super) fn apply_open_policy(
        &mut self,
        located: &read_path::LocatedManifest,
    ) -> Result<(), RvfError> {
        let manifest = &located.manifest;
        if located.mode == rvf_wire::tail_scan::ScanMode::ByteGranular
            || located.header.checksum_algo == 0
            || manifest
                .segment_dir
                .iter()
                .any(|e| e.offset % rvf_types::SEGMENT_ALIGNMENT as u64 != 0)
        {
            self.security.legacy_dialect.store(true, Ordering::Relaxed);
        }

        let policy = self.security.policy;
        if policy == SecurityPolicy::Permissive {
            // Verify nothing, but still recognise a root page this crate
            // wrote (valid CRC, names the manifest we found) so a read-write
            // `open()` of a signed file cuts it before appending instead of
            // leaving a stale page in the middle of the file. The file is
            // unsigned from that write on; that is what Permissive means.
            if let Some(page) = read_trailing_page(&self.file) {
                let v = root_page::view(&page);
                if v.crc_ok && v.sig_algo != 0 && v.l1_offset == located.offset {
                    self.security
                        .root_page_present
                        .store(true, Ordering::Release);
                }
            }
            return Ok(());
        }
        if let Err(e) = check_signed_root(&self.file, &self.security.trusted_signers, located) {
            log::warn!(
                "rvf: {} policy rejected {}: {e}",
                policy_name(policy),
                self.path.display()
            );
            if policy.requires_signature() {
                return Err(RvfError::Security(e));
            }
            // WarnOnly: continue; committed hashes (if the manifest has them)
            // are still enforced below.
        } else {
            self.security
                .root_page_present
                .store(true, Ordering::Release);
            if policy.requires_signature() {
                // A signed page proves authenticity, not freshness (F3-02).
                match self.security.freshness {
                    None => return Err(err(ErrorCode::FreshnessRequired)),
                    Some(crate::options::Freshness::MinEpoch(m)) if located.manifest.epoch < m => {
                        return Err(err(ErrorCode::StreamStaleEpoch));
                    }
                    _ => {}
                }
                if let Some(off) = located.unsigned_tail {
                    let epoch = located.newer_ignored.map_or(0, |n| n.1);
                    if !self.security.serve_signed_prefix {
                        // An unsigned tail: interrupted commit, planted manifest or
                        // a page swapped back over newer data. Refuse unless the
                        // caller opted in to serving the signed prefix.
                        return Err(RvfError::Security(SecurityError::ContentHashMismatch {
                            pointer_name: "newer_manifest_beyond_signed_page",
                            expected_hash: compute_shake256_128(&located.payload),
                            actual_hash: [0; 16],
                            seg_offset: off,
                        }));
                    }
                    log::warn!(
                        "rvf: serving the signed state (epoch {}) although a newer manifest (epoch {epoch}) follows the signed page at 0x{off:X}",
                        located.manifest.epoch
                    );
                    self.security.unsigned_tail = Some(off);
                    self.security.newer_ignored = located.newer_ignored;
                }
            }
        }

        if let Some(entries) = &manifest.seg_hashes {
            let by_id: HashMap<u64, [u8; 16]> = entries.iter().copied().collect();
            let mut by_offset = HashMap::new();
            for e in &manifest.segment_dir {
                if let Some(h) = by_id.get(&e.seg_id) {
                    by_offset.insert(
                        e.offset,
                        Committed {
                            hash: *h,
                            seg_id: e.seg_id,
                            seg_type: e.seg_type,
                            payload_len: e.payload_length,
                        },
                    );
                }
            }
            self.security.committed = Some(by_offset);
        }
        if self.security.signing_key.is_some() {
            self.security.last_manifest = Some(LastManifest {
                seg_id: located.header.segment_id,
                offset: located.offset,
                len: SEGMENT_HEADER_SIZE as u64 + located.payload.len() as u64,
                hash: compute_shake256_128(&located.payload),
            });
        }
        Ok(())
    }

    /// Compaction input for a signing handle (F-01, F-02).
    ///
    /// Returns the file body and the segments to carry over, taken from the
    /// segment directory this handle holds and nothing else. Every carried
    /// segment must match the hash this handle committed to (recorded when it
    /// wrote the segment, or loaded from the verified manifest at open) in
    /// payload and in every header field; one mismatch refuses the whole
    /// compaction before any state changes. Segments that are in the file but
    /// not in the directory are not carried, and a hash is never recomputed
    /// from bytes read back from the file.
    #[allow(clippy::type_complexity)]
    pub(super) fn verified_compaction_sources(
        &self,
    ) -> Result<(Vec<u8>, Vec<(usize, u64, u64, u8, [u8; 16])>), RvfError> {
        let mut bytes = Vec::new();
        {
            let mut r = BufReader::new(&self.file);
            r.seek(SeekFrom::Start(0))
                .and_then(|_| r.read_to_end(&mut bytes))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
        }
        let body = rvf_wire::tail_scan::split_root_page(&bytes).0.len();
        bytes.truncate(body);
        let writer = self
            .seg_writer
            .as_ref()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
        let skip = [
            SegmentType::Vec as u8,
            SegmentType::Manifest as u8,
            SegmentType::Journal as u8,
            SegmentType::Index as u8,
            SegmentType::Meta as u8,
        ];
        let mismatch = |offset: u64, expected: [u8; 16], actual: [u8; 16]| {
            RvfError::Security(SecurityError::ContentHashMismatch {
                pointer_name: "compaction_source",
                expected_hash: expected,
                actual_hash: actual,
                seg_offset: offset,
            })
        };
        let mut out = Vec::new();
        for &(id, off, plen, ty) in &self.segment_dir {
            if skip.contains(&ty) {
                continue;
            }
            let committed = writer
                .hash_of(id)
                .ok_or_else(|| mismatch(off, [0; 16], [0; 16]))?;
            let start = usize::try_from(off).map_err(|_| mismatch(off, committed, [0; 16]))?;
            let end = usize::try_from(plen)
                .ok()
                .and_then(|p| p.checked_add(SEGMENT_HEADER_SIZE))
                .and_then(|t| start.checked_add(t))
                .filter(|&e| e <= bytes.len() && off % rvf_types::SEGMENT_ALIGNMENT as u64 == 0)
                .ok_or_else(|| mismatch(off, committed, [0; 16]))?;
            let header = rvf_wire::read_segment_header(&bytes[start..end])
                .map_err(|_| mismatch(off, committed, [0; 16]))?;
            let actual = compute_shake256_128(&bytes[start + SEGMENT_HEADER_SIZE..end]);
            let c = Committed {
                hash: committed,
                seg_id: id,
                seg_type: ty,
                payload_len: plen,
            };
            if actual != committed || !c.header_ok(&header) {
                return Err(mismatch(off, committed, actual));
            }
            out.push((start, id, plen, ty, committed));
        }
        Ok((bytes, out))
    }

    /// Signed stores only: rebuild the committed-hash map from the directory
    /// and the writer's recorded hashes, so segments written (or moved by
    /// compaction) during this session read back under enforcement.
    pub(super) fn refresh_committed(&mut self) {
        if self.security.signing_key.is_none() {
            return;
        }
        let Some(writer) = self.seg_writer.as_ref() else {
            return;
        };
        let map = self
            .segment_dir
            .iter()
            .filter_map(|&(id, offset, plen, ty)| {
                writer.hash_of(id).map(|h| {
                    (
                        offset,
                        Committed {
                            hash: h,
                            seg_id: id,
                            seg_type: ty,
                            payload_len: plen,
                        },
                    )
                })
            })
            .collect();
        self.security.committed = Some(map);
    }

    /// `Paranoid`: read every directory segment once so a bad hash fails the
    /// open instead of a later query.
    pub(super) fn verify_all_segments(&self) -> Result<(), RvfError> {
        if self.security.policy != SecurityPolicy::Paranoid {
            return Ok(());
        }
        for &(_, offset, _, _) in &self.segment_dir {
            let mut reader = BufReader::new(&self.file);
            self.read_seg(&mut reader, offset)
                .map_err(|e| Self::map_read_error(&e, ErrorCode::InvalidChecksum))?;
        }
        Ok(())
    }

    /// Append the signed root page for the last committed manifest.
    pub(super) fn seal_root_page(&mut self) -> Result<(), RvfError> {
        let Some(key) = self.security.signing_key else {
            return Ok(());
        };
        #[cfg(feature = "ed25519")]
        {
            let last = self
                .security
                .last_manifest
                .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
            let live =
                (self.vectors.len() as u64).saturating_sub(self.deletion_bitmap.count() as u64);
            let page = root_page::build_signed_page(
                &root_page::PageContent {
                    l1_offset: last.offset,
                    l1_length: last.len,
                    l1_content_hash: last.hash,
                    l1_segment_id: last.seg_id,
                    total_vectors: live,
                    dimension: self.options.dimension,
                    profile: self.options.profile,
                    epoch: self.epoch,
                    identity: &self.file_identity,
                },
                &key,
            );
            let mut w = Self::append_cursor(&self.file, &self.security)?;
            w.write_all(&page)
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            w.flush().map_err(|_| err(ErrorCode::FsyncFailed))?;
            drop(w);
            self.file
                .sync_all()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            self.security
                .root_page_present
                .store(true, Ordering::Release);
            Ok(())
        }
        #[cfg(not(feature = "ed25519"))]
        {
            let _ = key;
            Err(err(ErrorCode::InvalidSignature))
        }
    }

    /// Seed a freshly created `SegmentWriter` for this store's security mode.
    pub(super) fn configure_writer(
        &self,
        writer: &mut SegmentWriter,
        hashes: Option<&[(u64, [u8; 16])]>,
    ) {
        if self.security.signing_key.is_some() {
            writer.use_cryptographic_hash();
            for (id, h) in hashes.unwrap_or(&[]) {
                writer.record_hash(*id, *h);
            }
        }
    }
}

fn policy_name(p: SecurityPolicy) -> &'static str {
    match p {
        SecurityPolicy::Permissive => "Permissive",
        SecurityPolicy::WarnOnly => "WarnOnly",
        SecurityPolicy::Strict => "Strict",
        SecurityPolicy::Paranoid => "Paranoid",
    }
}
