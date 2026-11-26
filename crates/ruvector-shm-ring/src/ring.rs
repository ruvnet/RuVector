//! One writer, many readers, over a [`Mapping`](crate::Mapping).
//!
//! ```text
//! header (128 bytes)
//!   0  magic "RVRING01"    8  version (1)     16 slot_count
//!   24 slot_size          32 max_record      40 ring_id
//!   64 head: seq of the newest committed record (0 means none yet)
//!   72 writer_epoch: incremented each time a writer attaches
//! slot i at 128 + i * slot_size
//!   0  seq (0 = empty, u64::MAX while the writer is filling the slot)
//!   8  len   16 t_ns   24 w_ns
//!   32 payload
//! ```
//!
//! Record `seq` lives in slot `(seq - 1) % slot_count`. The writer stores
//! `u64::MAX` in that word, writes the fields, stores `seq`, then stores
//! `head`. A reader copies the slot and keeps the copy only when the seq word
//! matches before and after the copy.

use std::sync::atomic::{fence, Ordering};

use crate::geom::{Geometry, SLOT_HEADER};
use crate::mem::Mapping;
use crate::RingError;

const MAGIC: u64 = u64::from_le_bytes(*b"RVRING01");
const VERSION: u64 = 1;
/// Marks a slot whose payload is being written. Never a committed seq.
const BUSY: u64 = u64::MAX;
const OFF_HEAD: usize = 64;
const OFF_EPOCH: usize = 72;

/// One committed record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// Sequence number, starting at 1. A gap means the slot was overwritten.
    pub seq: u64,
    /// Timestamp supplied by the writer. The ring does not interpret it.
    pub t_ns: u64,
    /// Second timestamp supplied by the writer.
    pub w_ns: u64,
    /// Payload. At most `max_record` bytes.
    pub data: Vec<u8>,
}

/// The single writer of a ring.
#[derive(Debug)]
pub struct RingWriter {
    map: Mapping,
    geometry: Geometry,
    head: u64,
}

/// A reader. Each reader keeps its own cursor and does not disturb the others.
#[derive(Debug)]
pub struct RingReader {
    map: Mapping,
    geometry: Geometry,
}

impl RingWriter {
    /// Format `map` as an empty ring and attach as its writer.
    ///
    /// `map` must be writable and at least [`Geometry::bytes`] long. Existing
    /// bytes in the mapping are cleared.
    pub fn format(map: Mapping, geometry: Geometry, ring_id: u64) -> Result<Self, RingError> {
        check_atomics()?;
        if !geometry.valid() {
            return Err(RingError::BadGeometry);
        }
        if !map.writable() {
            return Err(RingError::NotWritable);
        }
        if map.len() < geometry.bytes() {
            return Err(RingError::MappingTooSmall {
                mapped: map.len(),
                need: geometry.bytes(),
            });
        }
        for off in (0..geometry.bytes()).step_by(8) {
            map.word(off).store(0, Ordering::Relaxed);
        }
        for (off, value) in [
            (0, MAGIC),
            (8, VERSION),
            (16, geometry.slot_count),
            (24, geometry.slot_size),
            (32, geometry.max_record),
            (40, ring_id),
            (OFF_HEAD, 0),
            (OFF_EPOCH, 0),
        ] {
            map.word(off).store(value, Ordering::Relaxed);
        }
        fence(Ordering::Release);
        Self::attach(map, geometry)
    }

    /// Attach to a ring that [`Self::format`] already prepared.
    ///
    /// Increments the writer epoch and continues from the current head.
    /// Does not clear committed records. A second attach does not stop an
    /// earlier writer. Two live writers corrupt the ring.
    pub fn attach(map: Mapping, geometry: Geometry) -> Result<Self, RingError> {
        check_header(&map, &geometry, true)?;
        let head = map.word(OFF_HEAD).load(Ordering::Acquire);
        map.word(OFF_EPOCH).fetch_add(1, Ordering::AcqRel);
        Ok(Self {
            map,
            geometry,
            head,
        })
    }

    /// Sequence of the newest record this writer has committed, or 0.
    pub fn head(&self) -> u64 {
        self.head
    }

    /// Geometry this writer was attached with.
    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    #[cfg(test)]
    pub(crate) fn shared_mapping(&self) -> Mapping {
        self.map.share()
    }

    /// How many times a writer has attached, including this one.
    pub fn epoch(&self) -> u64 {
        self.map.word(OFF_EPOCH).load(Ordering::Acquire)
    }

    /// Commit one record and return its sequence number.
    pub fn write(&mut self, t_ns: u64, w_ns: u64, data: &[u8]) -> Result<u64, RingError> {
        if data.len() as u64 > self.geometry.max_record {
            return Err(RingError::RecordTooLarge {
                len: data.len(),
                max: self.geometry.max_record,
            });
        }
        if self.head >= BUSY - 1 {
            return Err(RingError::SequenceExhausted);
        }
        let seq = self.head + 1;
        let base = self.geometry.slot(seq);
        let map = &self.map;
        map.word(base).store(BUSY, Ordering::Relaxed);
        fence(Ordering::Release);
        map.word(base + 8)
            .store(data.len() as u64, Ordering::Relaxed);
        map.word(base + 16).store(t_ns, Ordering::Relaxed);
        map.word(base + 24).store(w_ns, Ordering::Relaxed);
        for (i, chunk) in data.chunks(8).enumerate() {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            map.word(base + SLOT_HEADER + i * 8)
                .store(u64::from_le_bytes(word), Ordering::Relaxed);
        }
        map.word(base).store(seq, Ordering::Release);
        map.word(OFF_HEAD).store(seq, Ordering::Release);
        self.head = seq;
        Ok(seq)
    }
}

impl RingReader {
    /// Attach to a formatted ring.
    ///
    /// `geometry` must be the geometry the writer formatted. The header is
    /// checked against it, and the mapping must cover [`Geometry::bytes`].
    pub fn attach(map: Mapping, geometry: Geometry) -> Result<Self, RingError> {
        check_header(&map, &geometry, false)?;
        Ok(Self { map, geometry })
    }

    /// Newest committed sequence, or 0 when the writer has not committed.
    pub fn head(&self) -> u64 {
        self.map.word(OFF_HEAD).load(Ordering::Acquire)
    }

    /// Geometry this reader was attached with.
    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    /// Read the next record at `*next`, or the oldest record still safe to read.
    ///
    /// `*next` starts at 1. On success it advances past the returned record.
    /// A reader that has fallen more than `slot_count - 2` records behind moves
    /// forward, and the returned `seq` is the first one still in the ring.
    ///
    /// `None` means there is no newer committed record, or this attempt did not
    /// observe a stable slot. A caller that still sees [`Self::head`] at or
    /// past `*next` should call again.
    pub fn read(&self, next: &mut u64) -> Option<Record> {
        let slots = self.geometry.slot_count;
        let bound = slots.saturating_mul(2).saturating_add(16) as usize;
        for _ in 0..bound {
            let head = self.head();
            if head < *next {
                return None;
            }
            let oldest = (head + 2).saturating_sub(slots).max(1);
            if *next < oldest {
                *next = oldest;
            }
            let base = self.geometry.slot(*next);
            let map = &self.map;
            let before = map.word(base).load(Ordering::Acquire);
            if before != *next {
                if before < *next {
                    *next += 1;
                }
                continue;
            }
            let len = map.word(base + 8).load(Ordering::Relaxed);
            let t_ns = map.word(base + 16).load(Ordering::Relaxed);
            let w_ns = map.word(base + 24).load(Ordering::Relaxed);
            let len = usize::try_from(len.min(self.geometry.max_record)).unwrap_or(0);
            let mut data = Vec::with_capacity(len.div_ceil(8) * 8);
            for i in 0..len.div_ceil(8) {
                let word = map.word(base + SLOT_HEADER + i * 8).load(Ordering::Relaxed);
                data.extend_from_slice(&word.to_le_bytes());
            }
            data.truncate(len);
            fence(Ordering::Acquire);
            if map.word(base).load(Ordering::Relaxed) != before {
                continue;
            }
            let seq = *next;
            *next += 1;
            return Some(Record {
                seq,
                t_ns,
                w_ns,
                data,
            });
        }
        None
    }
}

fn check_atomics() -> Result<(), RingError> {
    // Stable Rust has no AtomicU64::is_lock_free. On 64-bit targets the
    // hardware atomic is lock-free. On 32-bit targets the library lock is
    // private to the process, so a second process must not share it.
    if cfg!(target_pointer_width = "64") {
        Ok(())
    } else {
        Err(RingError::AtomicsNotLockFree)
    }
}

fn check_header(map: &Mapping, geometry: &Geometry, require_write: bool) -> Result<(), RingError> {
    check_atomics()?;
    if !geometry.valid() {
        return Err(RingError::BadGeometry);
    }
    if require_write && !map.writable() {
        return Err(RingError::NotWritable);
    }
    if map.len() < geometry.bytes() {
        return Err(RingError::MappingTooSmall {
            mapped: map.len(),
            need: geometry.bytes(),
        });
    }
    let magic = map.word(0).load(Ordering::Acquire);
    if magic != MAGIC {
        return Err(RingError::BadMagic);
    }
    let version = map.word(8).load(Ordering::Relaxed);
    if version != VERSION {
        return Err(RingError::BadVersion(version));
    }
    let slot_count = map.word(16).load(Ordering::Relaxed);
    let slot_size = map.word(24).load(Ordering::Relaxed);
    let max_record = map.word(32).load(Ordering::Relaxed);
    if slot_count != geometry.slot_count
        || slot_size != geometry.slot_size
        || max_record != geometry.max_record
    {
        return Err(RingError::HeaderMismatch);
    }
    Ok(())
}

impl RingReader {
    #[cfg(test)]
    pub(crate) fn shared_mapping(&self) -> Mapping {
        self.map.share()
    }
}
