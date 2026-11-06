//! A record ring whose cursors live in memory another process can map.
//!
//! A publisher and its readers in other processes need the head and the
//! per-slot sequence in the mapping itself, a capacity chosen in bytes, and a
//! read-only descriptor for each reader.
//!
//! * [`Geometry::fit`] turns a payload limit and a byte budget into a slot grid.
//! * [`RingWriter`] is the one writer. Its head and each slot's sequence live
//!   in the mapping.
//! * Each [`RingReader`] keeps its own cursor. A reader that falls behind
//!   skips to the oldest record that is still intact, so it cannot stop the writer.
//! * On Linux and macOS, `allocate` builds a shared-memory object and a
//!   read-only descriptor for the same bytes. Who receives that descriptor is
//!   the caller's decision.
//!
//! The ring does not name streams, choose tenants, or apply a backpressure
//! policy. Sequence numbers stop before `u64::MAX`, which marks a slot that
//! the writer is filling. 64-bit atomics must be lock-free. This build
//! accepts 64-bit targets, where they are. A 32-bit target is refused,
//! because its fallback lock would be private to one process.
//!
//! ```
//! use ruvector_shm_ring::Geometry;
//!
//! let geometry = Geometry::fit(640, 1024 * 1024).expect("two slots fit");
//! assert!(geometry.bytes() <= 1024 * 1024);
//! assert!(geometry.slot_count >= 2);
//! ```

#![deny(missing_docs)]

use std::fmt;

mod geom;
mod mem;
mod ring;

pub use geom::{Geometry, HEADER, SLOT_HEADER};
pub use mem::Mapping;
pub use ring::{Record, RingReader, RingWriter};

#[cfg(unix)]
pub use mem::{allocate, SharedMemory};

/// Why a ring operation was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RingError {
    /// 64-bit atomics on this target take a lock, so they are not safe in
    /// memory shared with another process.
    AtomicsNotLockFree,
    /// Slot count, slot size, or payload limit cannot describe a ring.
    BadGeometry,
    /// The mapping is shorter than the geometry.
    MappingTooSmall {
        /// Bytes the mapping covers.
        mapped: usize,
        /// Bytes the geometry needs.
        need: usize,
    },
    /// A writer needs a writable mapping.
    NotWritable,
    /// The mapping does not start with this crate's magic.
    BadMagic,
    /// The header version is not the version this crate writes.
    BadVersion(u64),
    /// The header's slot grid does not match the geometry passed to attach.
    HeaderMismatch,
    /// The payload is larger than the geometry's `max_record`.
    RecordTooLarge {
        /// Payload length that was refused.
        len: usize,
        /// Largest payload the ring accepts.
        max: u64,
    },
    /// The next sequence number would collide with the in-progress marker.
    SequenceExhausted,
}

impl fmt::Display for RingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AtomicsNotLockFree => {
                write!(f, "64-bit atomics are not lock-free on this target")
            }
            Self::BadGeometry => write!(f, "ring geometry is not usable"),
            Self::MappingTooSmall { mapped, need } => {
                write!(f, "mapping is {mapped} bytes, ring needs {need}")
            }
            Self::NotWritable => write!(f, "writer mapping is read-only"),
            Self::BadMagic => write!(f, "mapping is not a formatted ring"),
            Self::BadVersion(version) => write!(f, "ring version {version} is not supported"),
            Self::HeaderMismatch => write!(f, "ring header does not match the given geometry"),
            Self::RecordTooLarge { len, max } => {
                write!(f, "record is {len} bytes, max is {max}")
            }
            Self::SequenceExhausted => write!(f, "ring sequence space is exhausted"),
        }
    }
}

impl std::error::Error for RingError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(max_record: u64, ring_bytes: u64) -> (RingWriter, RingReader, Geometry) {
        let geometry = Geometry::fit(max_record, ring_bytes).unwrap();
        let map = Mapping::anonymous(geometry.bytes()).unwrap();
        let writer = RingWriter::format(map.share(), geometry, 7).unwrap();
        let reader = RingReader::attach(map, geometry).unwrap();
        (writer, reader, geometry)
    }

    #[test]
    fn geometry_fits_records_and_refuses_tiny_rings() {
        let geometry = Geometry::fit(640, 4 << 20).unwrap();
        assert_eq!(geometry.slot_size, 704);
        assert_eq!(geometry.slot_count, ((4 << 20) - 128) / 704);
        assert!(geometry.bytes() <= 4 << 20);
        assert_eq!(Geometry::fit(640, 128 + 704), None);
        assert_eq!(Geometry::fit(0, 4096), None);
        assert_eq!(Geometry::fit(1, 128 + 128).unwrap().slot_count, 2);
    }

    #[test]
    fn records_arrive_in_order_with_their_fields() {
        let (mut writer, reader, _) = ring(100, 64 << 10);
        let mut next = 1;
        assert!(reader.read(&mut next).is_none());
        for i in 1..=50u64 {
            let data: Vec<u8> = (0..(i % 13) as u8).collect();
            assert_eq!(writer.write(i * 10, 99, &data).unwrap(), i);
        }
        for i in 1..=50u64 {
            let record = reader.read(&mut next).unwrap();
            assert_eq!((record.seq, record.t_ns, record.w_ns), (i, i * 10, 99));
            assert_eq!(record.data, (0..(i % 13) as u8).collect::<Vec<_>>());
        }
        assert!(reader.read(&mut next).is_none());
    }

    #[test]
    fn a_reader_lapped_by_the_writer_jumps_forward() {
        let (mut writer, reader, geometry) = ring(8, 128 + 8 * 64);
        assert_eq!(geometry.slot_count, 8);
        for i in 1..=30u64 {
            writer.write(i, 0, &i.to_le_bytes()).unwrap();
        }
        let mut next = 1;
        let first = reader.read(&mut next).unwrap();
        assert_eq!(first.seq, 24);
        assert_eq!(first.data, 24u64.to_le_bytes());
        let rest: Vec<u64> = std::iter::from_fn(|| reader.read(&mut next))
            .map(|record| record.seq)
            .collect();
        assert_eq!(rest, (25..=30).collect::<Vec<_>>());
    }

    #[test]
    fn two_readers_keep_independent_cursors() {
        let (mut writer, reader_a, geometry) = ring(16, 64 << 10);
        let reader_b = RingReader::attach(reader_a.shared_mapping(), geometry).unwrap();
        for i in 1..=5u64 {
            writer.write(i, 0, &[i as u8]).unwrap();
        }
        let mut next_a = 1;
        let mut next_b = 1;
        assert_eq!(reader_a.read(&mut next_a).unwrap().seq, 1);
        assert_eq!(reader_a.read(&mut next_a).unwrap().seq, 2);
        assert_eq!(reader_b.read(&mut next_b).unwrap().seq, 1);
        assert_eq!(next_a, 3);
        assert_eq!(next_b, 2);
    }

    #[test]
    fn an_oversized_record_is_refused_and_a_second_writer_keeps_the_head() {
        let (mut writer, _, geometry) = ring(4, 64 << 10);
        let err = writer.write(0, 0, b"12345").unwrap_err();
        assert_eq!(err, RingError::RecordTooLarge { len: 5, max: 4 });
        assert_eq!(writer.head(), 0);
        writer.write(1, 0, b"ab").unwrap();
        let again = RingWriter::attach(writer_share(&writer), geometry).unwrap();
        assert_eq!(again.head(), 1);
        assert_eq!(again.epoch(), 2);
    }

    fn writer_share(writer: &RingWriter) -> Mapping {
        writer.shared_mapping()
    }

    #[test]
    fn a_reader_refuses_a_different_geometry() {
        let (writer, _, geometry) = ring(8, 64 << 10);
        let other = Geometry::from_parts(geometry.slot_count, geometry.slot_size, 32).unwrap();
        assert_ne!(other.max_record, geometry.max_record);
        let err = RingReader::attach(writer.shared_mapping(), other).unwrap_err();
        assert_eq!(err, RingError::HeaderMismatch);
    }

    #[test]
    fn concurrent_reader_never_sees_a_torn_record() {
        let (mut writer, reader, _) = ring(256, 128 + 16 * 320);
        let reader = std::sync::Arc::new(reader);
        let handle = {
            let reader = std::sync::Arc::clone(&reader);
            std::thread::spawn(move || {
                let (mut next, mut got, mut last) = (1u64, 0u64, 0u64);
                while last < 40_000 {
                    if let Some(record) = reader.read(&mut next) {
                        assert!(record.seq > last);
                        let fill = (record.seq % 251) as u8;
                        assert_eq!(record.data.len(), 200 + (record.seq % 56) as usize);
                        assert!(
                            record.data.iter().all(|byte| *byte == fill),
                            "torn record {}",
                            record.seq
                        );
                        assert_eq!(record.t_ns, record.seq);
                        last = record.seq;
                        got += 1;
                    }
                }
                got
            })
        };
        for seq in 1..=40_000u64 {
            let data = vec![(seq % 251) as u8; 200 + (seq % 56) as usize];
            writer.write(seq, 0, &data).unwrap();
        }
        assert!(handle.join().unwrap() > 0);
    }

    #[cfg(unix)]
    #[test]
    fn readers_cannot_map_the_ring_writable() {
        use std::os::fd::AsFd;
        let shared = allocate(64 << 10).unwrap();
        let err = Mapping::map(shared.read_only.as_fd(), 4096, true).unwrap_err();
        assert!(
            matches!(err.raw_os_error(), Some(libc::EACCES | libc::EPERM)),
            "{err}"
        );
        assert!(Mapping::map(shared.read_only.as_fd(), 4096, false).is_ok());
        assert!(Mapping::map(shared.read_only.as_fd(), 128 << 10, false).is_err());
    }
}
