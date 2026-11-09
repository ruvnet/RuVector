//! Byte budget of a ring, turned into a fixed slot grid.

/// Bytes before the first slot.
pub const HEADER: usize = 128;
/// Bytes of bookkeeping at the start of every slot.
pub const SLOT_HEADER: usize = 32;

/// Shape of a ring. Fixed when the ring is formatted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// Number of slots. At least 2.
    pub slot_count: u64,
    /// Bytes per slot, including [`SLOT_HEADER`]. A multiple of 64.
    pub slot_size: u64,
    /// Largest payload, in bytes.
    pub max_record: u64,
}

impl Geometry {
    /// Fit `max_record` payloads into `ring_bytes`.
    ///
    /// Returns `None` when `max_record` is 0 or when fewer than two slots fit.
    /// The header takes 128 bytes. Each slot is rounded up to 64 bytes.
    pub fn fit(max_record: u64, ring_bytes: u64) -> Option<Self> {
        if max_record == 0 {
            return None;
        }
        let payload = max_record.checked_add(7)? / 8 * 8;
        let raw = (SLOT_HEADER as u64).checked_add(payload)?;
        let slot_size = raw.div_ceil(64).checked_mul(64)?;
        if slot_size == 0 {
            return None;
        }
        let usable = ring_bytes.checked_sub(HEADER as u64)?;
        let slot_count = usable / slot_size;
        let geometry = Self {
            slot_count,
            slot_size,
            max_record,
        };
        geometry.valid().then_some(geometry)
    }

    /// Rebuild a geometry from parts already stored in a header.
    ///
    /// Returns `None` when the parts could not have come from [`Self::fit`].
    pub fn from_parts(slot_count: u64, slot_size: u64, max_record: u64) -> Option<Self> {
        let geometry = Self {
            slot_count,
            slot_size,
            max_record,
        };
        geometry.valid().then_some(geometry)
    }

    /// Bytes a mapping must cover.
    ///
    /// Returns 0 when the slot grid does not fit in `usize`. The writer
    /// refuses that geometry.
    pub fn bytes(&self) -> usize {
        self.byte_len().unwrap_or(0)
    }

    fn byte_len(&self) -> Option<usize> {
        let slots = self.slot_count.checked_mul(self.slot_size)?;
        usize::try_from((HEADER as u64).checked_add(slots)?).ok()
    }

    pub(crate) fn valid(&self) -> bool {
        self.slot_count >= 2
            && self.max_record > 0
            && self.slot_size.is_multiple_of(64)
            && self.slot_size >= SLOT_HEADER as u64 + self.max_record.div_ceil(8) * 8
            && self.byte_len().is_some_and(|n| n > HEADER)
    }

    pub(crate) fn slot(&self, seq: u64) -> usize {
        let index = (seq - 1) % self.slot_count;
        let off = index
            .checked_mul(self.slot_size)
            .and_then(|n| usize::try_from(n).ok())
            .expect("slot offset fits when the geometry is valid");
        HEADER + off
    }
}
