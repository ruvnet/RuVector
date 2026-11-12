//! Byte-level helpers: little-endian fixed ints and canonical LEB128 varints.
//! Every read is bounds-checked and returns `Corrupt`, never panics.

use crate::error::{AnalyticsError, CorruptKind, Result};

/// Append a canonical LEB128 `u64`.
pub(crate) fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Cursor over a byte slice.
pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

fn truncated() -> AnalyticsError {
    AnalyticsError::Corrupt(CorruptKind::Truncated)
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    pub(crate) fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(truncated)?;
        let s = self.buf.get(self.pos..end).ok_or_else(truncated)?;
        self.pos = end;
        Ok(s)
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16> {
        let b = self.bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        let mut a = [0u8; 4];
        a.copy_from_slice(self.bytes(4)?);
        Ok(u32::from_le_bytes(a))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        let mut a = [0u8; 8];
        a.copy_from_slice(self.bytes(8)?);
        Ok(u64::from_le_bytes(a))
    }

    /// Canonical LEB128: at most 10 bytes, no redundant trailing zero group,
    /// no bits above 64.
    pub(crate) fn varint(&mut self) -> Result<u64> {
        let enc = || AnalyticsError::Corrupt(CorruptKind::Encoding);
        let mut v: u64 = 0;
        for i in 0..10u32 {
            let byte = self.u8()?;
            let low = u64::from(byte & 0x7f);
            if i == 9 && low > 1 {
                return Err(enc());
            }
            v |= low << (7 * i);
            if byte & 0x80 == 0 {
                if i > 0 && byte == 0 {
                    return Err(enc());
                }
                return Ok(v);
            }
        }
        Err(enc())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip_and_canonical() {
        for v in [
            0u64,
            1,
            127,
            128,
            300,
            u32::MAX as u64,
            u64::MAX - 1,
            u64::MAX,
        ] {
            let mut b = Vec::new();
            put_varint(&mut b, v);
            let mut r = Reader::new(&b);
            assert_eq!(r.varint().unwrap(), v);
            assert_eq!(r.remaining(), 0);
        }
        // Overlong zero, 11-byte and >64-bit encodings are refused.
        for bad in [
            &[0x80u8, 0x00][..],
            &[0xff; 11][..],
            &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02][..],
        ] {
            assert!(Reader::new(bad).varint().is_err());
        }
        assert!(Reader::new(&[0x80]).varint().is_err());
    }
}
