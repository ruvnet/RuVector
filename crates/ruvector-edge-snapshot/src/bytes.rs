//! Little-endian byte cursor used by every decoder in this crate. All reads
//! are bounds-checked; nothing here panics on hostile input.

/// Bounds-checked little-endian reader.
pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub(crate) fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let s = self.buf.get(self.pos..end)?;
        self.pos = end;
        Some(s)
    }

    pub(crate) fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|s| s[0])
    }

    pub(crate) fn u16(&mut self) -> Option<u16> {
        self.take(2).map(|s| u16::from_le_bytes([s[0], s[1]]))
    }

    pub(crate) fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .and_then(|s| s.try_into().ok())
            .map(u32::from_le_bytes)
    }

    pub(crate) fn u64(&mut self) -> Option<u64> {
        self.take(8)
            .and_then(|s| s.try_into().ok())
            .map(u64::from_le_bytes)
    }

    pub(crate) fn arr32(&mut self) -> Option<[u8; 32]> {
        self.take(32).and_then(|s| s.try_into().ok())
    }

    /// `u16` length-prefixed UTF-8 string, at most `max` bytes.
    pub(crate) fn str16(&mut self, max: usize) -> Option<&'a str> {
        let n = usize::from(self.u16()?);
        if n > max {
            return None;
        }
        core::str::from_utf8(self.take(n)?).ok()
    }

    pub(crate) fn f32s_into(&mut self, n: usize, out: &mut Vec<f32>) -> Option<()> {
        let bytes = self.take(n.checked_mul(4)?)?;
        out.extend(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
        Some(())
    }
}

/// Append a `u16` length-prefixed string (caller guarantees `s.len() ≤ u16::MAX`).
pub(crate) fn put_str16(out: &mut Vec<u8>, s: &str) {
    let n = u16::try_from(s.len()).unwrap_or(u16::MAX);
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&s.as_bytes()[..usize::from(n)]);
}

/// Append little-endian `f32`s.
pub(crate) fn put_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}
