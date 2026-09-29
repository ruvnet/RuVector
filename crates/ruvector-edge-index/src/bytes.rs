//! Little-endian payload writer/reader. The reader pulls chunk bodies one
//! at a time from a [`Source`] (never a joined copy) and checks every
//! length against the *declared* payload bytes remaining before
//! allocating; every length product is `checked_mul`, so a hostile count
//! cannot wrap on wasm32 or force a huge allocation.

use crate::error::DecodeError;

const BUF: usize = 4096;

/// Byte sink for payload encoders.
pub(crate) trait Sink {
    fn put(&mut self, b: &[u8]);

    fn put_u8(&mut self, v: u8) {
        self.put(&[v]);
    }
    fn put_u16(&mut self, v: u16) {
        self.put(&v.to_le_bytes());
    }
    fn put_u32(&mut self, v: u32) {
        self.put(&v.to_le_bytes());
    }
    fn put_u64(&mut self, v: u64) {
        self.put(&v.to_le_bytes());
    }
    fn put_u32s(&mut self, v: &[u32]) {
        let mut buf = [0u8; BUF];
        for chunk in v.chunks(BUF / 4) {
            for (i, x) in chunk.iter().enumerate() {
                buf[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
            self.put(&buf[..chunk.len() * 4]);
        }
    }
    fn put_u64s(&mut self, v: &[u64]) {
        for x in v {
            self.put(&x.to_le_bytes());
        }
    }
    fn put_f32s(&mut self, v: &[f32]) {
        for x in v {
            self.put(&x.to_bits().to_le_bytes());
        }
    }
}

impl Sink for Vec<u8> {
    fn put(&mut self, b: &[u8]) {
        self.extend_from_slice(b);
    }
}

/// Sink that only counts, so encoders can declare the payload length up
/// front from the very routine that writes it.
pub(crate) struct Counter(pub u64);

impl Sink for Counter {
    fn put(&mut self, b: &[u8]) {
        self.0 += b.len() as u64;
    }
    fn put_u32s(&mut self, v: &[u32]) {
        self.0 += 4 * v.len() as u64;
    }
}

/// Supplier of payload bodies, validated as they are pulled.
pub(crate) trait Source {
    /// Next body as `(buffer, body start)`, or `None` when exhausted.
    fn next_body(&mut self) -> Result<Option<(Vec<u8>, usize)>, DecodeError>;
}

const TRUNC: DecodeError = DecodeError::Malformed("payload shorter than its fields");
const OVERFLOW: DecodeError = DecodeError::Malformed("length overflow");

/// Streaming reader over a [`Source`] with a declared payload length.
pub(crate) struct Reader<'s> {
    src: &'s mut dyn Source,
    cur: Vec<u8>,
    off: usize,
    remaining: usize,
}

impl<'s> Reader<'s> {
    pub fn new(src: &'s mut dyn Source, declared: usize) -> Self {
        Self {
            src,
            cur: Vec::new(),
            off: 0,
            remaining: declared,
        }
    }

    pub fn take_into(&mut self, out: &mut [u8]) -> Result<(), DecodeError> {
        if out.len() > self.remaining {
            return Err(TRUNC);
        }
        let mut done = 0;
        while done < out.len() {
            if self.off >= self.cur.len() {
                self.cur = Vec::new(); // free the spent row before pulling the next
                let (b, start) = self.src.next_body()?.ok_or(TRUNC)?;
                self.cur = b;
                self.off = start;
                continue;
            }
            let n = (self.cur.len() - self.off).min(out.len() - done);
            out[done..done + n].copy_from_slice(&self.cur[self.off..self.off + n]);
            done += n;
            self.off += n;
        }
        self.remaining -= out.len();
        Ok(())
    }

    fn arr<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        let mut b = [0u8; N];
        self.take_into(&mut b)?;
        Ok(b)
    }

    pub fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.arr::<1>()?[0])
    }
    pub fn u16(&mut self) -> Result<u16, DecodeError> {
        Ok(u16::from_le_bytes(self.arr()?))
    }
    pub fn u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    pub fn u64(&mut self) -> Result<u64, DecodeError> {
        Ok(u64::from_le_bytes(self.arr()?))
    }

    /// Element count `count · per`, checked against overflow and against
    /// the declared bytes left (`width` bytes per element).
    fn elems(&self, count: usize, per: usize, width: usize) -> Result<usize, DecodeError> {
        let n = count.checked_mul(per).ok_or(OVERFLOW)?;
        match n.checked_mul(width) {
            Some(b) if b <= self.remaining => Ok(n),
            Some(_) => Err(TRUNC),
            None => Err(OVERFLOW),
        }
    }

    /// `count · per` bytes.
    pub fn bytes_vec(&mut self, count: usize, per: usize) -> Result<Vec<u8>, DecodeError> {
        let n = self.elems(count, per, 1)?;
        let mut v = vec![0u8; n];
        self.take_into(&mut v)?;
        Ok(v)
    }

    /// `count · per` little-endian `u32`s.
    pub fn u32_vec(&mut self, count: usize, per: usize) -> Result<Vec<u32>, DecodeError> {
        let n = self.elems(count, per, 4)?;
        let mut v = Vec::with_capacity(n);
        let mut buf = [0u8; BUF];
        let mut left = n;
        while left > 0 {
            let take = left.min(BUF / 4);
            self.take_into(&mut buf[..take * 4])?;
            for w in buf[..take * 4].chunks_exact(4) {
                v.push(u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
            }
            left -= take;
        }
        Ok(v)
    }

    pub fn u64_vec(&mut self, n: usize) -> Result<Vec<u64>, DecodeError> {
        let n = self.elems(n, 1, 8)?;
        (0..n).map(|_| self.u64()).collect()
    }

    pub fn f32_vec(&mut self, n: usize) -> Result<Vec<f32>, DecodeError> {
        let n = self.elems(n, 1, 4)?;
        (0..n).map(|_| self.u32().map(f32::from_bits)).collect()
    }

    /// Every declared byte was consumed.
    pub fn finish(&self) -> Result<(), DecodeError> {
        if self.remaining == 0 && self.off >= self.cur.len() {
            Ok(())
        } else {
            Err(DecodeError::Malformed("trailing bytes after payload"))
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Source over in-memory segments.
    pub(crate) struct Segs(pub Vec<Vec<u8>>);

    impl Source for Segs {
        fn next_body(&mut self) -> Result<Option<(Vec<u8>, usize)>, DecodeError> {
            Ok((!self.0.is_empty()).then(|| (self.0.remove(0), 0)))
        }
    }

    #[test]
    fn roundtrip_across_segments() {
        let mut v = Vec::new();
        v.put_u8(7);
        v.put_u16(0xBEEF);
        v.put_u64(u64::MAX - 3);
        let words: Vec<u32> = (0..3000).collect();
        v.put_u32s(&words);
        v.put_f32s(&[1.5, -2.25]);
        let mut c = Counter(0);
        c.put_u8(7);
        c.put_u16(0);
        c.put_u64(0);
        c.put_u32s(&words);
        c.put_f32s(&[0.0, 0.0]);
        assert_eq!(c.0, v.len() as u64);
        let (a, rest) = v.split_at(5);
        let (b, cc) = rest.split_at(4001);
        let mut src = Segs(vec![a.to_vec(), b.to_vec(), cc.to_vec()]);
        let mut r = Reader::new(&mut src, v.len());
        assert_eq!(r.u8().unwrap(), 7);
        assert_eq!(r.u16().unwrap(), 0xBEEF);
        assert_eq!(r.u64().unwrap(), u64::MAX - 3);
        assert_eq!(r.u32_vec(1000, 3).unwrap(), words);
        assert_eq!(r.f32_vec(2).unwrap(), vec![1.5, -2.25]);
        r.finish().unwrap();
        assert!(r.u8().is_err());
    }

    #[test]
    fn hostile_lengths_are_rejected_before_allocating() {
        let mut src = Segs(vec![vec![0; 3]]);
        let mut r = Reader::new(&mut src, 3);
        assert_eq!(r.bytes_vec(usize::MAX, 1).unwrap_err(), TRUNC);
        assert_eq!(r.bytes_vec(usize::MAX, 2).unwrap_err(), OVERFLOW);
        assert_eq!(r.u32_vec(usize::MAX / 2, 4).unwrap_err(), OVERFLOW);
        // 2^16 · 2^16 wraps a 32-bit usize to 0: must be an overflow error.
        assert!(r.bytes_vec(1 << 16, 1 << 16).is_err());
        // Declared longer than supplied: a typed error, not a panic.
        let mut src = Segs(vec![vec![1, 2]]);
        let mut r = Reader::new(&mut src, 10);
        assert_eq!(r.bytes_vec(4, 1).unwrap_err(), TRUNC);
    }
}
