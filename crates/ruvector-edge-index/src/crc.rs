//! CRC-32C (Castagnoli), slicing-by-8 with compile-time tables.
//!
//! The per-chunk integrity check. It is cheap (no SHA-NI needed on wasm32)
//! and localises bit rot to one part; the single whole-payload sha256 is
//! what pins an epoch (`meta.index_sha256`).

const POLY: u32 = 0x82F6_3B78;

const fn tables() -> [[u32; 256]; 8] {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ POLY } else { c >> 1 };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut i = 0;
    while i < 256 {
        let mut s = 1;
        while s < 8 {
            t[s][i] = (t[s - 1][i] >> 8) ^ t[0][(t[s - 1][i] & 0xFF) as usize];
            s += 1;
        }
        i += 1;
    }
    t
}

static T: [[u32; 256]; 8] = tables();

/// Incremental CRC-32C.
pub(crate) struct Crc32c(u32);

impl Crc32c {
    pub fn new() -> Self {
        Self(!0)
    }

    pub fn update(&mut self, b: &[u8]) {
        let mut c = self.0;
        let mut words = b.chunks_exact(8);
        for w in &mut words {
            let lo = c ^ u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            let hi = u32::from_le_bytes([w[4], w[5], w[6], w[7]]);
            c = T[7][(lo & 0xFF) as usize]
                ^ T[6][((lo >> 8) & 0xFF) as usize]
                ^ T[5][((lo >> 16) & 0xFF) as usize]
                ^ T[4][(lo >> 24) as usize]
                ^ T[3][(hi & 0xFF) as usize]
                ^ T[2][((hi >> 8) & 0xFF) as usize]
                ^ T[1][((hi >> 16) & 0xFF) as usize]
                ^ T[0][(hi >> 24) as usize];
        }
        for &x in words.remainder() {
            c = (c >> 8) ^ T[0][((c ^ u32::from(x)) & 0xFF) as usize];
        }
        self.0 = c;
    }

    pub fn finish(&self) -> u32 {
        !self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors_and_split_invariance() {
        let mut c = Crc32c::new();
        c.update(b"123456789");
        assert_eq!(c.finish(), 0xE306_9283);
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 31 + 7) as u8).collect();
        let mut whole = Crc32c::new();
        whole.update(&data);
        let mut parts = Crc32c::new();
        for p in data.chunks(13) {
            parts.update(p);
        }
        assert_eq!(whole.finish(), parts.finish());
        assert_eq!(Crc32c::new().finish(), 0);
    }
}
