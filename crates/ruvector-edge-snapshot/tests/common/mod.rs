//! Shared fixtures for the integration tests.
#![allow(dead_code)]

use ruvector_edge_snapshot::*;

pub const TENANT_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaa";
pub const TENANT_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbb";
pub const UID: &str = "0123456789abcdef0123456789abcdef";
pub const UID2: &str = "fedcba9876543210fedcba9876543210";
pub const CLOCK: FixedClock = FixedClock(1_790_000_000_000);

/// Deterministic xorshift.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Row `i` of a deterministic data set (ids sort in `i` order).
pub fn row(i: u64, dim: u16, rng: &mut Rng) -> Row {
    Row {
        id: format!("doc-{i:09}"),
        values: (0..dim).map(|_| rng.f32()).collect(),
        metadata: (!i.is_multiple_of(3)).then(|| format!("{{\"n\":{i},\"tag\":\"t{}\"}}", i % 7)),
    }
}

pub fn rows(n: u64, dim: u16, seed: u64) -> Vec<Row> {
    let mut rng = Rng(seed | 1);
    (0..n).map(|i| row(i, dim, &mut rng)).collect()
}

pub fn shard(tenant: &str) -> ShardRef {
    ShardRef::new(tenant, "vector", UID, 0).unwrap()
}

/// Small limits so a few hundred rows span several segments and chunks.
pub fn small_limits(dim: u16) -> SnapshotLimits {
    let seg = 12 + 2 + 256 + 1 + 2 + 4096 + usize::from(dim) * 4;
    SnapshotLimits {
        max_chunk_bytes: rvf_wire_padded(seg) * 2,
        max_segment_payload: seg,
        max_rows: 1_000_000,
        max_rows_per_chunk: MAX_ROWS_PER_CHUNK,
    }
}

pub fn rvf_wire_padded(payload: usize) -> usize {
    (64 + payload + 63) & !63
}

pub struct Snap {
    pub chunks: Vec<Chunk>,
    pub sealed: SealedManifest,
    pub object: Vec<u8>,
}

pub fn write_snapshot(
    tenant: &str,
    rows: &[Row],
    dim: u16,
    epoch: u64,
    _chain: &WitnessChain,
    limits: SnapshotLimits,
    signer: Option<&dyn Signer>,
) -> Snap {
    let spec = SnapshotSpec {
        shard: shard(tenant),
        epoch,
        dim,
        metric: Metric::Cosine,
        audit_head: [9; 32],
    };
    let mut w = SnapshotWriter::new(spec, limits).unwrap();
    let mut chunks = Vec::new();
    for r in rows {
        if let Some(c) = w.push(r).unwrap() {
            chunks.push(c);
        }
    }
    let (tail, sealed) = w.finish(&CLOCK, signer).unwrap();
    chunks.extend(tail);
    let object = sealed.to_object();
    Snap {
        chunks,
        sealed,
        object,
    }
}

pub struct TestSigner(pub ed25519_dalek::SigningKey);
impl TestSigner {
    pub fn fixed(seed: u8) -> Self {
        TestSigner(ed25519_dalek::SigningKey::from_bytes(&[seed; 32]))
    }
    pub fn public(&self) -> [u8; 32] {
        self.0.verifying_key().to_bytes()
    }
}
impl Signer for TestSigner {
    fn key_id(&self) -> &str {
        "snap-k1"
    }
    fn sign(&self, msg: &[u8]) -> Result<[u8; 64], SnapshotError> {
        use ed25519_dalek::Signer as _;
        Ok(self.0.sign(msg).to_bytes())
    }
}

pub fn target(tenant: &str) -> RestoreTarget {
    RestoreTarget {
        shard: shard(tenant),
        dim: None,
        metric: None,
    }
}

pub const QUOTA: RestoreQuota = RestoreQuota {
    max_rows: 10_000_000,
    max_bytes: 1 << 34,
};

/// Full restore; returns rows or the first error.
pub fn restore(
    object: &[u8],
    chunks: &[Vec<u8>],
    tenant: &str,
    entries: &[WitnessEntry],
    head: [u8; 32],
    sig: SignaturePolicy<'_>,
) -> Result<Vec<Row>, SnapshotError> {
    let proof = ChainProof::from_genesis(tenant, entries, head);
    let mut s = RestoreSession::begin(object, &target(tenant), QUOTA, proof, sig)?;
    let mut out = Vec::new();
    for (i, c) in chunks.iter().enumerate() {
        s.accept_chunk(i as u32, c, &mut out)?;
    }
    s.finish()?;
    Ok(out)
}

pub fn assert_rows_identical(a: &[Row], b: &[Row]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!(x.bitwise_eq(y), "row {} differs", x.id);
    }
}

/// Exact top-k by cosine distance, ties by id (the drill's query oracle).
pub fn topk(rows: &[Row], q: &[f32], k: usize) -> Vec<(String, u32)> {
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let qn = norm(q);
    let mut scored: Vec<(f64, &str)> = rows
        .iter()
        .map(|r| {
            let dot: f64 = r
                .values
                .iter()
                .zip(q)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            (1.0 - dot / (norm(&r.values) * qn), r.id.as_str())
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(b.1)));
    scored
        .into_iter()
        .take(k)
        .map(|(d, id)| (id.to_string(), (d as f32).to_bits()))
        .collect()
}
