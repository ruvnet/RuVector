use ruvector_proof_gate::{HashChainGate, WriteGate, WritePayload, WriteReceipt};

/// A single write-provenance-bound retrieval result.
#[derive(Debug, Clone)]
pub struct ResultItem {
    pub vector_id: u64,
    pub rank: u32,
    pub score: f32,
    /// The write receipt produced when this vector was ingested. The
    /// retrieval receipt binds *copies* of this receipt's fields, making
    /// the cited ingestion record part of what the retrieval receipt
    /// commits to. This is a tamper-evident record of "which write receipt
    /// was cited", not a proof of write-chain membership — see the threat
    /// model in `receipt::result_leaf` and the crate docs.
    pub write_receipt: WriteReceipt,
}

/// Deterministic xorshift64 stream, identical construction to the one used
/// by `ruvector_proof_gate::synthetic_payloads`, so datasets are reproducible
/// without an external RNG dependency.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as i64 as f64 / i64::MAX as f64) as f32
    }
}

/// A brute-force, exact cosine-similarity retrieval index whose ingestion
/// path is wrapped by a `ruvector_proof_gate::HashChainGate`. Every stored
/// vector therefore carries a tamper-evident `WriteReceipt` from the moment
/// it is admitted, and `search` returns those receipts alongside each
/// result so a retrieval receipt can commit to them.
///
/// Brute force (not HNSW/ANN) is deliberate: this experiment isolates the
/// cost and integrity properties of the *provenance layer*, not retrieval
/// recall, which would otherwise conflate two independent variables. See
/// the nightly research README for the explicit scope statement.
pub struct RetrievalIndex {
    gate: HashChainGate,
    ids: Vec<u64>,
    vectors: Vec<Vec<f32>>,
    write_receipts: Vec<WriteReceipt>,
}

impl RetrievalIndex {
    /// Ingest `n` deterministic `dims`-dimensional vectors, admitting each
    /// through the write gate so a real chained `WriteReceipt` exists.
    pub fn ingest(n: usize, dims: usize, seed: u64) -> Self {
        let mut gate = HashChainGate::new();
        let mut rng = Xorshift64::new(seed);
        let mut ids = Vec::with_capacity(n);
        let mut vectors = Vec::with_capacity(n);
        let mut write_receipts = Vec::with_capacity(n);
        for i in 0..n {
            let vector: Vec<f32> = (0..dims).map(|_| rng.next_f32()).collect();
            let payload = WritePayload::new(i as u64, vector.clone());
            let receipt = gate.admit(&payload).expect("null-error gate never rejects");
            ids.push(i as u64);
            vectors.push(vector);
            write_receipts.push(receipt);
        }
        Self {
            gate,
            ids,
            vectors,
            write_receipts,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Current write-chain commitment: the tamper-evident state root of
    /// everything ingested so far. Bound into every retrieval receipt so a
    /// receipt also attests to *which version of the index* answered the
    /// query.
    pub fn index_state_root(&self) -> [u8; 32] {
        self.gate.chain_root()
    }

    /// Independently confirms the write gate's internal chain has not been
    /// tampered with (delegates to `HashChainGate::verify_integrity`).
    pub fn verify_write_history(&self) -> bool {
        self.gate.verify_integrity()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    /// Exact brute-force top-k cosine search. Ground truth by construction
    /// (no approximation), so recall is always 1.0 and out of scope as a
    /// measured variable in this experiment.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<ResultItem> {
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, Self::cosine(query, v)))
            .collect();
        // total_cmp: NaN-safe total order (a NaN-scored candidate sorts
        // last instead of panicking a public API on adversarial input).
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(rank, (idx, score))| ResultItem {
                vector_id: self.ids[idx],
                rank: rank as u32,
                score,
                write_receipt: self.write_receipts[idx].clone(),
            })
            .collect()
    }
}

/// Deterministic query generator: derives query vectors from the same
/// xorshift stream family as `RetrievalIndex::ingest`, seeded independently
/// so queries are reproducible but not identical to any stored vector.
pub fn synthetic_queries(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Xorshift64::new(seed);
    (0..n)
        .map(|_| (0..dims).map(|_| rng.next_f32()).collect())
        .collect()
}
