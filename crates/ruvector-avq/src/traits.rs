//! Generic quantizer interface so AVQ, uniform PQ and scalar
//! quantization are all swappable behind one type.

/// Encodes vectors to a compressed code.
pub trait Encoder: Send + Sync {
    /// Compressed bytes per encoded vector.
    fn code_size(&self) -> usize;

    /// Original dimension this encoder accepts.
    fn dim(&self) -> usize;

    /// Encode `n` rows of dimension `dim()` packed in `xs`. Output
    /// `codes` must be `n * code_size()` bytes.
    fn encode(&self, xs: &[f32], codes: &mut [u8]);
}

/// Computes inner-product scores against codes given a query.
pub trait Scorer: Encoder {
    /// Score `n_codes` against `query`. `out[i]` receives the
    /// (approximate) inner product `<query, decode(codes[i])>`.
    fn score_ip(&self, query: &[f32], codes: &[u8], out: &mut [f32]);

    /// Top-`k` by descending score. Returns indices into the code
    /// table. Linear scan — meant for benchmarking quantizer quality
    /// rather than as a production index.
    fn topk_ip(&self, query: &[f32], codes: &[u8], k: usize) -> Vec<(u32, f32)> {
        let n = codes.len() / self.code_size();
        let mut scores = vec![0.0f32; n];
        self.score_ip(query, codes, &mut scores);
        let mut idx: Vec<u32> = (0..n as u32).collect();
        idx.sort_by(|&a, &b| {
            scores[b as usize]
                .partial_cmp(&scores[a as usize])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        idx.truncate(k);
        idx.into_iter().map(|i| (i, scores[i as usize])).collect()
    }
}
