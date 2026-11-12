//! Token -> sentence pooling and L2 normalisation, on flat `&[f32]` slices.
//!
//! No `ndarray`, no `rayon` — the batches here are small (a decision engine's
//! options), and keeping this dependency-free lets it compile identically for
//! native and wasm and stay inside the parity gate.

use crate::manifest::Pooling;

/// Pool one sequence of token embeddings into a single vector.
///
/// * `tokens` — flat `[seq_len * hidden]` row-major token embeddings.
/// * `mask` — attention mask `[seq_len]` (1 = real token, 0 = padding).
/// * `hidden` — embedding dimension.
///
/// Mean pooling averages the masked tokens; CLS pooling takes token 0. The
/// result is NOT normalised here — call [`l2_normalize`] after.
pub fn pool(strategy: Pooling, tokens: &[f32], mask: &[i64], hidden: usize) -> Vec<f32> {
    match strategy {
        Pooling::Cls => cls_pool(tokens, hidden),
        Pooling::Mean => mean_pool(tokens, mask, hidden),
    }
}

fn cls_pool(tokens: &[f32], hidden: usize) -> Vec<f32> {
    // First token's hidden vector. Guard against a short buffer.
    let end = hidden.min(tokens.len());
    let mut v = tokens[..end].to_vec();
    v.resize(hidden, 0.0);
    v
}

fn mean_pool(tokens: &[f32], mask: &[i64], hidden: usize) -> Vec<f32> {
    let seq_len = tokens.len().checked_div(hidden).unwrap_or(0);
    let mut acc = vec![0.0f32; hidden];
    let mut count = 0.0f32;
    for i in 0..seq_len {
        let m = mask.get(i).copied().unwrap_or(0);
        if m != 0 {
            let start = i * hidden;
            let row = &tokens[start..start + hidden];
            for (a, x) in acc.iter_mut().zip(row) {
                *a += *x;
            }
            count += 1.0;
        }
    }
    if count > 0.0 {
        for a in &mut acc {
            *a /= count;
        }
    }
    acc
}

/// L2-normalise in place so cosine similarity becomes a dot product. A zero
/// vector is left untouched.
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity of two vectors (unnormalised inputs handled).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 1e-12 && nb > 1e-12 {
        dot / (na * nb)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_ignores_padding() {
        // 3 tokens x 2 hidden, third token is padding (mask 0).
        let tokens = vec![1.0, 1.0, 3.0, 3.0, 99.0, 99.0];
        let mask = vec![1i64, 1, 0];
        let v = mean_pool(&tokens, &mask, 2);
        // mean of [1,1] and [3,3] = [2,2]; padding excluded.
        assert_eq!(v, vec![2.0, 2.0]);
    }

    #[test]
    fn cls_pool_takes_first_token() {
        let tokens = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0];
        let v = cls_pool(&tokens, 3);
        assert_eq!(v, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn l2_normalize_unit_length() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_is_safe() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn cosine_identical_is_one() {
        let a = vec![0.1, 0.2, 0.3];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pool_dispatch_matches_helpers() {
        let tokens = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![1i64, 1];
        assert_eq!(pool(Pooling::Cls, &tokens, &mask, 2), vec![1.0, 2.0]);
        assert_eq!(pool(Pooling::Mean, &tokens, &mask, 2), vec![2.0, 3.0]);
    }
}
