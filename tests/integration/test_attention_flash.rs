//! Integration tests for ruvector-attention: Flash Attention, ScaledDotProduct,
//! MultiHeadAttention, and the Attention trait.
//!
//! All tests use real types and real computations -- no mocks.

use ruvector_attention::attention::multi_head::MultiHeadAttention;
use ruvector_attention::attention::ScaledDotProductAttention;
use ruvector_attention::sparse::flash::FlashAttention;
use ruvector_attention::traits::Attention;

// ---------------------------------------------------------------------------
// Helper: build key/value data
// ---------------------------------------------------------------------------
fn make_kv(n: usize, dim: usize, scale: f32) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let keys: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.01 + scale) % 2.0 - 1.0)
                .collect()
        })
        .collect();
    let values: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.017 + scale * 0.5) % 2.0 - 1.0)
                .collect()
        })
        .collect();
    (keys, values)
}

// ===========================================================================
// 1. ScaledDotProductAttention: basic computation
// ===========================================================================
#[test]
fn scaled_dot_product_basic() {
    let dim = 32;
    let attn = ScaledDotProductAttention::new(dim);

    let query = vec![0.5f32; dim];
    let (keys, values) = make_kv(8, dim, 0.3);

    let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    let result = attn
        .compute(&query, &keys_refs, &values_refs)
        .expect("scaled dot product should succeed");

    assert_eq!(
        result.len(),
        dim,
        "output should have same dimension as input"
    );

    // Output should be a convex combination of values -- each component should be
    // within the range of value components.
    for &val in &result {
        assert!(
            val.is_finite(),
            "all output values should be finite"
        );
    }
}

// ===========================================================================
// 2. ScaledDotProductAttention: uniform keys produce uniform weights
// ===========================================================================
#[test]
fn scaled_dot_product_uniform_keys() {
    let dim = 16;
    let attn = ScaledDotProductAttention::new(dim);

    let query = vec![1.0f32; dim];
    // All keys identical -> all weights equal -> output = average of values
    let keys = vec![vec![1.0f32; dim]; 4];
    let values = vec![
        vec![1.0f32; dim],
        vec![2.0f32; dim],
        vec![3.0f32; dim],
        vec![4.0f32; dim],
    ];

    let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    let result = attn
        .compute(&query, &keys_refs, &values_refs)
        .expect("compute");

    // Average of 1,2,3,4 = 2.5
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 2.5).abs() < 1e-3,
            "with uniform keys, output[{i}] should be ~2.5, got {v}"
        );
    }
}

// ===========================================================================
// 3. FlashAttention: matches standard attention output
// ===========================================================================
#[test]
fn flash_attention_matches_standard() {
    let dim = 32;
    let flash = FlashAttention::new(dim, 8);
    let standard = ScaledDotProductAttention::new(dim);

    let query = vec![0.5f32; dim];
    let (keys, values) = make_kv(16, dim, 0.7);

    let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    let flash_result = flash
        .compute(&query, &keys_refs, &values_refs)
        .expect("flash compute");
    let standard_result = standard
        .compute(&query, &keys_refs, &values_refs)
        .expect("standard compute");

    assert_eq!(flash_result.len(), standard_result.len());
    for (i, (f, s)) in flash_result.iter().zip(standard_result.iter()).enumerate() {
        assert!(
            (f - s).abs() < 1e-4,
            "flash[{i}]={f} vs standard[{i}]={s}, diff={}",
            (f - s).abs()
        );
    }
}

// ===========================================================================
// 4. FlashAttention: large sequence length (memory efficiency test)
// ===========================================================================
#[test]
fn flash_attention_large_sequence() {
    let dim = 64;
    let n = 1024;
    let block_size = 64;
    let flash = FlashAttention::new(dim, block_size);

    let query = vec![0.3f32; dim];
    let (keys, values) = make_kv(n, dim, 0.1);

    let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    let result = flash
        .compute(&query, &keys_refs, &values_refs)
        .expect("flash on large sequence");

    assert_eq!(result.len(), dim);
    for &v in &result {
        assert!(v.is_finite(), "output values should be finite for large n");
    }
}

// ===========================================================================
// 5. FlashAttention: empty keys returns error
// ===========================================================================
#[test]
fn flash_attention_empty_keys_error() {
    let dim = 32;
    let flash = FlashAttention::new(dim, 8);
    let query = vec![1.0f32; dim];

    let result = flash.compute(&query, &[], &[]);
    assert!(result.is_err(), "empty keys should produce an error");
}

// ===========================================================================
// 6. FlashAttention: dimension mismatch returns error
// ===========================================================================
#[test]
fn flash_attention_dimension_mismatch() {
    let flash = FlashAttention::new(32, 8);
    let wrong_query = vec![1.0f32; 16]; // dim 16, but flash expects 32

    let keys = vec![vec![0.5f32; 32]];
    let values = vec![vec![1.0f32; 32]];
    let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    let result = flash.compute(&wrong_query, &keys_refs, &values_refs);
    assert!(
        result.is_err(),
        "dimension mismatch should produce an error"
    );
}

// ===========================================================================
// 7. FlashAttention: causal mode
// ===========================================================================
#[test]
fn flash_attention_causal_mode() {
    let dim = 32;
    let flash = FlashAttention::causal(dim, 8);

    let query = vec![1.0f32; dim];
    let (keys, values) = make_kv(20, dim, 0.5);

    let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    let result = flash
        .compute(&query, &keys_refs, &values_refs)
        .expect("causal flash attention");

    assert_eq!(result.len(), dim);
    for &v in &result {
        assert!(v.is_finite(), "causal attention output should be finite");
    }
}

// ===========================================================================
// 8. MultiHeadAttention: basic computation
// ===========================================================================
#[test]
fn multi_head_attention_basic() {
    let dim = 8;
    let num_heads = 2;
    let attn = MultiHeadAttention::new(dim, num_heads);

    assert_eq!(attn.dim(), dim);
    assert_eq!(attn.num_heads(), num_heads);

    let query = vec![1.0f32; dim];
    let keys = vec![vec![0.5f32; dim]; 4];
    let values = vec![vec![1.0f32; dim]; 4];

    let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    let result = attn
        .compute(&query, &keys_refs, &values_refs)
        .expect("multi-head compute");

    assert_eq!(result.len(), dim, "output dimension should match input");
}

// ===========================================================================
// 9. MultiHeadAttention: dimension not divisible by heads panics
// ===========================================================================
#[test]
#[should_panic(expected = "divisible")]
fn multi_head_attention_invalid_heads_panics() {
    let _attn = MultiHeadAttention::new(10, 3);
}

// ===========================================================================
// 10. FlashAttention: with mask filtering
// ===========================================================================
#[test]
fn flash_attention_with_mask() {
    let dim = 16;
    let flash = FlashAttention::new(dim, 4);

    let query = vec![1.0f32; dim];
    let keys = vec![vec![0.5f32; dim]; 8];
    let values = vec![
        vec![1.0f32; dim],
        vec![2.0f32; dim],
        vec![3.0f32; dim],
        vec![4.0f32; dim],
        vec![5.0f32; dim],
        vec![6.0f32; dim],
        vec![7.0f32; dim],
        vec![8.0f32; dim],
    ];

    let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    // Mask: only attend to first 4 (true) and skip last 4 (false)
    let mask = vec![true, true, true, true, false, false, false, false];

    let masked_result = flash
        .compute_with_mask(&query, &keys_refs, &values_refs, Some(&mask))
        .expect("masked flash attention");

    // Unmasked with only first 4 values
    let first_4_keys: Vec<&[f32]> = keys_refs[..4].to_vec();
    let first_4_values: Vec<&[f32]> = values_refs[..4].to_vec();
    let unmasked_result = flash
        .compute(&query, &first_4_keys, &first_4_values)
        .expect("unmasked with first 4");

    // Masked result should equal result with only the unmasked entries
    for (i, (m, u)) in masked_result.iter().zip(unmasked_result.iter()).enumerate() {
        assert!(
            (m - u).abs() < 1e-4,
            "masked[{i}]={m} vs unmasked[{i}]={u}"
        );
    }
}

// ===========================================================================
// 11. FlashAttention: block_size=1 (degenerate but valid)
// ===========================================================================
#[test]
fn flash_attention_block_size_one() {
    let dim = 8;
    let flash = FlashAttention::new(dim, 1);

    let query = vec![0.5f32; dim];
    let (keys, values) = make_kv(10, dim, 0.2);

    let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    let result = flash
        .compute(&query, &keys_refs, &values_refs)
        .expect("block_size=1 should work");

    assert_eq!(result.len(), dim);
}

// ===========================================================================
// 12. ScaledDotProductAttention: single key-value pair
// ===========================================================================
#[test]
fn scaled_dot_product_single_kv() {
    let dim = 4;
    let attn = ScaledDotProductAttention::new(dim);

    let query = vec![1.0f32; dim];
    let keys = vec![vec![0.5f32; dim]];
    let values = vec![vec![42.0f32; dim]];

    let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    let result = attn
        .compute(&query, &keys_refs, &values_refs)
        .expect("single kv");

    // With a single key-value, softmax weight is 1.0, so output = value
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 42.0).abs() < 1e-4,
            "single kv output[{i}] should be ~42.0, got {v}"
        );
    }
}
