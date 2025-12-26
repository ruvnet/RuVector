//! # PostgreSQL Attention Operators
//!
//! SQL-callable functions for attention mechanisms in PostgreSQL.

use super::{
    softmax, Attention, AttentionType, FlashAttention, MultiHeadAttention, ScaledDotAttention,
};
use pgrx::prelude::*;
use pgrx::JsonB;

/// Compute attention score between query and key vectors
///
/// # SQL Example
/// ```sql
/// SELECT ruvector_attention_score(
///     ARRAY[1.0, 0.0, 0.0]::float4[],
///     ARRAY[1.0, 0.0, 0.0]::float4[],
///     'scaled_dot'
/// );
/// ```
#[pg_extern(immutable, parallel_safe)]
pub fn ruvector_attention_score(
    query: Vec<f32>,
    key: Vec<f32>,
    attention_type: default!(&str, "'scaled_dot'"),
) -> f32 {
    // Parse attention type
    let attn_type = attention_type
        .parse::<AttentionType>()
        .unwrap_or(AttentionType::ScaledDot);

    // Validate dimensions
    if query.is_empty() || key.is_empty() {
        return 0.0;
    }

    if query.len() != key.len() {
        pgrx::error!(
            "Query and key dimensions must match: {} vs {}",
            query.len(),
            key.len()
        );
    }

    // Create attention mechanism
    let attention: Box<dyn Attention> = match attn_type {
        AttentionType::ScaledDot => Box::new(ScaledDotAttention::new(query.len())),
        AttentionType::FlashV2 => Box::new(FlashAttention::with_head_dim(query.len())),
        _ => Box::new(ScaledDotAttention::new(query.len())),
    };

    // Compute attention score
    let keys = vec![&key[..]];
    let scores = attention.attention_scores(&query, &keys);

    scores.first().copied().unwrap_or(0.0)
}

/// Apply softmax to an array of scores
///
/// # SQL Example
/// ```sql
/// SELECT ruvector_softmax(ARRAY[1.0, 2.0, 3.0]::float4[]);
/// -- Returns: {0.09, 0.24, 0.67}
/// ```
#[pg_extern(immutable, parallel_safe)]
pub fn ruvector_softmax(scores: Vec<f32>) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    softmax(&scores)
}

/// Compute multi-head attention between query and multiple keys
///
/// # SQL Example
/// ```sql
/// SELECT ruvector_multi_head_attention(
///     ARRAY[1.0, 0.0, 0.0, 0.0]::float4[],  -- query
///     '[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]'::jsonb,  -- keys
///     '[[1.0, 2.0], [3.0, 4.0]]'::jsonb,  -- values
///     2  -- num_heads
/// );
/// ```
#[pg_extern(immutable, parallel_safe)]
pub fn ruvector_multi_head_attention(
    query: Vec<f32>,
    keys_json: JsonB,
    values_json: JsonB,
    num_heads: default!(i32, 4),
) -> Vec<f32> {
    // Parse keys and values from JSON
    let keys: Vec<Vec<f32>> = match keys_json.0.as_array() {
        Some(arr) => arr
            .iter()
            .filter_map(|v| {
                v.as_array().map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect()
                })
            })
            .collect(),
        None => return Vec::new(),
    };

    let values: Vec<Vec<f32>> = match values_json.0.as_array() {
        Some(arr) => arr
            .iter()
            .filter_map(|v| {
                v.as_array().map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect()
                })
            })
            .collect(),
        None => return Vec::new(),
    };

    // Validate inputs
    if query.is_empty() || keys.is_empty() || values.is_empty() {
        return Vec::new();
    }

    if keys.len() != values.len() {
        pgrx::error!(
            "Keys and values must have same length: {} vs {}",
            keys.len(),
            values.len()
        );
    }

    let num_heads = num_heads.max(1) as usize;
    let total_dim = query.len();

    // Check dimension compatibility
    if total_dim % num_heads != 0 {
        pgrx::error!(
            "Query dimension {} must be divisible by num_heads {}",
            total_dim,
            num_heads
        );
    }

    // Validate all keys have same dimension
    for (i, key) in keys.iter().enumerate() {
        if key.len() != total_dim {
            pgrx::error!(
                "Key {} has dimension {} but expected {}",
                i,
                key.len(),
                total_dim
            );
        }
    }

    // Create multi-head attention
    let mha = MultiHeadAttention::new(num_heads, total_dim);

    // Convert to slice references
    let key_refs: Vec<&[f32]> = keys.iter().map(|k| &k[..]).collect();
    let value_refs: Vec<&[f32]> = values.iter().map(|v| &v[..]).collect();

    // Compute attention
    mha.forward(&query, &key_refs, &value_refs)
}

/// Compute Flash Attention v2 (memory-efficient)
///
/// # SQL Example
/// ```sql
/// SELECT ruvector_flash_attention(
///     ARRAY[1.0, 0.0, 0.0, 0.0]::float4[],
///     '[[1.0, 0.0, 0.0, 0.0]]'::jsonb,
///     '[[5.0, 10.0]]'::jsonb,
///     64  -- block_size
/// );
/// ```
#[pg_extern(immutable, parallel_safe)]
pub fn ruvector_flash_attention(
    query: Vec<f32>,
    keys_json: JsonB,
    values_json: JsonB,
    block_size: default!(i32, 64),
) -> Vec<f32> {
    // Parse keys and values from JSON
    let keys: Vec<Vec<f32>> = match keys_json.0.as_array() {
        Some(arr) => arr
            .iter()
            .filter_map(|v| {
                v.as_array().map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect()
                })
            })
            .collect(),
        None => return Vec::new(),
    };

    let values: Vec<Vec<f32>> = match values_json.0.as_array() {
        Some(arr) => arr
            .iter()
            .filter_map(|v| {
                v.as_array().map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect()
                })
            })
            .collect(),
        None => return Vec::new(),
    };

    // Validate inputs
    if query.is_empty() || keys.is_empty() || values.is_empty() {
        return Vec::new();
    }

    if keys.len() != values.len() {
        pgrx::error!("Keys and values must have same length");
    }

    let block_size = block_size.max(1) as usize;

    // Create Flash Attention
    let flash = FlashAttention::new(query.len(), block_size);

    // Convert to slice references
    let key_refs: Vec<&[f32]> = keys.iter().map(|k| &k[..]).collect();
    let value_refs: Vec<&[f32]> = values.iter().map(|v| &v[..]).collect();

    // Compute attention
    flash.forward(&query, &key_refs, &value_refs)
}

/// Get information about available attention types
///
/// # SQL Example
/// ```sql
/// SELECT * FROM ruvector_attention_types();
/// ```
#[pg_extern]
pub fn ruvector_attention_types() -> TableIterator<
    'static,
    (
        name!(name, String),
        name!(complexity, String),
        name!(best_for, String),
    ),
> {
    let types = vec![
        AttentionType::ScaledDot,
        AttentionType::MultiHead,
        AttentionType::FlashV2,
        AttentionType::Linear,
        AttentionType::Gat,
        AttentionType::Sparse,
        AttentionType::Moe,
        AttentionType::Cross,
        AttentionType::Sliding,
        AttentionType::Poincare,
    ];

    TableIterator::new(types.into_iter().map(|t| {
        (
            t.name().to_string(),
            t.complexity().to_string(),
            t.best_for().to_string(),
        )
    }))
}

/// Compute attention scores between a query and multiple keys
///
/// # SQL Example
/// ```sql
/// SELECT ruvector_attention_scores(
///     ARRAY[1.0, 0.0, 0.0]::float4[],
///     '[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]'::jsonb
/// );
/// -- Returns array of attention scores
/// ```
#[pg_extern(immutable, parallel_safe)]
pub fn ruvector_attention_scores(
    query: Vec<f32>,
    keys_json: JsonB,
    attention_type: default!(&str, "'scaled_dot'"),
) -> Vec<f32> {
    // Parse keys from JSON
    let keys: Vec<Vec<f32>> = match keys_json.0.as_array() {
        Some(arr) => arr
            .iter()
            .filter_map(|v| {
                v.as_array().map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64().map(|f| f as f32))
                        .collect()
                })
            })
            .collect(),
        None => return Vec::new(),
    };

    if query.is_empty() || keys.is_empty() {
        return Vec::new();
    }

    // Parse attention type
    let attn_type = attention_type
        .parse::<AttentionType>()
        .unwrap_or(AttentionType::ScaledDot);

    // Create attention mechanism
    let attention: Box<dyn Attention> = match attn_type {
        AttentionType::ScaledDot => Box::new(ScaledDotAttention::new(query.len())),
        AttentionType::FlashV2 => Box::new(FlashAttention::with_head_dim(query.len())),
        _ => Box::new(ScaledDotAttention::new(query.len())),
    };

    // Convert to slice references
    let key_refs: Vec<&[f32]> = keys.iter().map(|k| &k[..]).collect();

    // Compute attention scores
    attention.attention_scores(&query, &key_refs)
}

#[cfg(feature = "pg_test")]
#[pgrx::pg_schema]
mod tests {
    use super::*;

    #[pg_test]
    fn test_ruvector_attention_score() {
        let query = vec![1.0, 0.0, 0.0];
        let key = vec![1.0, 0.0, 0.0];

        let score = ruvector_attention_score(query, key, "scaled_dot");

        // Perfect match should give high score (after softmax, it would be 1.0)
        assert!(score > 0.99);
    }

    #[pg_test]
    fn test_ruvector_softmax() {
        let scores = vec![1.0, 2.0, 3.0];
        let result = ruvector_softmax(scores);

        assert_eq!(result.len(), 3);

        // Should sum to 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Higher input should have higher output
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[pg_test]
    fn test_ruvector_multi_head_attention() {
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let keys = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let values = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let result = ruvector_multi_head_attention(query, keys, values, 2);

        assert_eq!(result.len(), 2);
        // Should be closer to first value
        assert!(result[0] < 2.0);
    }

    #[pg_test]
    fn test_ruvector_flash_attention() {
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let keys = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let values = vec![vec![5.0, 10.0]];

        let result = ruvector_flash_attention(query, keys, values, 64);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 5.0).abs() < 0.01);
        assert!((result[1] - 10.0).abs() < 0.01);
    }

    #[pg_test]
    fn test_ruvector_attention_scores() {
        let query = vec![1.0, 0.0, 0.0];
        let keys = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let scores = ruvector_attention_scores(query, keys, "scaled_dot");

        assert_eq!(scores.len(), 3);

        // Should sum to 1 (softmax)
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // First key matches best
        assert!(scores[0] > scores[1]);
        assert!(scores[0] > scores[2]);
    }

    #[pg_test]
    fn test_ruvector_attention_types_query() {
        // This would be run as SQL: SELECT * FROM ruvector_attention_types();
        // Testing that the function doesn't panic
        let types = ruvector_attention_types();
        let results: Vec<_> = types.collect();

        // Should have multiple attention types
        assert!(results.len() >= 5);
    }
}
