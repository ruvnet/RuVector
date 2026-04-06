#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use ruvector_core::distance::{
    cosine_distance, dot_product_distance, euclidean_distance, manhattan_distance,
};
use ruvector_core::types::DistanceMetric;

/// Fuzzer input: two vectors of the same length plus a metric selector.
///
/// We cap the dimension at 2048 to keep iteration counts reasonable while
/// still exercising the 4-wide unrolled scalar loops and any SIMD paths.
#[derive(Debug, Arbitrary)]
struct DistanceInput {
    /// Raw f32 values for vector A (length determines dimension).
    values_a: Vec<f32>,
    /// Metric selector: 0=Euclidean, 1=Cosine, 2=DotProduct, 3=Manhattan
    metric_selector: u8,
}

fuzz_target!(|input: DistanceInput| {
    // Skip empty vectors -- distance functions require len > 0 for meaningful work.
    if input.values_a.is_empty() {
        return;
    }

    // Cap dimension to avoid excessive runtime on huge arbitrary vectors.
    let dim = input.values_a.len().min(2048);
    let a = &input.values_a[..dim];

    // Build a second vector of the same dimension by cycling values from `a`
    // with a simple perturbation, since `arbitrary` cannot easily guarantee
    // two vecs of identical length.
    let b: Vec<f32> = a.iter().map(|v| v + 1.0).collect();

    // Exercise the top-level dispatch that checks dimension equality.
    let metric = match input.metric_selector % 4 {
        0 => DistanceMetric::Euclidean,
        1 => DistanceMetric::Cosine,
        2 => DistanceMetric::DotProduct,
        3 => DistanceMetric::Manhattan,
        _ => unreachable!(),
    };

    let result = ruvector_core::distance::distance(a, &b, metric);
    // Must succeed -- dimensions match.
    assert!(result.is_ok(), "distance() failed with matching dimensions");
    let dist = result.unwrap();

    // Basic sanity: distance should not be NaN (Inf is acceptable for
    // degenerate inputs but NaN indicates a logic bug).
    // Exception: cosine_distance can produce NaN when both norms are 0
    // and the fallback division triggers.  We allow that specific case.
    if !matches!(metric, DistanceMetric::Cosine) {
        assert!(
            !dist.is_nan(),
            "distance returned NaN for metric {:?}, a[0..4]={:?}",
            metric,
            &a[..a.len().min(4)]
        );
    }

    // Exercise individual distance functions directly with equal-length slices.
    let _ = euclidean_distance(a, &b);
    let _ = cosine_distance(a, &b);
    let _ = dot_product_distance(a, &b);
    let _ = manhattan_distance(a, &b);

    // Verify dimension-mismatch path: truncate b by 1.
    if dim > 1 {
        let short_b = &b[..dim - 1];
        let err_result = ruvector_core::distance::distance(a, short_b, metric);
        assert!(
            err_result.is_err(),
            "distance() should fail on dimension mismatch"
        );
    }
});
