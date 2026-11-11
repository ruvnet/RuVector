//! Exact squared-L2 assignment with bounded work on losing candidates.

/// Inputs have already passed the public API's shape and finiteness checks.
/// All squared terms are nonnegative: once a partial distance reaches the
/// best distance, that later centroid cannot win, including in a tie.
pub(crate) fn nearest(row: &[f32], centroids: &[f32], dimensions: usize) -> (usize, f64) {
    // End-to-end diffuse PQ benchmarks favor the original scalar path at small
    // dimensions: pruning checks cost more than the arithmetic they can skip.
    if dimensions <= 16 {
        exhaustive(row, centroids, dimensions)
    } else {
        pruned::<8>(row, centroids, dimensions)
    }
}

fn exhaustive(row: &[f32], centroids: &[f32], dimensions: usize) -> (usize, f64) {
    let mut best = (0, f64::INFINITY);
    for (j, centroid) in centroids.chunks_exact(dimensions).enumerate() {
        let distance: f64 = row
            .iter()
            .zip(centroid)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum();
        if distance < best.1 {
            best = (j, distance);
            if distance == 0.0 {
                break;
            }
        }
    }
    best
}

fn pruned<const BLOCK: usize>(row: &[f32], centroids: &[f32], dimensions: usize) -> (usize, f64) {
    let mut best = (0, f64::INFINITY);
    for (j, centroid) in centroids.chunks_exact(dimensions).enumerate() {
        let mut distance = 0.0;
        for (xs, ys) in row.chunks(BLOCK).zip(centroid.chunks(BLOCK)) {
            // Retain left-to-right f64 addition. Reassociating these terms, or
            // expanding norms and dot products, can change near-tie winners.
            for (&x, &y) in xs.iter().zip(ys) {
                let difference = f64::from(x) - f64::from(y);
                distance += difference * difference;
            }
            if distance >= best.1 {
                break;
            }
        }
        if distance < best.1 {
            best = (j, distance);
            if distance == 0.0 {
                break;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::nearest;

    /// The pre-optimization exhaustive implementation is an independent oracle.
    fn oracle(row: &[f32], centroids: &[f32], dimensions: usize) -> (usize, f64) {
        let mut best = (0, f64::INFINITY);
        for (j, centroid) in centroids.chunks_exact(dimensions).enumerate() {
            let distance: f64 = row
                .iter()
                .zip(centroid)
                .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                .sum();
            if distance < best.1 {
                best = (j, distance);
            }
        }
        best
    }

    fn compare(row: &[f32], centroids: &[f32]) {
        let expected = oracle(row, centroids, row.len());
        let actual = nearest(row, centroids, row.len());
        assert_eq!(actual.0, expected.0, "{} dimensions", row.len());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }

    fn random(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 40) as f32 / 16777216.0) * 2.0 - 1.0
    }

    #[test]
    fn assignment_matches_exhaustive_on_diffuse_and_separated_data() {
        let mut state = 42;
        for dimensions in [
            1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 384, 385, 768, 769,
        ] {
            for clusters in [1, 2, 7, 32] {
                for separated in [false, true] {
                    let centroids = (0..clusters * dimensions)
                        .map(|i| {
                            random(&mut state)
                                + if separated {
                                    (i / dimensions) as f32 * 4.0
                                } else {
                                    0.0
                                }
                        })
                        .collect::<Vec<_>>();
                    for i in 0..13 {
                        let row = (0..dimensions)
                            .map(|d| {
                                if separated {
                                    centroids[(i % clusters) * dimensions + d]
                                        + random(&mut state) * 0.01
                                } else {
                                    random(&mut state)
                                }
                            })
                            .collect::<Vec<_>>();
                        compare(&row, &centroids);
                    }
                }
            }
        }
    }

    #[test]
    fn ties_and_zero_distances_keep_the_lowest_index() {
        for dimensions in [1, 3, 8, 16, 32, 128, 384, 768] {
            let row = vec![0.0; dimensions];
            let mut centroids = vec![1.0; dimensions];
            centroids.extend(vec![-1.0; dimensions]);
            compare(&row, &centroids);
            assert_eq!(nearest(&row, &centroids, dimensions).0, 0);
            centroids.extend(vec![-0.0; dimensions]);
            centroids.extend(vec![0.0; dimensions]);
            compare(&row, &centroids);
            assert_eq!(nearest(&row, &centroids, dimensions).0, 2);
        }
    }

    #[test]
    fn extreme_finite_coordinates_and_subnormals_preserve_exact_distance() {
        let values = [
            f32::MAX,
            -f32::MAX,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
            0.0,
            -0.0,
            1.0,
        ];
        for dimensions in [1, 3, 8, 16, 32, 128, 384, 768] {
            for offset in 0..values.len() {
                let row = (0..dimensions)
                    .map(|d| values[(d + offset) % values.len()])
                    .collect::<Vec<_>>();
                let centroids = (0..7 * dimensions)
                    .map(|d| values[(d * 3 + offset + 1) % values.len()])
                    .collect::<Vec<_>>();
                compare(&row, &centroids);
            }
        }
    }

    #[test]
    fn near_ties_and_awkward_tail_dimensions_preserve_ordered_addition() {
        for dimensions in [3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 129, 385, 769] {
            let row = vec![0.0; dimensions];
            let mut first = vec![1.0; dimensions];
            let mut second = first.clone();
            first[dimensions - 1] = f32::from_bits(1.0_f32.to_bits() + 1);
            let mut centroids = first.clone();
            centroids.extend_from_slice(&second);
            compare(&row, &centroids);
            assert_eq!(nearest(&row, &centroids, dimensions).0, 1);

            // Tiny trailing contributions disappear when added after a huge
            // leading square; summation reassociation must not break this tie.
            first[0] = 1e20;
            second[0] = 1e20;
            let mut centroids = first;
            centroids.extend(second);
            compare(&row, &centroids);
            assert_eq!(nearest(&row, &centroids, dimensions).0, 0);
        }
    }
}
