//! Tests for manifold projection (ADR-012).
//!
//! The properties that matter are determinism, exact out-of-sample extension,
//! and preservation of structure that genuinely lies in a low-dimensional
//! subspace. Each test constructs data whose correct projection is known.

use sevensense_vector::projection::{
    normalize_coordinates, to_poincare_ball, PcaProjection, ProjectionError,
};

/// Deterministic pseudo-random values in [-1, 1].
fn pseudo_random(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// Points on a line through 3-space, with negligible off-axis variation.
fn collinear(n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            vec![t, 2.0 * t, -t]
        })
        .collect()
}

/// Points spanning a 2D plane embedded in `dim`-dimensional space.
fn plane_in_high_dim(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let noise = pseudo_random(7, n * dim);
    (0..n)
        .map(|i| {
            let u = (i % 20) as f32 - 10.0;
            let v = (i / 20) as f32 - 5.0;
            let mut point = vec![0.0f32; dim];
            // Two orthogonal directions carry the structure.
            point[0] = u;
            point[1] = v;
            point[2] = 0.5 * u + 0.25 * v;
            for (d, value) in point.iter_mut().enumerate().skip(3) {
                *value = noise[i * dim + d] * 0.001;
            }
            point
        })
        .collect()
}

#[test]
fn a_single_dominant_axis_explains_nearly_all_variance() {
    let pca = PcaProjection::fit(&collinear(50), 2, 42).unwrap();

    assert!(
        pca.explained_variance()[0] > 0.99,
        "first component explained only {}",
        pca.explained_variance()[0]
    );
}

#[test]
fn fitting_is_deterministic_for_a_given_seed() {
    let data = plane_in_high_dim(100, 32);

    let first = PcaProjection::fit(&data, 3, 1234).unwrap();
    let second = PcaProjection::fit(&data, 3, 1234).unwrap();

    assert_eq!(
        first, second,
        "the same data and seed must produce identical components"
    );
}

#[test]
fn projection_is_stable_across_repeated_calls() {
    let data = plane_in_high_dim(60, 16);
    let pca = PcaProjection::fit(&data, 2, 99).unwrap();

    let once = pca.project(&data[7]).unwrap();
    let twice = pca.project(&data[7]).unwrap();

    assert_eq!(once, twice);
}

#[test]
fn out_of_sample_projection_matches_the_batch_path() {
    // The streaming path projects one point at a time; it must agree exactly
    // with projecting the whole corpus.
    let data = plane_in_high_dim(80, 24);
    let pca = PcaProjection::fit(&data, 3, 5).unwrap();

    let batch = pca.project_batch(&data).unwrap();
    for (i, point) in data.iter().enumerate() {
        assert_eq!(pca.project(point).unwrap(), batch[i], "point {i} disagreed");
    }
}

#[test]
fn a_point_absent_from_training_still_projects() {
    let data = plane_in_high_dim(60, 16);
    let pca = PcaProjection::fit(&data, 2, 11).unwrap();

    let unseen: Vec<f32> = (0..16).map(|d| if d < 3 { 3.5 } else { 0.0 }).collect();
    let projected = pca.project(&unseen).unwrap();

    assert_eq!(projected.len(), 2);
    assert!(projected.iter().all(|v| v.is_finite()));
}

#[test]
fn structure_in_a_low_dimensional_subspace_survives_projection() {
    // Data genuinely lying in a plane should keep its pairwise distances when
    // projected to that plane.
    let data = plane_in_high_dim(100, 40);
    let pca = PcaProjection::fit(&data, 2, 3).unwrap();
    let projected = pca.project_batch(&data).unwrap();

    let original_distance = |a: usize, b: usize| -> f32 {
        data[a]
            .iter()
            .zip(data[b].iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    };
    let projected_distance = |a: usize, b: usize| -> f32 {
        projected[a]
            .iter()
            .zip(projected[b].iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    };

    let mut worst_error = 0.0f32;
    for a in 0..30 {
        for b in (a + 1)..30 {
            let original = original_distance(a, b);
            if original < 1e-3 {
                continue;
            }
            let error = (projected_distance(a, b) - original).abs() / original;
            worst_error = worst_error.max(error);
        }
    }

    assert!(
        worst_error < 0.05,
        "worst pairwise distance error was {worst_error}"
    );
}

#[test]
fn components_are_orthonormal() {
    let data = plane_in_high_dim(80, 32);
    let pca = PcaProjection::fit(&data, 3, 21).unwrap();
    let projected = pca.project_batch(&data).unwrap();

    // Orthogonality shows up as uncorrelated projected axes.
    let n = projected.len() as f32;
    for a in 0..3 {
        for b in (a + 1)..3 {
            let mean_a: f32 = projected.iter().map(|p| p[a]).sum::<f32>() / n;
            let mean_b: f32 = projected.iter().map(|p| p[b]).sum::<f32>() / n;
            let covariance: f32 = projected
                .iter()
                .map(|p| (p[a] - mean_a) * (p[b] - mean_b))
                .sum::<f32>()
                / n;
            let var_a: f32 = projected.iter().map(|p| (p[a] - mean_a).powi(2)).sum::<f32>() / n;
            let var_b: f32 = projected.iter().map(|p| (p[b] - mean_b).powi(2)).sum::<f32>() / n;

            let correlation = covariance / (var_a.sqrt() * var_b.sqrt()).max(f32::EPSILON);
            assert!(
                correlation.abs() < 0.1,
                "components {a} and {b} correlate at {correlation}"
            );
        }
    }
}

#[test]
fn explained_variance_is_ordered_and_bounded() {
    let data = plane_in_high_dim(100, 32);
    let pca = PcaProjection::fit(&data, 3, 8).unwrap();
    let variance = pca.explained_variance();

    assert_eq!(variance.len(), 3);
    for window in variance.windows(2) {
        assert!(
            window[0] >= window[1] - 1e-5,
            "components out of order: {window:?}"
        );
    }
    assert!(variance.iter().all(|v| (0.0..=1.0).contains(v)));
    assert!(variance.iter().sum::<f32>() <= 1.01);
}

#[test]
fn works_at_the_embedding_dimension() {
    // Perch emits 1536-dimensional vectors; the whole point of randomized SVD
    // is that this stays cheap.
    let raw = pseudo_random(77, 200 * 1536);
    let data: Vec<Vec<f32>> = (0..200)
        .map(|i| raw[i * 1536..(i + 1) * 1536].to_vec())
        .collect();

    let start = std::time::Instant::now();
    let pca = PcaProjection::fit(&data, 3, 42).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(pca.dims(), 3);
    assert_eq!(pca.input_dim(), 1536);
    assert!(
        elapsed.as_secs() < 10,
        "fitting 200x1536 took {elapsed:?}, which suggests the full scatter matrix is being formed"
    );

    let projected = pca.project(&data[0]).unwrap();
    assert!(projected.iter().all(|v| v.is_finite()));
}

#[test]
fn normalization_maps_every_axis_into_the_unit_range() {
    let mut points = vec![
        vec![0.0, -50.0],
        vec![100.0, 50.0],
        vec![50.0, 0.0],
    ];

    let (min, max) = normalize_coordinates(&mut points);

    assert_eq!(min, vec![0.0, -50.0]);
    assert_eq!(max, vec![100.0, 50.0]);
    for point in &points {
        assert!(point.iter().all(|v| (-1.0..=1.0).contains(v)), "{point:?}");
    }
    assert_eq!(points[0], vec![-1.0, -1.0]);
    assert_eq!(points[1], vec![1.0, 1.0]);
    assert_eq!(points[2], vec![0.0, 0.0]);
}

#[test]
fn normalization_handles_an_axis_with_no_extent() {
    let mut points = vec![vec![5.0, 1.0], vec![5.0, 3.0]];
    normalize_coordinates(&mut points);

    // A constant axis collapses to zero rather than dividing by zero.
    assert_eq!(points[0][0], 0.0);
    assert_eq!(points[1][0], 0.0);
    assert_eq!(points[0][1], -1.0);
    assert_eq!(points[1][1], 1.0);
}

#[test]
fn normalization_of_an_empty_set_is_a_no_op() {
    let mut points: Vec<Vec<f32>> = Vec::new();
    let (min, max) = normalize_coordinates(&mut points);

    assert!(min.is_empty());
    assert!(max.is_empty());
}

#[test]
fn poincare_images_lie_strictly_inside_the_unit_ball() {
    // A boundary point is at infinite hyperbolic distance, so the map must never
    // reach it -- including for inputs far outside the ball.
    let points = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.5, 0.5, 0.5],
        vec![1.0, 1.0, 1.0],
        vec![100.0, -100.0, 50.0],
    ];

    for scale in [0.5, 1.0, 5.0] {
        for image in to_poincare_ball(&points, scale) {
            let norm = image.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(norm < 1.0, "norm {norm} reached the boundary at scale {scale}");
        }
    }
}

#[test]
fn poincare_mapping_preserves_direction_and_ordering() {
    let points = vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![3.0, 0.0]];
    let images = to_poincare_ball(&points, 1.0);

    // Direction is unchanged, so the second coordinate stays zero.
    assert!(images.iter().all(|p| p[1].abs() < 1e-6));
    // Radial ordering is preserved even though spacing compresses.
    assert!(images[0][0] < images[1][0]);
    assert!(images[1][0] < images[2][0]);
}

#[test]
fn poincare_images_yield_finite_geodesic_distances() {
    // The reason the radius is clamped: an image on the boundary is at infinite
    // distance, which would poison every downstream metric.
    let points = vec![
        vec![0.0, 0.0],
        vec![0.3, 0.4],
        vec![1000.0, -1000.0],
        vec![-5000.0, 0.1],
    ];
    let images = to_poincare_ball(&points, 2.0);

    for a in 0..images.len() {
        for b in 0..images.len() {
            // Curvature is negative for hyperbolic space.
            let distance = sevensense_vector::poincare_distance(&images[a], &images[b], -1.0);
            assert!(
                distance.is_finite(),
                "distance between images {a} and {b} was {distance}"
            );
        }
    }
}

#[test]
fn the_origin_maps_to_the_origin() {
    let images = to_poincare_ball(&[vec![0.0, 0.0, 0.0]], 1.0);
    assert_eq!(images[0], vec![0.0, 0.0, 0.0]);
}

#[test]
fn too_few_points_is_rejected() {
    let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let error = PcaProjection::fit(&data, 3, 0).unwrap_err();

    assert!(matches!(error, ProjectionError::TooFewPoints { .. }));
}

#[test]
fn inconsistent_vector_lengths_are_rejected() {
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0]];
    let error = PcaProjection::fit(&data, 1, 0).unwrap_err();

    assert_eq!(
        error,
        ProjectionError::DimensionMismatch {
            expected: 2,
            found: 1
        }
    );
}

#[test]
fn an_invalid_target_dimensionality_is_rejected() {
    let data = collinear(20);

    assert!(matches!(
        PcaProjection::fit(&data, 0, 0).unwrap_err(),
        ProjectionError::InvalidTarget { .. }
    ));
    assert!(matches!(
        PcaProjection::fit(&data, 10, 0).unwrap_err(),
        ProjectionError::InvalidTarget { .. }
    ));
}

#[test]
fn input_without_variance_is_rejected() {
    // Every point identical: there is no axis to find, and silently returning
    // arbitrary components would be worse than an error.
    let data = vec![vec![1.0, 1.0, 1.0]; 20];
    let error = PcaProjection::fit(&data, 2, 0).unwrap_err();

    assert_eq!(error, ProjectionError::DegenerateInput);
}

#[test]
fn projecting_a_wrongly_sized_vector_is_rejected() {
    let pca = PcaProjection::fit(&collinear(20), 2, 0).unwrap();
    let error = pca.project(&[1.0, 2.0]).unwrap_err();

    assert_eq!(
        error,
        ProjectionError::DimensionMismatch {
            expected: 3,
            found: 2
        }
    );
}

#[test]
fn a_fitted_projection_survives_a_serde_round_trip() {
    // The API serves fitted projections; they must reload unchanged so saved
    // viewports stay valid.
    let pca = PcaProjection::fit(&plane_in_high_dim(60, 16), 3, 17).unwrap();

    let json = serde_json::to_string(&pca).unwrap();
    let restored: PcaProjection = serde_json::from_str(&json).unwrap();

    assert_eq!(pca, restored);
    assert_eq!(
        pca.project(&vec![1.0; 16]).unwrap(),
        restored.project(&vec![1.0; 16]).unwrap()
    );
}
