//! Coherence field + boundary detection.
//!
//! Sensors/zones do not vote on an answer; they contribute to the *stability* of
//! a physical graph. Zones are nodes; edge weight is delta-pattern coherence
//! (quiet zones agree strongly; a zone whose physical state moved disagrees).
//! Dynamic min-cut then isolates the side that broke away — that is the moved
//! boundary, not a class label.

use ruvector_mincut::MinCutBuilder;

/// Where coherence broke this window.
#[derive(Debug, Clone, PartialEq)]
pub struct Boundary {
    /// The single most-changed zone (the headline `changed_boundary`).
    pub zone: String,
    /// Every zone on the changed side of the cut.
    pub side: Vec<String>,
    /// Cleanliness of the separation in `[0, 1]`: high = the changed side is
    /// weakly coupled to the rest (a sharp, coherent boundary).
    pub coherence: f32,
}

/// Detect the moved boundary from per-zone delta vectors (each vector is the
/// per-modality |delta| for that zone, in a fixed modality order).
pub fn detect_boundary(deltas: &[(String, Vec<f64>)]) -> Option<Boundary> {
    let k = deltas.len();
    if k == 0 {
        return None;
    }
    let norm = |v: &[f64]| -> f64 { v.iter().map(|x| x * x).sum::<f64>().sqrt() };
    if k == 1 {
        return Some(Boundary {
            zone: deltas[0].0.clone(),
            side: vec![deltas[0].0.clone()],
            coherence: 0.0,
        });
    }

    // Pairwise distances and the scale that maps distance -> coherence weight.
    let dist = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    };
    let mut max_d = 0.0f64;
    for i in 0..k {
        for j in (i + 1)..k {
            max_d = max_d.max(dist(&deltas[i].1, &deltas[j].1));
        }
    }
    let scale = if max_d > 1e-9 { max_d } else { 1.0 };
    const EPS: f64 = 1e-3;
    let weight = |a: &[f64], b: &[f64]| -> f64 {
        (1.0 - dist(a, b) / scale).max(EPS) // quiet-quiet ~1, outlier ~EPS
    };

    // Complete weighted graph over zones; global min cut isolates the outlier.
    let mut edges = Vec::with_capacity(k * (k - 1) / 2);
    for i in 0..k {
        for j in (i + 1)..k {
            edges.push((i as u64, j as u64, weight(&deltas[i].1, &deltas[j].1)));
        }
    }
    let mincut = MinCutBuilder::new()
        .exact()
        .with_edges(edges)
        .build()
        .ok()?;
    let result = mincut.min_cut();
    let (a, b) = result.partition?;
    // Changed side = the smaller partition (the part that broke away).
    let (changed, _rest) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if changed.is_empty() {
        return None;
    }

    let side: Vec<String> = changed
        .iter()
        .map(|&i| deltas[i as usize].0.clone())
        .collect();
    // Headline zone = largest-magnitude delta on the changed side.
    let zone = changed
        .iter()
        .max_by(|&&i, &&j| {
            norm(&deltas[i as usize].1)
                .partial_cmp(&norm(&deltas[j as usize].1))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|&i| deltas[i as usize].0.clone())
        .unwrap_or_else(|| side[0].clone());

    // Coherence = how weakly the changed side couples to the rest.
    let changed_set: std::collections::HashSet<usize> =
        changed.iter().map(|&i| i as usize).collect();
    let mut cross_sum = 0.0;
    let mut cross_n = 0;
    for i in 0..k {
        for j in (i + 1)..k {
            if changed_set.contains(&i) != changed_set.contains(&j) {
                cross_sum += weight(&deltas[i].1, &deltas[j].1);
                cross_n += 1;
            }
        }
    }
    let mean_cross = if cross_n > 0 {
        cross_sum / cross_n as f64
    } else {
        1.0
    };
    let coherence = (1.0 - mean_cross).clamp(0.0, 1.0) as f32;

    Some(Boundary {
        zone,
        side,
        coherence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolates_the_changed_zone() {
        // Three quiet zones, one (B) moved.
        let deltas = vec![
            ("A".to_string(), vec![0.0, 0.0, 0.0]),
            ("B".to_string(), vec![3.0, 2.0, 2.5]),
            ("C".to_string(), vec![0.0, 0.0, 0.0]),
            ("D".to_string(), vec![0.0, 0.0, 0.0]),
        ];
        let b = detect_boundary(&deltas).unwrap();
        assert_eq!(b.zone, "B");
        assert_eq!(b.side, vec!["B".to_string()]);
        assert!(b.coherence > 0.8, "coherence {}", b.coherence);
    }

    #[test]
    fn no_change_means_low_coherence_boundary() {
        let deltas = vec![
            ("A".to_string(), vec![0.0, 0.0]),
            ("B".to_string(), vec![0.0, 0.0]),
            ("C".to_string(), vec![0.0, 0.0]),
        ];
        let b = detect_boundary(&deltas).unwrap();
        // Everything agrees -> no clean boundary.
        assert!(b.coherence < 0.2, "coherence {}", b.coherence);
    }
}
