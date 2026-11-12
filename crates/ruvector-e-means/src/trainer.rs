use crate::{cosine_step, nearest, validate_rows, Config, Error, Finalization, Model};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

const INITIAL_COUNT: f64 = 1e-10;
const MIN_COUNT: f64 = 1e-300;
const REVIVAL_THRESHOLD: f64 = 0.1;
// BF16 epsilon, as used by Appendix H, even though this implementation is f32.
const SPLIT_DELTA: f64 = 1.0 / 128.0;

/// Train on finite flat row-major f32 vectors. The algorithm follows equations
/// (2)-(8) of arXiv:2611.09501v1, using exact CPU assignment. Shuffled partial
/// batches are included, so every row participates in every epoch. Scratch
/// memory is O(n + kd); no n-by-k distance matrix is allocated.
pub fn fit(data: &[f32], dimensions: usize, config: &Config) -> Result<Model, Error> {
    validate_rows(data, dimensions)?;
    let samples = data.len() / dimensions;
    let params = config.parameters(samples)?;
    let k = config.clusters;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut order: Vec<usize> = (0..samples).collect();
    shuffle(&mut order, &mut rng);
    let mut centroids = Vec::with_capacity(k * dimensions);
    for &i in &order[..k] {
        centroids.extend_from_slice(&data[i * dimensions..(i + 1) * dimensions]);
    }
    let mut counts = vec![INITIAL_COUNT; k];
    let mut sums: Vec<f64> = centroids
        .iter()
        .map(|&v| f64::from(v) * INITIAL_COUNT)
        .collect();
    let mut batch_counts = vec![0.0; k];
    let mut batch_sums = vec![0.0; sums.len()];
    let use_last_mean = config.finalization == Finalization::LastEpochMean;
    let mut last_counts = if use_last_mean {
        vec![0usize; k]
    } else {
        Vec::new()
    };
    let mut last_sums = if use_last_mean {
        vec![0.0; sums.len()]
    } else {
        Vec::new()
    };
    let mut revival_scratch = RevivalScratch::default();
    let mut updates = 0;
    let mut revived_clusters = 0;
    for epoch in 0..config.epochs {
        let record_last = use_last_mean && epoch + 1 == config.epochs;
        shuffle(&mut order, &mut rng);
        for batch in order.chunks(params.batch_size) {
            batch_counts.fill(0.0);
            batch_sums.fill(0.0);
            for &i in batch {
                let row = &data[i * dimensions..(i + 1) * dimensions];
                let (j, _) = nearest(row, &centroids, dimensions);
                batch_counts[j] += 1.0;
                let batch_sum = &mut batch_sums[j * dimensions..(j + 1) * dimensions];
                for (sum, &x) in batch_sum.iter_mut().zip(row) {
                    *sum += f64::from(x);
                }
                if record_last {
                    last_counts[j] += 1;
                    let last_sum = &mut last_sums[j * dimensions..(j + 1) * dimensions];
                    for (sum, &x) in last_sum.iter_mut().zip(row) {
                        *sum += f64::from(x);
                    }
                }
            }
            let step = cosine_step(params.peak_step, updates, params.updates);
            for j in 0..k {
                let count = (1.0 - step) * counts[j] + step * batch_counts[j];
                let denominator = count.max(MIN_COUNT);
                counts[j] = denominator;
                let range = j * dimensions..(j + 1) * dimensions;
                let sum_row = &mut sums[range.clone()];
                let centroid_row = &mut centroids[range.clone()];
                if count < MIN_COUNT {
                    // A peak step of 1 can erase empty-cluster statistics.
                    // Preserve that centroid with a negligible pseudo-count.
                    for (sum, centroid) in sum_row.iter_mut().zip(centroid_row) {
                        *sum = denominator * f64::from(*centroid);
                        *centroid = bounded_f32(*sum / denominator);
                    }
                } else {
                    for ((sum, &batch_sum), centroid) in
                        sum_row.iter_mut().zip(&batch_sums[range]).zip(centroid_row)
                    {
                        *sum = (1.0 - step) * *sum + step * batch_sum;
                        *centroid = bounded_f32(*sum / denominator);
                    }
                }
            }
            updates += 1;
            if updates % 5 == 0 {
                revived_clusters += revive(
                    &mut counts,
                    &mut sums,
                    &mut centroids,
                    dimensions,
                    &mut rng,
                    &mut revival_scratch,
                );
            }
        }
    }
    if use_last_mean {
        for (j, &count) in last_counts.iter().enumerate() {
            if count > 0 {
                for d in 0..dimensions {
                    centroids[j * dimensions + d] =
                        bounded_f32(last_sums[j * dimensions + d] / count as f64);
                }
            }
        }
    }
    let mut assignments = Vec::with_capacity(samples);
    let mut loss = 0.0;
    for row in data.chunks_exact(dimensions) {
        let (j, distance) = nearest(row, &centroids, dimensions);
        assignments.push(j);
        loss += distance;
    }
    Ok(Model {
        centroids,
        dimensions,
        assignments,
        mse: loss / samples as f64,
        batch_size: params.batch_size,
        peak_step: params.peak_step,
        updates,
        revived_clusters,
    })
}

fn bounded_f32(value: f64) -> f32 {
    value.clamp(-f64::from(f32::MAX), f64::from(f32::MAX)) as f32
}

fn shuffle(order: &mut [usize], rng: &mut ChaCha8Rng) {
    for i in (1..order.len()).rev() {
        let limit = (i + 1) as u64;
        // Rejection sampling avoids modulo bias, including on wasm32.
        let threshold = limit.wrapping_neg() % limit;
        let value = loop {
            let value = rng.next_u64();
            if value >= threshold {
                break value;
            }
        };
        order.swap(i, (value % limit) as usize);
    }
}

#[derive(Default)]
struct RevivalScratch {
    dead: Vec<usize>,
    available: Vec<usize>,
}

fn revive(
    counts: &mut [f64],
    sums: &mut [f64],
    centroids: &mut [f32],
    dimensions: usize,
    rng: &mut ChaCha8Rng,
    scratch: &mut RevivalScratch,
) -> usize {
    let threshold = REVIVAL_THRESHOLD * counts.iter().sum::<f64>() / counts.len() as f64;
    scratch.dead.clear();
    scratch
        .dead
        .extend((0..counts.len()).filter(|&j| counts[j] < threshold));
    if scratch.dead.is_empty() {
        return 0;
    }
    // Select live donors proportional to their EMA count, without replacement.
    // Newly revived slots cannot donate again in the same revival pass.
    let available = &mut scratch.available;
    available.clear();
    available.extend((0..counts.len()).filter(|&j| counts[j] >= threshold));
    let mut revived = 0;
    for &j in &scratch.dead {
        let mass: f64 = available.iter().map(|&i| counts[i]).sum();
        if mass <= 0.0 {
            break;
        }
        let unit = (rng.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        let mut pick = unit * mass;
        let mut donor_pos = available.len() - 1;
        for (pos, &i) in available.iter().enumerate() {
            pick -= counts[i];
            if pick <= 0.0 {
                donor_pos = pos;
                break;
            }
        }
        let donor = available.remove(donor_pos);
        counts[donor] *= 0.5;
        counts[j] = counts[donor];
        for d in 0..dimensions {
            let shift = if d % 2 == 0 {
                SPLIT_DELTA
            } else {
                -SPLIT_DELTA
            };
            let value = f64::from(centroids[donor * dimensions + d]);
            centroids[j * dimensions + d] = bounded_f32(value * (1.0 + shift));
            centroids[donor * dimensions + d] = bounded_f32(value * (1.0 - shift));
            sums[j * dimensions + d] = f64::from(centroids[j * dimensions + d]) * counts[j];
            sums[donor * dimensions + d] =
                f64::from(centroids[donor * dimensions + d]) * counts[donor];
        }
        revived += 1;
    }
    revived
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn donor_split_preserves_mass_and_weighted_sum() {
        let mut counts = vec![10.0, 0.0];
        let mut sums = vec![40.0, 80.0, 0.0, 0.0];
        let mut centroids = vec![4.0, 8.0, 0.0, 0.0];
        let n = revive(
            &mut counts,
            &mut sums,
            &mut centroids,
            2,
            &mut ChaCha8Rng::seed_from_u64(2),
            &mut RevivalScratch::default(),
        );
        assert_eq!(n, 1);
        assert_eq!(counts, [5.0, 5.0]);
        assert_eq!(sums[0] + sums[2], 40.0);
        assert_eq!(sums[1] + sums[3], 80.0);
        assert_ne!(&centroids[..2], &centroids[2..]);
    }
}
