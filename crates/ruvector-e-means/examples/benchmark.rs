//! Portable native fit/predict benchmark. No extra dependencies or CPU flags.
//!
//! cargo run -p ruvector-e-means --release --example benchmark --
//!   --case all --repeats 5 --seed 42 --finalization both
//!
//! Dataset creation and warmup are excluded from reported median timings.

use ruvector_e_means::{fit, Config, Finalization};
use std::{hint::black_box, time::Instant};

#[derive(Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    pub samples: usize,
    pub clusters: usize,
    pub dimensions: usize,
}

pub const CASES: [Case; 4] = [
    Case {
        name: "embedding128",
        samples: 4096,
        clusters: 64,
        dimensions: 128,
    },
    Case {
        name: "embedding768",
        samples: 2048,
        clusters: 32,
        dimensions: 768,
    },
    Case {
        name: "pq8",
        samples: 8192,
        clusters: 256,
        dimensions: 8,
    },
    Case {
        name: "revival",
        samples: 2048,
        clusters: 64,
        dimensions: 32,
    },
];

/// Deterministic synthetic inputs, not a reproduction of the paper's datasets.
/// Cases cover clustered and diffuse normalized embeddings, diffuse low-D
/// codebook training, and duplicate/zero rows that exercise cluster revival.
pub fn generate(case: Case, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut random = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^= value >> 31;
        ((value >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    };
    let centers: Vec<f32> = (0..case.clusters * case.dimensions)
        .map(|_| random())
        .collect();
    let mut data = Vec::with_capacity(case.samples * case.dimensions);
    for i in 0..case.samples {
        for d in 0..case.dimensions {
            let value = match case.name {
                "embedding128" => {
                    centers[(i % case.clusters) * case.dimensions + d] + random() * 0.1
                }
                "revival" if i % 5 == 0 => 0.0,
                "revival" => ((i % 4) as f32 - 1.5) * if d % 2 == 0 { 1.0 } else { -1.0 },
                _ => random(),
            };
            data.push(value);
        }
        if case.name.starts_with("embedding") {
            let row = &mut data[i * case.dimensions..(i + 1) * case.dimensions];
            let norm = row
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt();
            for value in row {
                *value = (f64::from(*value) / norm) as f32;
            }
        }
    }
    data
}

pub fn digest(centroids: &[f32], assignments: &[usize]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for bits in centroids
        .iter()
        .map(|v| u64::from(v.to_bits()))
        .chain(assignments.iter().map(|&v| v as u64))
    {
        for byte in bits.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    hash
}

pub fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[middle - 1] + samples[middle]) / 2.0
    } else {
        samples[middle]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut case_name = String::from("all");
    let mut repeats = 5usize;
    let mut seed = 42u64;
    let mut finalization = String::from("both");
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!("benchmark [--case all|embedding128|embedding768|pq8|revival] [--repeats 1..99] [--seed u64] [--finalization both|lastEpochMean|ema]");
            return Ok(());
        }
        let value = args.next().ok_or("every option requires a value")?;
        match flag.as_str() {
            "--case" => case_name = value,
            "--repeats" => repeats = value.parse()?,
            "--seed" => seed = value.parse()?,
            "--finalization" => finalization = value,
            _ => return Err(format!("unknown option: {flag}").into()),
        }
    }
    if !(1..=99).contains(&repeats)
        || !["both", "lastEpochMean", "ema"].contains(&finalization.as_str())
    {
        return Err("invalid repeats or finalization".into());
    }
    let cases: Vec<Case> = CASES
        .into_iter()
        .filter(|case| case_name == "all" || case.name == case_name)
        .collect();
    if cases.is_empty() {
        return Err("unknown benchmark case".into());
    }
    for case in cases {
        let data = generate(case, seed);
        for (label, mode) in [
            ("lastEpochMean", Finalization::LastEpochMean),
            ("ema", Finalization::Ema),
        ] {
            if finalization != "both" && finalization != label {
                continue;
            }
            let config = Config {
                clusters: case.clusters,
                seed,
                finalization: mode,
                ..Config::default()
            };
            let model = fit(black_box(&data), case.dimensions, &config)?;
            black_box(model.predict(black_box(&data))?);
            let mut fit_ms = Vec::with_capacity(repeats);
            let mut predict_ms = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let start = Instant::now();
                black_box(fit(black_box(&data), case.dimensions, black_box(&config))?);
                fit_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                let start = Instant::now();
                black_box(model.predict(black_box(&data))?);
                predict_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            println!("{{\"case\":\"{}\",\"samples\":{},\"clusters\":{},\"dimensions\":{},\"seed\":{},\"epochs\":{},\"repeats\":{},\"finalization\":\"{}\",\"fitMedianMs\":{:.6},\"predictMedianMs\":{:.6},\"mse\":{:.12},\"revivedClusters\":{},\"digest\":\"{:016x}\"}}", case.name, case.samples, case.clusters, case.dimensions, seed, config.epochs, repeats, label, median_ms(&mut fit_ms), median_ms(&mut predict_ms), model.mse, model.revived_clusters, digest(&model.centroids, &model.assignments));
        }
    }
    Ok(())
}
