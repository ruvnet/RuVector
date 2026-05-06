//! ruview-lora-init — ADR-183 Tier 3 iter 19
//!
//! Generates a properly zero-initialized LoRA adapter JSON (`node-N.json`)
//! for the SONA online-adaptation pipeline.
//!
//! Standard LoRA init: loraB = zeros, loraA = Gaussian(0, std).
//! With loraB=0 the initial delta is exactly 0, so the output equals the
//! base model. SONA then learns from live vitals data to adapt loraB.
//!
//! Usage:
//!   ruview-lora-init --node 1 --out /usr/local/share/ruvector/node-1.json
//!   ruview-lora-init --node 2 --out /usr/local/share/ruvector/node-2.json

use std::io::Write as _;

const EMBED_DIM: usize = 128;
const RANK: usize = 4;
const DEFAULT_SCALING: f32 = 2.0;
const LORA_A_STD: f32 = 0.02;

fn usage() -> ! {
    eprintln!("usage: ruview-lora-init --node N [--out PATH] [--scaling F] [--seed U]");
    std::process::exit(1);
}

fn lcg_next(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = (*s >> 33) as u32;
    // Box-Muller via two uniform samples (second half of state)
    let u1 = (bits as f32 + 0.5) / (u32::MAX as f32 + 1.0);
    u1
}

/// Approximate standard normal via Box-Muller (two LCG samples).
fn randn(s: &mut u64) -> f32 {
    let u1 = lcg_next(s).max(1e-8);
    let u2 = lcg_next(s);
    // Box-Muller
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f32::consts::TAU * u2;
    r * theta.cos()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut node: Option<u32> = None;
    let mut out_path: Option<String> = None;
    let mut scaling = DEFAULT_SCALING;
    let mut seed = 0x8c37_91c5_dead_beefu64;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--node" => { i += 1; node = args[i].parse().ok(); }
            "--out"  => { i += 1; out_path = Some(args[i].clone()); }
            "--scaling" => { i += 1; scaling = args[i].parse().unwrap_or(DEFAULT_SCALING); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap_or(seed); }
            "--help" | "-h" => usage(),
            _ => {}
        }
        i += 1;
    }

    let node_id = node.unwrap_or_else(|| { eprintln!("--node N required"); usage() });
    let path = out_path.unwrap_or_else(|| {
        format!("/usr/local/share/ruvector/node-{node_id}.json")
    });

    // Mix node_id into seed so each node gets different loraA init
    seed ^= (node_id as u64) * 0x517cc1b727220a95;

    // loraA: [EMBED_DIM × RANK] = [128 × 4] — small Gaussian noise
    let mut lora_a_rows = Vec::with_capacity(EMBED_DIM);
    for _ in 0..EMBED_DIM {
        let row: Vec<f32> = (0..RANK).map(|_| randn(&mut seed) * LORA_A_STD).collect();
        lora_a_rows.push(row);
    }

    // loraB: [RANK × EMBED_DIM] = [4 × 128] — all zeros (standard LoRA init)
    let lora_b_rows: Vec<Vec<f32>> = (0..RANK)
        .map(|_| vec![0.0f32; EMBED_DIM])
        .collect();

    // Serialise as compact JSON
    let mut out = String::with_capacity(64 * 1024);
    out.push_str("{\"config\":{\"rank\":");
    out.push_str(&RANK.to_string());
    out.push_str(",\"alpha\":");
    out.push_str(&(RANK * 2).to_string());
    out.push_str("},\"inputDim\":");
    out.push_str(&EMBED_DIM.to_string());
    out.push_str(",\"outputDim\":");
    out.push_str(&EMBED_DIM.to_string());
    out.push_str(",\"sona\":{\"step\":0,\"lr\":1e-4,\"beta1\":0.9,\"beta2\":0.999},");
    out.push_str("\"weights\":{\"loraA\":");
    push_matrix(&mut out, &lora_a_rows);
    out.push_str(",\"loraB\":");
    push_matrix(&mut out, &lora_b_rows);
    out.push_str(",\"scaling\":");
    // Write scaling as decimal
    out.push_str(&format!("{:.1}", scaling));
    out.push_str("}}");

    let mut f = std::fs::File::create(&path)
        .unwrap_or_else(|e| { eprintln!("cannot create {path}: {e}"); std::process::exit(1) });
    f.write_all(out.as_bytes())
        .unwrap_or_else(|e| { eprintln!("write error: {e}"); std::process::exit(1) });

    let size = out.len();
    eprintln!("Wrote {path} ({size} bytes) — node={node_id} loraA=Gaussian({LORA_A_STD}) loraB=zeros scaling={scaling}");
}

fn push_matrix(out: &mut String, rows: &[Vec<f32>]) {
    out.push('[');
    for (ri, row) in rows.iter().enumerate() {
        out.push('[');
        for (ci, v) in row.iter().enumerate() {
            if *v == 0.0 {
                out.push_str("0.0");
            } else {
                out.push_str(&format!("{:.8e}", v));
            }
            if ci + 1 < row.len() { out.push(','); }
        }
        out.push(']');
        if ri + 1 < rows.len() { out.push(','); }
    }
    out.push(']');
}
