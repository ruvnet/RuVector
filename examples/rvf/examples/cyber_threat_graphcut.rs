//! Cybersecurity Network Threat Detection via Graph Cut + RuVector
//!
//! MRF/mincut optimization for network intrusion detection:
//!   1. Generate ~2000 synthetic network flows (24h period, ~4% attack rate)
//!   2. Inject realistic attack patterns: PortScan, BruteForce, Exfiltration,
//!      C2Beacon, DDoS, LateralMovement (CICIDS2017/NSL-KDD inspired)
//!   3. Extract 32-dim flow embeddings, build temporal+similarity graph, mincut
//!   4. Evaluate: precision, recall, F1, FPR; per-attack detection rates
//!
//! Run: cargo run --example cyber_threat_graphcut --release

use rvf_crypto::{create_witness_chain, shake256_256, verify_witness_chain, WitnessEntry};
use rvf_runtime::filter::FilterValue;
use rvf_runtime::options::DistanceMetric;
use rvf_runtime::{FilterExpr, MetadataEntry, MetadataValue, QueryOptions, RvfOptions, RvfStore};
use rvf_types::DerivationType;
use tempfile::TempDir;

const DIM: usize = 32;
const N_FLOWS: usize = 2000;
const FIELD_PROTOCOL: u16 = 0;
const FIELD_PORT_CAT: u16 = 1;
const FIELD_SUBNET: u16 = 2;
const FIELD_ATTACK: u16 = 3;

fn lcg_next(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}
fn lcg_f64(s: &mut u64) -> f64 {
    lcg_next(s);
    (*s >> 11) as f64 / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg_f64(s).max(1e-15);
    let u2 = lcg_f64(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Attack {
    Normal,
    PortScan,
    BruteForce,
    Exfiltration,
    C2Beacon,
    DDoS,
    LateralMovement,
}
impl Attack {
    fn label(&self) -> &'static str {
        match self {
            Attack::Normal => "normal",
            Attack::PortScan => "portscan",
            Attack::BruteForce => "bruteforce",
            Attack::Exfiltration => "exfil",
            Attack::C2Beacon => "c2beacon",
            Attack::DDoS => "ddos",
            Attack::LateralMovement => "lateral",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Protocol {
    Tcp,
    Udp,
    Icmp,
}
impl Protocol {
    fn label(&self) -> &'static str {
        match self {
            Protocol::Tcp => "TCP",
            Protocol::Udp => "UDP",
            Protocol::Icmp => "ICMP",
        }
    }
}

fn port_category(port: u16) -> &'static str {
    match port {
        80 | 443 | 8080 | 8443 => "web",
        22 => "ssh",
        53 => "dns",
        3389 => "rdp",
        0 => "none",
        _ => "other",
    }
}

#[derive(Debug, Clone)]
struct Flow {
    index: usize,
    time_s: f64, // seconds into 24h window
    duration_ms: f64,
    bytes_sent: u64,
    bytes_recv: u64,
    pkts_sent: u32,
    pkts_recv: u32,
    protocol: Protocol,
    dst_port: u16,
    src_subnet: u8, // 10.x.0.0/16 subnet id (0-4 internal)
    dst_subnet: u8,
    is_internal_dst: bool,
    syn_flag: bool,
    ack_flag: bool,
    rst_flag: bool,
    psh_flag: bool,
    fin_flag: bool,
    payload_entropy: f64, // 0-8 bits
    truth: Attack,
}

fn generate_flows(n: usize, seed: u64) -> Vec<Flow> {
    let mut rng = seed;
    let mut flows = Vec::with_capacity(n);
    let ports = [
        80u16, 443, 22, 53, 3389, 8080, 8443, 25, 110, 993, 3306, 5432,
    ];

    for i in 0..n {
        let time_s = lcg_f64(&mut rng) * 86400.0;
        let hour = (time_s / 3600.0) as u32;
        // More traffic during business hours
        let biz_mult = if (8..18).contains(&hour) { 1.0 } else { 0.4 };
        let _ = biz_mult; // used implicitly via attack injection timing

        let proto_r = lcg_f64(&mut rng);
        let protocol = if proto_r < 0.75 {
            Protocol::Tcp
        } else if proto_r < 0.92 {
            Protocol::Udp
        } else {
            Protocol::Icmp
        };

        let port_idx = (lcg_next(&mut rng) % ports.len() as u64) as usize;
        let dst_port = if protocol == Protocol::Icmp {
            0
        } else {
            ports[port_idx]
        };
        let src_subnet = (lcg_next(&mut rng) % 5) as u8;
        let dst_r = lcg_f64(&mut rng);
        let is_internal = dst_r < 0.3;
        let dst_subnet = if is_internal {
            (lcg_next(&mut rng) % 5) as u8
        } else {
            100 + (lcg_next(&mut rng) % 50) as u8
        };

        let duration_ms =
            (50.0 + lcg_f64(&mut rng) * 2000.0 + lcg_normal(&mut rng).abs() * 500.0).max(1.0);
        let bytes_sent = (200.0 + lcg_f64(&mut rng) * 5000.0 + lcg_normal(&mut rng).abs() * 1000.0)
            .max(40.0) as u64;
        let bytes_recv = (500.0 + lcg_f64(&mut rng) * 20000.0 + lcg_normal(&mut rng).abs() * 3000.0)
            .max(40.0) as u64;
        let pkts_sent =
            (1.0 + bytes_sent as f64 / (400.0 + lcg_f64(&mut rng) * 800.0)).max(1.0) as u32;
        let pkts_recv =
            (1.0 + bytes_recv as f64 / (400.0 + lcg_f64(&mut rng) * 800.0)).max(1.0) as u32;
        let entropy = 4.0 + lcg_normal(&mut rng) * 1.0;

        flows.push(Flow {
            index: i,
            time_s,
            duration_ms,
            bytes_sent,
            bytes_recv,
            pkts_sent,
            pkts_recv,
            protocol,
            dst_port,
            src_subnet,
            dst_subnet,
            is_internal_dst: is_internal,
            syn_flag: lcg_f64(&mut rng) < 0.6,
            ack_flag: lcg_f64(&mut rng) < 0.8,
            rst_flag: lcg_f64(&mut rng) < 0.05,
            psh_flag: lcg_f64(&mut rng) < 0.3,
            fin_flag: lcg_f64(&mut rng) < 0.4,
            payload_entropy: entropy.clamp(0.0, 8.0),
            truth: Attack::Normal,
        });
    }

    // Inject attacks (~4% = ~80 flows)
    // PortScan: rapid small packets to many ports from one source
    let scan_src = 2u8;
    let scan_start = 14400.0; // 4am
    for j in 0..15 {
        let idx = 50 + j * 3;
        if idx >= n {
            break;
        }
        let f = &mut flows[idx];
        f.truth = Attack::PortScan;
        f.time_s = scan_start + j as f64 * 0.5;
        f.duration_ms = 2.0 + lcg_f64(&mut rng) * 10.0;
        f.bytes_sent = 40 + (lcg_next(&mut rng) % 60) as u64;
        f.bytes_recv = 0;
        f.pkts_sent = 1;
        f.pkts_recv = 0;
        f.protocol = Protocol::Tcp;
        f.dst_port = 1024 + (lcg_next(&mut rng) % 64000) as u16;
        f.src_subnet = scan_src;
        f.syn_flag = true;
        f.ack_flag = false;
        f.rst_flag = false;
        f.payload_entropy = 0.5 + lcg_f64(&mut rng) * 0.5;
    }

    // BruteForce: repeated SSH/RDP from one source, many RSTs
    let bf_src = 3u8;
    for j in 0..12 {
        let idx = 200 + j * 2;
        if idx >= n {
            break;
        }
        let f = &mut flows[idx];
        f.truth = Attack::BruteForce;
        f.time_s = 28800.0 + j as f64 * 2.0; // 8am
        f.duration_ms = 100.0 + lcg_f64(&mut rng) * 200.0;
        f.bytes_sent = 80 + (lcg_next(&mut rng) % 200) as u64;
        f.bytes_recv = 60 + (lcg_next(&mut rng) % 100) as u64;
        f.pkts_sent = 3 + (lcg_next(&mut rng) % 5) as u32;
        f.pkts_recv = 2;
        f.protocol = Protocol::Tcp;
        f.dst_port = if lcg_f64(&mut rng) < 0.7 { 22 } else { 3389 };
        f.src_subnet = bf_src;
        f.dst_subnet = 120;
        f.is_internal_dst = false;
        f.syn_flag = true;
        f.rst_flag = lcg_f64(&mut rng) < 0.8;
        f.payload_entropy = 2.0 + lcg_f64(&mut rng) * 1.5;
    }

    // Exfiltration: large outbound at night
    for j in 0..10 {
        let idx = 400 + j * 4;
        if idx >= n {
            break;
        }
        let f = &mut flows[idx];
        f.truth = Attack::Exfiltration;
        f.time_s = 7200.0 + j as f64 * 120.0; // 2-3am
        f.duration_ms = 5000.0 + lcg_f64(&mut rng) * 15000.0;
        f.bytes_sent = 500_000 + (lcg_next(&mut rng) % 2_000_000) as u64;
        f.bytes_recv = 200 + (lcg_next(&mut rng) % 500) as u64;
        f.pkts_sent = 300 + (lcg_next(&mut rng) % 500) as u32;
        f.pkts_recv = 5;
        f.protocol = Protocol::Tcp;
        f.dst_port = 443;
        f.src_subnet = 1;
        f.dst_subnet = 130;
        f.is_internal_dst = false;
        f.psh_flag = true;
        f.payload_entropy = 7.2 + lcg_f64(&mut rng) * 0.6;
    }

    // C2Beacon: periodic small payloads to same external IP
    let c2_dst = 140u8;
    for j in 0..15 {
        let idx = 600 + j * 5;
        if idx >= n {
            break;
        }
        let f = &mut flows[idx];
        f.truth = Attack::C2Beacon;
        f.time_s = 3600.0 * j as f64 / 15.0 * 24.0; // spread across day
        f.duration_ms = 50.0 + lcg_f64(&mut rng) * 100.0;
        f.bytes_sent = 80 + (lcg_next(&mut rng) % 150) as u64;
        f.bytes_recv = 60 + (lcg_next(&mut rng) % 120) as u64;
        f.pkts_sent = 1;
        f.pkts_recv = 1;
        f.protocol = Protocol::Tcp;
        f.dst_port = 443;
        f.src_subnet = 0;
        f.dst_subnet = c2_dst;
        f.is_internal_dst = false;
        f.payload_entropy = 6.5 + lcg_f64(&mut rng) * 1.0;
    }

    // DDoS: high packet rate from many sources to single target
    for j in 0..18 {
        let idx = 900 + j * 2;
        if idx >= n {
            break;
        }
        let f = &mut flows[idx];
        f.truth = Attack::DDoS;
        f.time_s = 43200.0 + j as f64 * 0.3; // noon burst
        f.duration_ms = 1.0 + lcg_f64(&mut rng) * 5.0;
        f.bytes_sent = 40 + (lcg_next(&mut rng) % 80) as u64;
        f.bytes_recv = 0;
        f.pkts_sent = 50 + (lcg_next(&mut rng) % 200) as u32;
        f.pkts_recv = 0;
        f.protocol = if lcg_f64(&mut rng) < 0.6 {
            Protocol::Udp
        } else {
            Protocol::Tcp
        };
        f.dst_port = 80;
        f.src_subnet = (lcg_next(&mut rng) % 5) as u8;
        f.dst_subnet = 110;
        f.is_internal_dst = false;
        f.syn_flag = true;
        f.ack_flag = false;
        f.payload_entropy = 1.0 + lcg_f64(&mut rng) * 1.0;
    }

    // LateralMovement: internal-to-internal, unusual ports
    for j in 0..10 {
        let idx = 1200 + j * 3;
        if idx >= n {
            break;
        }
        let f = &mut flows[idx];
        f.truth = Attack::LateralMovement;
        f.time_s = 50400.0 + j as f64 * 60.0; // 2pm
        f.duration_ms = 200.0 + lcg_f64(&mut rng) * 1000.0;
        f.bytes_sent = 1000 + (lcg_next(&mut rng) % 10000) as u64;
        f.bytes_recv = 500 + (lcg_next(&mut rng) % 5000) as u64;
        f.pkts_sent = 10 + (lcg_next(&mut rng) % 30) as u32;
        f.pkts_recv = 8 + (lcg_next(&mut rng) % 20) as u32;
        f.protocol = Protocol::Tcp;
        f.dst_port = 445 + (lcg_next(&mut rng) % 100) as u16; // SMB-ish
        f.src_subnet = j as u8 % 5;
        f.dst_subnet = (j as u8 + 1) % 5;
        f.is_internal_dst = true;
        f.psh_flag = true;
        f.payload_entropy = 5.5 + lcg_f64(&mut rng) * 1.5;
    }

    flows.sort_by(|a, b| a.time_s.partial_cmp(&b.time_s).unwrap());
    for (i, f) in flows.iter_mut().enumerate() {
        f.index = i;
    }
    flows
}

fn extract_embedding(f: &Flow) -> Vec<f32> {
    let log_dur = (f.duration_ms.max(1.0)).ln() as f32 / 10.0;
    let log_bs = (f.bytes_sent.max(1) as f64).ln() as f32 / 15.0;
    let log_br = (f.bytes_recv.max(1) as f64).ln() as f32 / 15.0;
    let pkt_ratio = if f.pkts_sent + f.pkts_recv > 0 {
        f.pkts_sent as f32 / (f.pkts_sent + f.pkts_recv) as f32
    } else {
        0.5
    };
    // Protocol one-hot
    let (p_tcp, p_udp, p_icmp) = match f.protocol {
        Protocol::Tcp => (1.0f32, 0.0, 0.0),
        Protocol::Udp => (0.0, 1.0, 0.0),
        Protocol::Icmp => (0.0, 0.0, 1.0),
    };
    // Port category one-hot
    let pc = port_category(f.dst_port);
    let (pw, ps, pd, pr, po) = match pc {
        "web" => (1.0f32, 0.0, 0.0, 0.0, 0.0),
        "ssh" => (0.0, 1.0, 0.0, 0.0, 0.0),
        "dns" => (0.0, 0.0, 1.0, 0.0, 0.0),
        "rdp" => (0.0, 0.0, 0.0, 1.0, 0.0),
        _ => (0.0, 0.0, 0.0, 0.0, 1.0),
    };
    let internal = if f.is_internal_dst { 1.0f32 } else { 0.0 };
    let subnet_enc = f.src_subnet as f32 / 5.0;
    // Time cyclic
    let hour_rad = f.time_s as f32 / 86400.0 * 2.0 * std::f32::consts::PI;
    let t_sin = hour_rad.sin();
    let t_cos = hour_rad.cos();
    let dow = 0.5f32; // mid-week placeholder
    let entropy = f.payload_entropy as f32 / 8.0;
    let avg_pkt = if f.pkts_sent + f.pkts_recv > 0 {
        (f.bytes_sent + f.bytes_recv) as f32 / (f.pkts_sent + f.pkts_recv) as f32 / 1500.0
    } else {
        0.0
    };
    let bpp = if f.bytes_sent > 0 && f.pkts_sent > 0 {
        f.bytes_sent as f32 / f.pkts_sent as f32 / 1500.0
    } else {
        0.0
    };
    // Flag pattern features
    let syn_flood = if f.syn_flag && !f.ack_flag {
        1.0f32
    } else {
        0.0
    };
    let rst_ind = if f.rst_flag { 1.0f32 } else { 0.0 };
    // Derived
    let vol = log_bs + log_br;
    let asym = (log_bs - log_br).abs();

    vec![
        log_dur,
        log_bs,
        log_br,
        pkt_ratio,
        p_tcp,
        p_udp,
        p_icmp,
        pw,
        ps,
        pd,
        pr,
        po,
        internal,
        subnet_enc,
        t_sin,
        t_cos,
        dow,
        entropy,
        avg_pkt,
        bpp,
        syn_flood,
        rst_ind,
        vol,
        asym,
        f.pkts_sent as f32 / 100.0,
        f.pkts_recv as f32 / 100.0,
        (f.duration_ms as f32 / 1000.0).min(5.0),
        (f.bytes_sent as f64 / f.duration_ms.max(1.0)) as f32 / 1000.0,
        entropy * syn_flood,
        entropy * asym,
        log_dur * rst_ind,
        vol * internal,
    ]
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len().min(b.len()) {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    let dn = na.sqrt() * nb.sqrt();
    if dn < 1e-15 {
        0.0
    } else {
        d / dn
    }
}

fn anomaly_score(f: &Flow) -> f64 {
    let vol = ((f.bytes_sent + f.bytes_recv).max(1) as f64).ln();
    let ent = f.payload_entropy;
    let dur = (f.duration_ms.max(1.0)).ln();
    let pps = if f.duration_ms > 0.0 {
        f.pkts_sent as f64 / (f.duration_ms / 1000.0)
    } else {
        0.0
    };
    let syn_no_ack = if f.syn_flag && !f.ack_flag { 1.0 } else { 0.0 };
    let rst_p = if f.rst_flag { 0.5 } else { 0.0 };
    let night = if f.time_s < 21600.0 || f.time_s > 79200.0 {
        0.3
    } else {
        0.0
    };
    let high_vol = if vol > 12.0 { (vol - 12.0) * 0.3 } else { 0.0 };
    let low_pkt = if f.bytes_sent > 0 && f.pkts_sent <= 1 && f.bytes_recv == 0 {
        0.4
    } else {
        0.0
    };
    let high_pps = if pps > 50.0 { (pps - 50.0) * 0.01 } else { 0.0 };
    let high_ent = if ent > 6.5 { (ent - 6.5) * 0.5 } else { 0.0 };
    let short_burst = if dur < 2.0 && f.pkts_sent > 10 {
        0.5
    } else {
        0.0
    };

    let internal_to_internal = if f.src_subnet == f.dst_subnet && f.dst_port > 1024 {
        0.3
    } else {
        0.0
    };
    0.3 * syn_no_ack
        + rst_p
        + night
        + high_vol
        + low_pkt
        + high_pps
        + high_ent
        + short_burst
        + internal_to_internal
        - 0.35
}

struct Edge {
    from: usize,
    to: usize,
    weight: f64,
}

fn build_graph(flows: &[Flow], embs: &[Vec<f32>], alpha: f64, beta: f64, k: usize) -> Vec<Edge> {
    let m = flows.len();
    let mut edges = Vec::new();

    // Temporal chain: consecutive flows from same source subnet
    for i in 1..m {
        if flows[i].src_subnet == flows[i - 1].src_subnet {
            let dt = (flows[i].time_s - flows[i - 1].time_s).abs();
            let tw = alpha * (-dt / 300.0).exp(); // decay over 5min
            if tw > 1e-6 {
                edges.push(Edge {
                    from: i - 1,
                    to: i,
                    weight: tw,
                });
                edges.push(Edge {
                    from: i,
                    to: i - 1,
                    weight: tw,
                });
            }
        }
    }

    // Same-destination smoothing
    for i in 0..m {
        for j in (i + 1)..m.min(i + 50) {
            if flows[i].dst_subnet == flows[j].dst_subnet && flows[i].dst_port == flows[j].dst_port
            {
                edges.push(Edge {
                    from: i,
                    to: j,
                    weight: alpha * 0.5,
                });
                edges.push(Edge {
                    from: j,
                    to: i,
                    weight: alpha * 0.5,
                });
            }
        }
    }

    // kNN similarity from embeddings
    for i in 0..m {
        let mut sims: Vec<(usize, f64)> = (0..m)
            .filter(|&j| j != i && (j as isize - i as isize).unsigned_abs() > 3)
            .map(|j| (j, cosine_sim(&embs[i], &embs[j]).max(0.0)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(j, s) in sims.iter().take(k) {
            if s > 0.1 {
                edges.push(Edge {
                    from: i,
                    to: j,
                    weight: beta * s,
                });
            }
        }
    }

    edges
}

fn solve_mincut(lambdas: &[f64], edges: &[Edge], gamma: f64) -> Vec<bool> {
    let m = lambdas.len();
    let (s, t, n) = (m, m + 1, m + 2);
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut caps: Vec<f64> = Vec::new();
    let ae =
        |adj: &mut Vec<Vec<(usize, usize)>>, caps: &mut Vec<f64>, u: usize, v: usize, c: f64| {
            let idx = caps.len();
            caps.push(c);
            caps.push(0.0);
            adj[u].push((v, idx));
            adj[v].push((u, idx + 1));
        };
    for i in 0..m {
        let p0 = lambdas[i].max(0.0);
        let p1 = (-lambdas[i]).max(0.0);
        if p0 > 1e-12 {
            ae(&mut adj, &mut caps, s, i, p0);
        }
        if p1 > 1e-12 {
            ae(&mut adj, &mut caps, i, t, p1);
        }
    }
    for e in edges {
        let c = gamma * e.weight;
        if c > 1e-12 {
            ae(&mut adj, &mut caps, e.from, e.to, c);
        }
    }
    loop {
        let mut par: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut vis = vec![false; n];
        let mut q = std::collections::VecDeque::new();
        vis[s] = true;
        q.push_back(s);
        while let Some(u) = q.pop_front() {
            if u == t {
                break;
            }
            for &(v, ei) in &adj[u] {
                if !vis[v] && caps[ei] > 1e-15 {
                    vis[v] = true;
                    par[v] = Some((u, ei));
                    q.push_back(v);
                }
            }
        }
        if !vis[t] {
            break;
        }
        let mut bn = f64::MAX;
        let mut v = t;
        while let Some((u, ei)) = par[v] {
            bn = bn.min(caps[ei]);
            v = u;
        }
        v = t;
        while let Some((u, ei)) = par[v] {
            caps[ei] -= bn;
            caps[ei ^ 1] += bn;
            v = u;
        }
    }
    let mut reach = vec![false; n];
    let mut stk = vec![s];
    reach[s] = true;
    while let Some(u) = stk.pop() {
        for &(v, ei) in &adj[u] {
            if !reach[v] && caps[ei] > 1e-15 {
                reach[v] = true;
                stk.push(v);
            }
        }
    }
    (0..m).map(|i| reach[i]).collect()
}

fn threshold_detect(flows: &[Flow]) -> Vec<bool> {
    // Simple baseline: volume > 3 sigma or entropy > 6.0
    let vols: Vec<f64> = flows
        .iter()
        .map(|f| (f.bytes_sent + f.bytes_recv) as f64)
        .collect();
    let mean = vols.iter().sum::<f64>() / vols.len() as f64;
    let std = (vols.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vols.len() as f64).sqrt();
    flows
        .iter()
        .enumerate()
        .map(|(i, f)| vols[i] > mean + 3.0 * std || f.payload_entropy > 6.0)
        .collect()
}

struct Metrics {
    precision: f64,
    recall: f64,
    f1: f64,
    fpr: f64,
}

fn evaluate(flows: &[Flow], preds: &[bool]) -> Metrics {
    let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0, 0, 0);
    for (f, &p) in flows.iter().zip(preds) {
        let truth = f.truth != Attack::Normal;
        match (p, truth) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
    }
    let prec = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let rec = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let f1 = if prec + rec > 0.0 {
        2.0 * prec * rec / (prec + rec)
    } else {
        0.0
    };
    let fpr = if tn + fp > 0 {
        fp as f64 / (tn + fp) as f64
    } else {
        0.0
    };
    Metrics {
        precision: prec,
        recall: rec,
        f1,
        fpr,
    }
}

fn per_attack_detection(flows: &[Flow], preds: &[bool]) -> Vec<(&'static str, usize, usize)> {
    let attacks = [
        Attack::PortScan,
        Attack::BruteForce,
        Attack::Exfiltration,
        Attack::C2Beacon,
        Attack::DDoS,
        Attack::LateralMovement,
    ];
    attacks
        .iter()
        .map(|at| {
            let total = flows.iter().filter(|f| f.truth == *at).count();
            let det = flows
                .iter()
                .zip(preds)
                .filter(|(f, &p)| f.truth == *at && p)
                .count();
            (at.label(), det, total)
        })
        .collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn main() {
    println!("=== Cyber Threat Detection via Graph Cut + RuVector ===\n");

    let (alpha, beta, gamma, k_nn) = (0.25, 0.12, 0.5, 3usize);
    let tmp = TempDir::new().expect("tmpdir");
    let opts = RvfOptions {
        dimension: DIM as u16,
        metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let mut store = RvfStore::create(&tmp.path().join("cyber_flows.rvnet"), opts).expect("create");

    println!(
        "  Flows: {} | Dim: {} | alpha={} beta={} gamma={} k={}\n",
        N_FLOWS, DIM, alpha, beta, gamma, k_nn
    );

    let flows = generate_flows(N_FLOWS, 42);
    let n_attack = flows.iter().filter(|f| f.truth != Attack::Normal).count();
    println!(
        "  Generated: {} flows, {} attacks ({:.1}%)",
        N_FLOWS,
        n_attack,
        n_attack as f64 / N_FLOWS as f64 * 100.0
    );

    println!("\n  Attack Distribution:");
    for at in &[
        Attack::PortScan,
        Attack::BruteForce,
        Attack::Exfiltration,
        Attack::C2Beacon,
        Attack::DDoS,
        Attack::LateralMovement,
    ] {
        let c = flows.iter().filter(|f| f.truth == *at).count();
        if c > 0 {
            println!("    {:>12}: {:>3}", at.label(), c);
        }
    }

    // Extract embeddings
    let embs: Vec<Vec<f32>> = flows.iter().map(|f| extract_embedding(f)).collect();

    // Build graph and solve
    let lam: Vec<f64> = flows.iter().map(|f| anomaly_score(f)).collect();
    let edges = build_graph(&flows, &embs, alpha, beta, k_nn);
    println!(
        "\n  Graph: {} edges ({:.1} per flow)",
        edges.len(),
        edges.len() as f64 / N_FLOWS as f64
    );

    let gc_preds = solve_mincut(&lam, &edges, gamma);
    let th_preds = threshold_detect(&flows);

    let gc_m = evaluate(&flows, &gc_preds);
    let th_m = evaluate(&flows, &th_preds);

    println!(
        "\n  {:>12} {:>8} {:>8} {:>8} {:>8}",
        "Method", "Prec", "Recall", "F1", "FPR"
    );
    println!("  {:->12} {:->8} {:->8} {:->8} {:->8}", "", "", "", "", "");
    println!(
        "  {:>12} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
        "Graph Cut", gc_m.precision, gc_m.recall, gc_m.f1, gc_m.fpr
    );
    println!(
        "  {:>12} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
        "Threshold", th_m.precision, th_m.recall, th_m.f1, th_m.fpr
    );

    println!("\n  Per-Attack Detection (Graph Cut vs Threshold):");
    println!("  {:>12} {:>8} {:>10}", "Attack", "GC", "Threshold");
    println!("  {:->12} {:->8} {:->10}", "", "", "");
    let gc_pa = per_attack_detection(&flows, &gc_preds);
    let th_pa = per_attack_detection(&flows, &th_preds);
    for (gc, th) in gc_pa.iter().zip(th_pa.iter()) {
        println!(
            "  {:>12} {:>3}/{:<3} {:>5}/{:<3}",
            gc.0, gc.1, gc.2, th.1, th.2
        );
    }

    // RVF ingestion
    println!("\n--- RVF Ingestion ---");
    let mut vecs = Vec::new();
    let mut ids = Vec::new();
    let mut meta = Vec::new();
    for (i, f) in flows.iter().enumerate() {
        vecs.push(embs[i].clone());
        ids.push(i as u64);
        meta.push(MetadataEntry {
            field_id: FIELD_PROTOCOL,
            value: MetadataValue::String(f.protocol.label().into()),
        });
        meta.push(MetadataEntry {
            field_id: FIELD_PORT_CAT,
            value: MetadataValue::String(port_category(f.dst_port).into()),
        });
        meta.push(MetadataEntry {
            field_id: FIELD_SUBNET,
            value: MetadataValue::U64(f.src_subnet as u64),
        });
        meta.push(MetadataEntry {
            field_id: FIELD_ATTACK,
            value: MetadataValue::String(f.truth.label().into()),
        });
    }
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let ing = store
        .ingest_batch(&refs, &ids, Some(&meta))
        .expect("ingest");
    println!("  Ingested: {} (rejected: {})", ing.accepted, ing.rejected);

    // Filtered queries
    println!("\n--- RVF Queries ---");
    // Find flows similar to a known portscan
    let scan_idx = flows
        .iter()
        .position(|f| f.truth == Attack::PortScan)
        .unwrap_or(0);
    let qv = &embs[scan_idx];
    let res = store.query(qv, 10, &QueryOptions::default()).expect("q");
    println!(
        "  PortScan-similar (flow {}): {} results",
        scan_idx,
        res.len()
    );
    for r in res.iter().take(5) {
        let at = flows[r.id as usize].truth.label();
        println!("    id={:>5} dist={:.4} attack={}", r.id, r.distance, at);
    }

    // Filter by protocol
    let ft = FilterExpr::Eq(FIELD_PROTOCOL, FilterValue::String("TCP".into()));
    let tr = store
        .query(
            qv,
            5,
            &QueryOptions {
                filter: Some(ft),
                ..Default::default()
            },
        )
        .expect("q");
    println!("  TCP-only: {} results", tr.len());

    let fu = FilterExpr::Eq(FIELD_PROTOCOL, FilterValue::String("UDP".into()));
    let ur = store
        .query(
            qv,
            5,
            &QueryOptions {
                filter: Some(fu),
                ..Default::default()
            },
        )
        .expect("q");
    println!("  UDP-only: {} results", ur.len());

    // Witness chain: incident response audit trail
    println!("\n--- Witness Chain (Incident Response) ---");
    let steps = [
        ("capture", 0x08u8),
        ("normalize", 0x02),
        ("embed", 0x02),
        ("graph_build", 0x02),
        ("mincut_solve", 0x02),
        ("classify", 0x02),
        ("alert", 0x01),
        ("forensics", 0x02),
        ("rvf_ingest", 0x08),
        ("query", 0x02),
        ("report", 0x01),
        ("seal", 0x01),
    ];
    let entries: Vec<WitnessEntry> = steps
        .iter()
        .enumerate()
        .map(|(i, (s, wt))| WitnessEntry {
            prev_hash: [0u8; 32],
            action_hash: shake256_256(format!("cyber:{}:{}", s, i).as_bytes()),
            timestamp_ns: 1_700_000_000_000_000_000 + i as u64 * 1_000_000_000,
            witness_type: *wt,
        })
        .collect();
    let chain = create_witness_chain(&entries);
    let verified = verify_witness_chain(&chain).expect("verify");
    println!("  {} entries, {} bytes, VALID", verified.len(), chain.len());
    for (i, (s, _)) in steps.iter().enumerate() {
        let wn = match verified[i].witness_type {
            0x01 => "PROV",
            0x02 => "COMP",
            0x08 => "DATA",
            _ => "????",
        };
        println!("    [{:>4}] {:>2} -> {}", wn, i, s);
    }

    // Lineage
    println!("\n--- Lineage ---");
    let child = store
        .derive(
            &tmp.path().join("alerts.rvnet"),
            DerivationType::Filter,
            None,
        )
        .expect("derive");
    let lineage_depth = child.lineage_depth();
    println!(
        "  parent: {} -> child: {} (depth {})",
        hex(store.file_id()),
        hex(child.parent_id()),
        lineage_depth
    );
    child.close().expect("close");

    // Summary
    println!("\n=== Summary ===");
    println!(
        "  {} flows | {} attacks ({:.1}%) | {} embeddings ingested",
        N_FLOWS,
        n_attack,
        n_attack as f64 / N_FLOWS as f64 * 100.0,
        ing.accepted
    );
    println!(
        "  Graph Cut  F1={:.3} Prec={:.3} Rec={:.3} FPR={:.3}",
        gc_m.f1, gc_m.precision, gc_m.recall, gc_m.fpr
    );
    println!(
        "  Threshold  F1={:.3} Prec={:.3} Rec={:.3} FPR={:.3}",
        th_m.f1, th_m.precision, th_m.recall, th_m.fpr
    );
    println!(
        "  Witness: {} steps | Lineage depth: {}",
        verified.len(),
        lineage_depth
    );

    store.close().expect("close");
    println!("\nDone.");
}
