//! Princeton-format FlyWire ingest: gzipped CSV dump → `Connectome`.
//!
//! The Princeton codex.flywire.ai release ships as two gzipped CSVs:
//!
//! - `neurons.csv.gz` — one row per neuron with columns
//!   `Root ID, Top in/out region, Community labels, Predicted NT type,
//!    Predicted NT confidence, Verified NT type, Verified Neuropeptide,
//!    Body Part, Function, Flow, Super Class, Class, Sub Class,
//!    Hemilineage, Nerve, Soma side, Primary Cell Type,
//!    Alternative Cell Type(s), Cable length (nm), Surface area (nm^2),
//!    Volume (nm^3)`.
//!
//! - `connections_princeton.csv.gz` — one row per (pre, post, neuropil)
//!   triple with columns `pre_root_id, post_root_id, neuropil,
//!   syn_count, nt_type`. Multiple rows may share the same (pre, post)
//!   pair — one per neuropil — and the loader aggregates them.
//!
//! This is distinct from the `v783 TSV` path in `loader.rs` /
//! `streaming.rs` which expects tab-delimited, uncompressed files with
//! `neuron_id` / `pre_id` / `post_id` column names and an optional
//! `classification.tsv`. The Princeton dump is what codex actually
//! serves today.
//!
//! Invariants:
//! - Deterministic: byte-identical input → byte-identical Connectome.
//! - No `unsafe`.
//! - Streaming: synapses are bucketed into per-pre `HashMap<(post),
//!   (weight, sign)>` so the ~3.8M Princeton rows collapse to ~1M
//!   unique directed pairs without ever materialising a `Vec<SynapseRow>`.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use flate2::read::GzDecoder;
use serde::Deserialize;

use super::loader::{
    classify_cell_type, default_bias_for, default_delay_ms, nt_to_sign, parse_nt,
};
use super::FlywireError;
use crate::connectome::generator::Connectome;
use crate::connectome::schema::{
    ConnectomeSerCfg, FlyWireNeuronId, NeuronId, NeuronMeta, Synapse,
};

/// One row of `neurons.csv.gz`. Header: `Root ID, Top in/out region,
/// Community labels, Predicted NT type, …`. Only fields we consume are
/// bound; the rest are ignored via `serde::Deserialize`'s column-name
/// matching with `#[serde(rename = ...)]`.
#[derive(Debug, Deserialize)]
struct PrincetonNeuronRow {
    #[serde(rename = "Root ID")]
    root_id: u64,
    #[serde(rename = "Predicted NT type", default)]
    predicted_nt: String,
    #[serde(rename = "Verified NT type", default)]
    verified_nt: String,
    #[serde(rename = "Community labels", default)]
    community_labels: String,
    #[serde(rename = "Flow", default)]
    flow: String,
    #[serde(rename = "Super Class", default)]
    super_class: String,
    #[serde(rename = "Class", default)]
    class_field: String,
    #[serde(rename = "Primary Cell Type", default)]
    primary_cell_type: String,
}

/// One row of `connections_princeton.csv.gz`. Header:
/// `pre_root_id, post_root_id, neuropil, syn_count, nt_type`.
#[derive(Debug, Deserialize)]
struct PrincetonConnectionRow {
    pre_root_id: u64,
    post_root_id: u64,
    #[serde(default)]
    #[allow(dead_code)]
    neuropil: String,
    syn_count: u32,
    #[serde(default)]
    nt_type: String,
}

/// Load a `Connectome` from the Princeton-format gzipped CSV files.
///
/// `neurons_path` → `neurons.csv.gz`
/// `connections_path` → `connections_princeton.csv.gz`
pub fn load_flywire_princeton(
    neurons_path: &Path,
    connections_path: &Path,
) -> Result<Connectome, FlywireError> {
    eprintln!(
        "[princeton] loading neurons from {}",
        neurons_path.display()
    );
    let (neurons, id_of, flywire_ids, meta, nt_per_neuron) = read_neurons(neurons_path)?;
    eprintln!("[princeton] neurons: n={}", neurons);

    eprintln!(
        "[princeton] loading connections from {}",
        connections_path.display()
    );
    let n = neurons;
    let (row_ptr, flat) = read_connections(connections_path, &id_of, &nt_per_neuron, n)?;
    eprintln!(
        "[princeton] connections: {} unique directed pairs",
        flat.len()
    );

    let cfg = ConnectomeSerCfg {
        num_neurons: n as u32,
        num_modules: 1,
        num_hub_modules: 0,
        seed: 0,
    };
    Ok(Connectome::from_parts(
        cfg,
        meta,
        flat,
        row_ptr,
        Some(flywire_ids),
    ))
}

/// Parse the neuron table. Returns (count, id_of, flywire_ids, meta,
/// nt_per_neuron).
#[allow(clippy::type_complexity)]
fn read_neurons(
    path: &Path,
) -> Result<
    (
        usize,
        HashMap<u64, NeuronId>,
        Vec<FlyWireNeuronId>,
        Vec<NeuronMeta>,
        Vec<crate::connectome::flywire::schema::NeuroTransmitter>,
    ),
    FlywireError,
> {
    let mut rdr = open_csv_gz(path)?;
    let mut id_of: HashMap<u64, NeuronId> = HashMap::with_capacity(150_000);
    let mut flywire_ids: Vec<FlyWireNeuronId> = Vec::with_capacity(150_000);
    let mut meta: Vec<NeuronMeta> = Vec::with_capacity(150_000);
    let mut nt_per_neuron = Vec::with_capacity(150_000);

    for (i, result) in rdr.deserialize::<PrincetonNeuronRow>().enumerate() {
        let rec: PrincetonNeuronRow = result.map_err(|e| FlywireError::MalformedRow {
            file: label_of(path),
            line: (i + 2) as u64,
            detail: e.to_string(),
        })?;
        if id_of.contains_key(&rec.root_id) {
            return Err(FlywireError::DuplicateNeuron(rec.root_id));
        }
        let idx = flywire_ids.len();
        id_of.insert(rec.root_id, NeuronId(idx as u32));
        flywire_ids.push(FlyWireNeuronId(rec.root_id));

        // Pick effective NT: Verified wins over Predicted. Empty → default to ACH.
        let nt_raw = if !rec.verified_nt.is_empty() {
            rec.verified_nt.as_str()
        } else if !rec.predicted_nt.is_empty() {
            rec.predicted_nt.as_str()
        } else {
            "ACH"
        };
        let nt = parse_nt(nt_raw, rec.root_id).unwrap_or_else(|_| {
            // Fall back to ACH on unknown rather than failing the whole
            // load — Princeton occasionally ships rows with empty /
            // non-standard NT strings; we log but don't fail.
            crate::connectome::flywire::schema::NeuroTransmitter::Acetylcholine
        });
        nt_per_neuron.push(nt);

        // Cell type / super class → NeuronClass. Princeton's "Community
        // labels" column carries tags like "sensory neuron", "soma in
        // brain" — when present and containing "sensory", fall through
        // to the PhotoReceptor/Chemosensory family so is_sensory() fires.
        let effective_cell_type: Option<String> = if !rec.primary_cell_type.is_empty() {
            Some(rec.primary_cell_type.clone())
        } else if !rec.class_field.is_empty() {
            Some(rec.class_field.clone())
        } else if !rec.super_class.is_empty() {
            Some(rec.super_class.clone())
        } else if rec.community_labels.to_ascii_lowercase().contains("sensory") {
            Some("sensory".to_string())
        } else {
            None
        };
        let flow_opt = if rec.flow.is_empty() {
            None
        } else {
            Some(rec.flow.as_str())
        };
        let class = classify_cell_type(effective_cell_type.as_deref(), flow_opt)
            .unwrap_or(crate::connectome::schema::NeuronClass::Other);
        meta.push(NeuronMeta {
            class,
            module: 0,
            bias_pa: default_bias_for(class),
        });
    }
    let n = flywire_ids.len();
    Ok((n, id_of, flywire_ids, meta, nt_per_neuron))
}

/// Parse the connections table and emit CSR-flat synapses. Aggregates
/// rows sharing the same (pre, post) pair by summing `syn_count`. Skips
/// rows whose `pre_root_id` or `post_root_id` is not in `id_of` (rare —
/// usually dangling pointers to non-neuron segments that the release
/// doesn't enforce). NT is per-row when present, else per-pre default.
fn read_connections(
    path: &Path,
    id_of: &HashMap<u64, NeuronId>,
    nt_per_neuron: &[crate::connectome::flywire::schema::NeuroTransmitter],
    n: usize,
) -> Result<(Vec<u32>, Vec<Synapse>), FlywireError> {
    let mut rdr = open_csv_gz(path)?;
    // per_pre[pre] -> HashMap<post, (count_sum, sign, nt_source)>.
    // Per-pair sum matches the aggregate behaviour of the v783 TSV
    // path (which has one row per pair already).
    type PairAcc = (u32, crate::connectome::schema::Sign);
    let mut per_pre: Vec<HashMap<u32, PairAcc>> = (0..n).map(|_| HashMap::new()).collect();
    let mut dangling_pre: u64 = 0;
    let mut dangling_post: u64 = 0;
    let mut self_loops: u64 = 0;
    let mut parsed: u64 = 0;

    for (i, result) in rdr.deserialize::<PrincetonConnectionRow>().enumerate() {
        let rec: PrincetonConnectionRow = result.map_err(|e| FlywireError::MalformedRow {
            file: label_of(path),
            line: (i + 2) as u64,
            detail: e.to_string(),
        })?;
        parsed += 1;
        let pre = match id_of.get(&rec.pre_root_id) {
            Some(id) => *id,
            None => {
                dangling_pre += 1;
                continue;
            }
        };
        let post = match id_of.get(&rec.post_root_id) {
            Some(id) => *id,
            None => {
                dangling_post += 1;
                continue;
            }
        };
        if pre == post {
            self_loops += 1;
            continue;
        }
        let nt = if !rec.nt_type.is_empty() {
            parse_nt(&rec.nt_type, rec.pre_root_id)
                .unwrap_or(nt_per_neuron[pre.idx()])
        } else {
            nt_per_neuron[pre.idx()]
        };
        let sign = nt_to_sign(nt);
        let count = rec.syn_count.max(1);
        let entry = per_pre[pre.idx()]
            .entry(post.idx() as u32)
            .or_insert((0, sign));
        entry.0 = entry.0.saturating_add(count);
        // Sign stays at the first-observed value — Princeton's
        // per-neuropil nt columns are usually consistent for a given
        // pre, and when they disagree the first one wins
        // deterministically because HashMap is keyed by (pre, post).
    }

    eprintln!(
        "[princeton]   parsed={} self_loops={} dangling_pre={} dangling_post={}",
        parsed, self_loops, dangling_pre, dangling_post
    );

    let mut row_ptr: Vec<u32> = Vec::with_capacity(n + 1);
    let total: usize = per_pre.iter().map(|m| m.len()).sum();
    let mut flat: Vec<Synapse> = Vec::with_capacity(total);
    row_ptr.push(0);
    for bucket in per_pre {
        let mut entries: Vec<(u32, PairAcc)> = bucket.into_iter().collect();
        entries.sort_by_key(|(post, _)| *post);
        for (post, (count, sign)) in entries {
            flat.push(Synapse {
                post: NeuronId(post),
                weight: count as f32,
                delay_ms: default_delay_ms(),
                sign,
            });
        }
        row_ptr.push(flat.len() as u32);
    }
    Ok((row_ptr, flat))
}

fn open_csv_gz(path: &Path) -> Result<csv::Reader<GzDecoder<File>>, FlywireError> {
    let file = File::open(path).map_err(|e| FlywireError::Io {
        file: label_of(path),
        detail: e.to_string(),
    })?;
    let gz = GzDecoder::new(file);
    Ok(csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(true)
        .flexible(true)
        .from_reader(gz))
}

fn label_of(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<?>")
        .to_string()
}
