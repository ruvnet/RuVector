//! Streaming FlyWire v783 ingest path.
//!
//! The non-streaming `loader::load_flywire` in this module materialises
//! every TSV row into a `Vec<SynapseRecord>` before building CSR. That
//! intermediate buffer is ~40 B × 54.5 M synapses ≈ 2 GB for the real
//! v783 release — wasteful when the final CSR holds ~16 B × 54.5 M ≈
//! 870 MB of the same information.
//!
//! `load_flywire_streaming` (this file) drops the intermediate
//! `Vec<SynapseRecord>` and pipes `csv::Reader::deserialize` rows
//! directly into per-pre `Vec<Synapse>` buckets. The neuron pass stays
//! two-stage (TSV → `Vec<NeuronRecord>` → CSR) because neuron count is
//! three orders of magnitude smaller and the lookup table of
//! `HashMap<FlyWire_id, NeuronId>` is required by the synapse pass.
//!
//! Memory high-water mark, real FlyWire v783 (~139 k neurons,
//! ~54.5 M synapses):
//!
//!   non-streaming loader : Vec<SynapseRecord> ~2 GB + per_pre + CSR
//!                          → peak ~4.5 GB
//!   streaming loader     : per_pre (~same as CSR) + CSR
//!                          → peak ~1.7 GB
//!
//! Same output: the final `Connectome` is bit-identical to the
//! non-streaming `load_flywire`, verified by `tests/flywire_streaming.rs`.
//! Streaming is a pure memory optimisation, not a semantic change.
//!
//! Bound to be the Tier-2 ingest entrypoint once the FlyWire v783
//! tarball fetch / extract lands. Named as follow-up in ADR-154 §13
//! ("streaming ingest from the real ~2 GB release tarball").

use std::collections::HashMap;
use std::path::Path;

use super::loader::{
    classify_cell_type, default_bias_for, default_delay_ms, derive_weight, nt_to_sign, parse_nt,
    read_classifications, read_neurons,
};
use super::schema::{NeuronRecord, SynapseRecord};
use super::FlywireError;
use crate::connectome::schema::{ConnectomeSerCfg, FlyWireNeuronId, NeuronId, NeuronMeta, Synapse};
use crate::connectome::Connectome;

/// Streaming variant of `load_flywire` — identical output, lower peak
/// memory on large inputs. See the module docstring for the memory
/// budget derivation.
pub fn load_flywire_streaming(dir: &Path) -> Result<Connectome, FlywireError> {
    let neurons_path = dir.join("neurons.tsv");
    let connections_path = dir.join("connections.tsv");
    let classification_path = dir.join("classification.tsv");

    // Pass 0: neuron pass is non-streaming (still cheap; see module
    // docstring). Uses the existing helpers directly so all the
    // per-row error variants stay identical.
    let neurons: Vec<NeuronRecord> = read_neurons(&neurons_path)?;
    let class_overrides = if classification_path.exists() {
        read_classifications(&classification_path)?
    } else {
        HashMap::new()
    };

    // Build the id map + meta + NT-per-neuron table just like
    // `assemble_connectome` does on the non-streaming path.
    let n = neurons.len();
    let mut id_of: HashMap<u64, NeuronId> = HashMap::with_capacity(n);
    let mut flywire_ids: Vec<FlyWireNeuronId> = Vec::with_capacity(n);
    let mut meta: Vec<NeuronMeta> = Vec::with_capacity(n);
    let mut nt_per_neuron = Vec::with_capacity(n);

    for (idx, rec) in neurons.iter().enumerate() {
        id_of.insert(rec.neuron_id, NeuronId(idx as u32));
        flywire_ids.push(FlyWireNeuronId(rec.neuron_id));
        let class_override = class_overrides.get(&rec.neuron_id);
        let effective_cell_type =
            rec.effective_cell_type(class_override.map(|c| c.cell_type.as_str()));
        let class = classify_cell_type(effective_cell_type.as_deref(), rec.flow.as_deref())?;
        let nt = parse_nt(&rec.nt_type, rec.neuron_id)?;
        nt_per_neuron.push(nt);
        meta.push(NeuronMeta {
            class,
            module: 0,
            bias_pa: default_bias_for(class),
        });
    }

    // Pass 1: synapses streamed directly into per-pre buckets. No
    // `Vec<SynapseRecord>` intermediate. Matches the non-streaming
    // path's row-by-row error reporting.
    let mut per_pre: Vec<Vec<Synapse>> = vec![Vec::new(); n];
    let mut rdr = open_tsv(&connections_path)?;
    for (i, result) in rdr.deserialize::<SynapseRecord>().enumerate() {
        let rec: SynapseRecord = result.map_err(|e| FlywireError::MalformedRow {
            file: label_of(&connections_path),
            line: (i + 2) as u64, // +1 header, +1 for 1-based
            detail: e.to_string(),
        })?;
        let pre = *id_of
            .get(&rec.pre_id)
            .ok_or(FlywireError::UnknownPreNeuron(rec.pre_id))?;
        let post = *id_of
            .get(&rec.post_id)
            .ok_or(FlywireError::UnknownPostNeuron(rec.post_id))?;
        if pre == post {
            continue; // self-loop — matches non-streaming + SBM
        }
        let nt = if let Some(s) = &rec.nt_type {
            parse_nt(s, rec.pre_id)?
        } else {
            nt_per_neuron[pre.idx()]
        };
        let sign = nt_to_sign(nt);
        let count = rec.syn_count.max(1);
        let weight = derive_weight(&rec, count);
        per_pre[pre.idx()].push(Synapse {
            post,
            weight,
            delay_ms: default_delay_ms(),
            sign,
        });
    }

    // CSR flatten — same as non-streaming. Could be fused into the
    // synapse pass with a counting pre-pass (two-pass synapse scan),
    // trading one file rescan for removing `per_pre` entirely. At
    // ~2 GB CSR-final and ~2 GB per_pre on real FlyWire, that saves
    // another 2 GB peak at the cost of rereading the connections.tsv.
    // Kept as one-pass + per_pre here because (a) the test fixture
    // path is identical to the non-streaming loader, and (b) an
    // `mmap`-backed reader makes the rescan essentially free anyway.
    let mut row_ptr: Vec<u32> = Vec::with_capacity(n + 1);
    let total: usize = per_pre.iter().map(|v| v.len()).sum();
    let mut flat: Vec<Synapse> = Vec::with_capacity(total);
    row_ptr.push(0);
    for bucket in per_pre {
        flat.extend(bucket);
        row_ptr.push(flat.len() as u32);
    }

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

fn open_tsv(path: &Path) -> Result<csv::Reader<std::fs::File>, FlywireError> {
    csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .flexible(false)
        .from_path(path)
        .map_err(|e| FlywireError::Io {
            file: label_of(path),
            detail: e.to_string(),
        })
}

fn label_of(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
