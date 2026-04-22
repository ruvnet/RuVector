//! FlyWire v783 TSV → `Connectome` loader.
//!
//! Streaming parse: one pass over `neurons.tsv`, one pass over
//! `classification.tsv` (optional override), one pass over
//! `connections.tsv`. Dense `NeuronId`s are assigned in the order neurons
//! are first seen in the neuron file; parallel arrays of `FlyWireNeuronId`
//! and `NeuronMeta` are preserved alongside the CSR.
//!
//! The loader is deterministic: given a byte-identical TSV input, the
//! output `Connectome` (synapses, row_ptr, meta, flywire_ids) is
//! bit-identical. Synapses within a neuron are stored in the order they
//! appear in `connections.tsv`.
//!
//! Errors are surfaced through the crate-level [`FlywireError`] so
//! callers can distinguish "bad CSV syntax" from "unknown cell type"
//! from "dangling synapse reference".

use std::collections::HashMap;
use std::path::Path;

use super::schema::{CellTypeRecord, NeuroTransmitter, NeuronRecord, SynapseRecord};
use super::FlywireError;
use crate::connectome::generator::Connectome;
use crate::connectome::schema::{
    ConnectomeSerCfg, FlyWireNeuronId, NeuronClass, NeuronId, NeuronMeta, Sign, Synapse,
};

/// Load a FlyWire v783 release from `dir`.
///
/// Expects three TSV files under `dir`: `neurons.tsv`,
/// `connections.tsv`, `classification.tsv`. The classification file is
/// optional; if absent, the cell-type column on `neurons.tsv` is used
/// directly.
///
/// See [`FlywireError`] for the failure modes.
pub fn load_flywire(dir: &Path) -> Result<Connectome, FlywireError> {
    let neurons_path = dir.join("neurons.tsv");
    let connections_path = dir.join("connections.tsv");
    let classification_path = dir.join("classification.tsv");
    let neurons = read_neurons(&neurons_path)?;
    let class_overrides = if classification_path.exists() {
        read_classifications(&classification_path)?
    } else {
        HashMap::new()
    };
    let synapses = read_synapses(&connections_path)?;
    assemble_connectome(neurons, class_overrides, synapses)
}

/// Parse `neurons.tsv` into a vector of [`NeuronRecord`]s. Duplicate
/// `neuron_id` entries yield [`FlywireError::DuplicateNeuron`].
pub fn read_neurons(path: &Path) -> Result<Vec<NeuronRecord>, FlywireError> {
    let mut rdr = open_tsv(path)?;
    let mut out: Vec<NeuronRecord> = Vec::new();
    let mut seen: HashMap<u64, usize> = HashMap::new();
    for (i, result) in rdr.deserialize::<NeuronRecord>().enumerate() {
        let rec: NeuronRecord = result.map_err(|e| FlywireError::MalformedRow {
            file: label_of(path),
            line: (i + 2) as u64, // +1 for header, +1 for 1-based
            detail: e.to_string(),
        })?;
        if seen.insert(rec.neuron_id, i).is_some() {
            return Err(FlywireError::DuplicateNeuron(rec.neuron_id));
        }
        out.push(rec);
    }
    Ok(out)
}

/// Parse `classification.tsv` into a `neuron_id → record` map.
pub fn read_classifications(path: &Path) -> Result<HashMap<u64, CellTypeRecord>, FlywireError> {
    let mut rdr = open_tsv(path)?;
    let mut out: HashMap<u64, CellTypeRecord> = HashMap::new();
    for (i, result) in rdr.deserialize::<CellTypeRecord>().enumerate() {
        let rec: CellTypeRecord = result.map_err(|e| FlywireError::MalformedRow {
            file: label_of(path),
            line: (i + 2) as u64,
            detail: e.to_string(),
        })?;
        out.insert(rec.neuron_id, rec);
    }
    Ok(out)
}

/// Parse `connections.tsv` into a vector of [`SynapseRecord`]s. Order
/// is preserved; the loader relies on file-declared order for CSR
/// determinism.
pub fn read_synapses(path: &Path) -> Result<Vec<SynapseRecord>, FlywireError> {
    let mut rdr = open_tsv(path)?;
    let mut out: Vec<SynapseRecord> = Vec::new();
    for (i, result) in rdr.deserialize::<SynapseRecord>().enumerate() {
        let rec: SynapseRecord = result.map_err(|e| FlywireError::MalformedRow {
            file: label_of(path),
            line: (i + 2) as u64,
            detail: e.to_string(),
        })?;
        out.push(rec);
    }
    Ok(out)
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

fn assemble_connectome(
    neurons: Vec<NeuronRecord>,
    class_overrides: HashMap<u64, CellTypeRecord>,
    synapses: Vec<SynapseRecord>,
) -> Result<Connectome, FlywireError> {
    // Dense id assignment in TSV declaration order.
    let mut id_of: HashMap<u64, NeuronId> = HashMap::with_capacity(neurons.len());
    let mut flywire_ids: Vec<FlyWireNeuronId> = Vec::with_capacity(neurons.len());
    let mut meta: Vec<NeuronMeta> = Vec::with_capacity(neurons.len());
    let mut nt_per_neuron: Vec<NeuroTransmitter> = Vec::with_capacity(neurons.len());

    for (idx, n) in neurons.iter().enumerate() {
        id_of.insert(n.neuron_id, NeuronId(idx as u32));
        flywire_ids.push(FlyWireNeuronId(n.neuron_id));
        let class_override = class_overrides.get(&n.neuron_id);
        let effective_cell_type =
            n.effective_cell_type(class_override.map(|c| c.cell_type.as_str()));
        let class = classify_cell_type(effective_cell_type.as_deref(), n.flow.as_deref())?;
        let nt = parse_nt(&n.nt_type, n.neuron_id)?;
        nt_per_neuron.push(nt);
        meta.push(NeuronMeta {
            class,
            module: 0,
            bias_pa: default_bias_for(class),
        });
    }

    // Partition synapses by pre-id in file-declared order.
    let n = neurons.len();
    let mut per_pre: Vec<Vec<Synapse>> = vec![Vec::new(); n];

    for syn in &synapses {
        let pre = *id_of
            .get(&syn.pre_id)
            .ok_or(FlywireError::UnknownPreNeuron(syn.pre_id))?;
        let post = *id_of
            .get(&syn.post_id)
            .ok_or(FlywireError::UnknownPostNeuron(syn.post_id))?;
        if pre == post {
            continue; // drop self-loops; matches SBM generator
        }
        let nt = if let Some(s) = &syn.nt_type {
            parse_nt(s, syn.pre_id)?
        } else {
            nt_per_neuron[pre.idx()]
        };
        let sign = nt_to_sign(nt);
        let count = syn.syn_count.max(1);
        let weight = derive_weight(syn, count);
        per_pre[pre.idx()].push(Synapse {
            post,
            weight,
            delay_ms: default_delay_ms(),
            sign,
        });
    }

    // CSR flatten (row_ptr + synapses), preserving per-pre order.
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

/// Normalize a raw NT-type string to the typed enum. Case-insensitive
/// match against the seven release-documented labels. Anything else is
/// [`FlywireError::UnknownNtType`] — no silent default.
pub fn parse_nt(raw: &str, context_id: u64) -> Result<NeuroTransmitter, FlywireError> {
    let upper = raw.trim().to_ascii_uppercase();
    match upper.as_str() {
        "ACH" | "ACETYLCHOLINE" => Ok(NeuroTransmitter::Acetylcholine),
        "GLUT" | "GLUTAMATE" => Ok(NeuroTransmitter::Glutamate),
        "GABA" => Ok(NeuroTransmitter::Gaba),
        "HIST" | "HISTAMINE" => Ok(NeuroTransmitter::Histamine),
        "SER" | "SEROTONIN" | "5-HT" | "5HT" => Ok(NeuroTransmitter::Serotonin),
        "DOP" | "DOPAMINE" | "DA" => Ok(NeuroTransmitter::Dopamine),
        "OCT" | "OCTOPAMINE" | "OA" => Ok(NeuroTransmitter::Octopamine),
        _ => Err(FlywireError::UnknownNtType {
            raw: raw.to_owned(),
            neuron_id: context_id,
        }),
    }
}

/// NT → fast-path sign mapping (research doc §4 table).
///
/// - ACH, GLUT                → +1 (Excitatory)
/// - GABA, HIST                → -1 (Inhibitory)
/// - SER, DOP, OCT (modulatory) → +1 in the fast path; analyses that
///   need to exclude slow edges must consult the NT side-channel.
pub fn nt_to_sign(nt: NeuroTransmitter) -> Sign {
    match nt {
        NeuroTransmitter::Acetylcholine | NeuroTransmitter::Glutamate => Sign::Excitatory,
        NeuroTransmitter::Gaba | NeuroTransmitter::Histamine => Sign::Inhibitory,
        NeuroTransmitter::Serotonin | NeuroTransmitter::Dopamine | NeuroTransmitter::Octopamine => {
            Sign::Excitatory
        }
    }
}

/// Map a FlyWire cell-type string to our coarse [`NeuronClass`].
///
/// Unknown cell types fall into `NeuronClass::Other` — this is
/// intentional: the FlyWire release documents ~8,000 cell types, and
/// the coarse bucket is the correct v1 behavior per the research doc.
/// Empty cell-type with a non-empty `flow` column still resolves via
/// the flow hint. If *both* are missing the entry is `Other`, not an
/// error (matches the release's "unresolved" neurons).
pub fn classify_cell_type(
    cell_type: Option<&str>,
    flow: Option<&str>,
) -> Result<NeuronClass, FlywireError> {
    if let Some(ct) = cell_type {
        if let Some(class) = classify_by_prefix(ct) {
            return Ok(class);
        }
    }
    if let Some(f) = flow {
        return Ok(classify_by_flow(f));
    }
    Ok(NeuronClass::Other)
}

/// Strict variant of [`classify_cell_type`]. Unmapped cell types yield
/// [`FlywireError::UnknownCellType`] instead of folding to
/// [`NeuronClass::Other`]. Intended for callers that want to audit
/// prefix-table coverage on a specific release.
pub fn classify_cell_type_strict(
    cell_type: Option<&str>,
    flow: Option<&str>,
    neuron_id: u64,
) -> Result<NeuronClass, FlywireError> {
    if let Some(ct) = cell_type {
        if let Some(class) = classify_by_prefix(ct) {
            return Ok(class);
        }
        return Err(FlywireError::UnknownCellType {
            raw: ct.to_owned(),
            neuron_id,
        });
    }
    if let Some(f) = flow {
        return Ok(classify_by_flow(f));
    }
    Ok(NeuronClass::Other)
}

fn classify_by_prefix(ct: &str) -> Option<NeuronClass> {
    // Order matters: more-specific prefixes first.
    let t = ct.trim();
    if t.starts_with("PR_") || t.starts_with("R1") || t.starts_with("R7") || t.starts_with("R8") {
        return Some(NeuronClass::PhotoReceptor);
    }
    if t.starts_with("ORN") || t.starts_with("PN_glom") || t.starts_with("PN_") {
        return Some(NeuronClass::Chemosensory);
    }
    if t.starts_with("JO") || t.starts_with("ML_mech") {
        return Some(NeuronClass::Mechanosensory);
    }
    if t.starts_with("KC") {
        return Some(NeuronClass::KenyonCell);
    }
    if t.starts_with("MBON") {
        return Some(NeuronClass::MbOutput);
    }
    if t.starts_with("EPG") || t.starts_with("PEN") || t.starts_with("FB_") || t.starts_with("PB_")
    {
        return Some(NeuronClass::CentralComplex);
    }
    if t.starts_with("LAL") {
        return Some(NeuronClass::LateralAccessory);
    }
    if t.starts_with("DNp") || t.starts_with("DNg") || t.starts_with("DN_") {
        return Some(NeuronClass::Descending);
    }
    if t.starts_with("Ascending") || t.starts_with("AN_") {
        return Some(NeuronClass::Ascending);
    }
    if t.starts_with("Motor") {
        return Some(NeuronClass::Motor);
    }
    if t.starts_with("LN_") || t.starts_with("LocalInter") {
        return Some(NeuronClass::LocalInter);
    }
    if t.starts_with("Proj") || t.starts_with("Projection") {
        return Some(NeuronClass::Projection);
    }
    if t.starts_with("DAN") || t.starts_with("SER_") || t.starts_with("OAN") {
        return Some(NeuronClass::Modulatory);
    }
    if t.starts_with("Loc_opt") || t.starts_with("LoOpt") || t.starts_with("Lo_") {
        return Some(NeuronClass::OpticLocal);
    }
    None
}

fn classify_by_flow(flow: &str) -> NeuronClass {
    match flow.trim().to_ascii_lowercase().as_str() {
        "afferent" => NeuronClass::Other,
        "efferent" => NeuronClass::Motor,
        "intrinsic" => NeuronClass::Other,
        "ascending" => NeuronClass::Ascending,
        "descending" => NeuronClass::Descending,
        _ => NeuronClass::Other,
    }
}

fn default_bias_for(class: NeuronClass) -> f32 {
    if class.is_sensory() {
        -0.5
    } else if class.is_motor() {
        0.5
    } else {
        0.0
    }
}

fn derive_weight(syn: &SynapseRecord, count: u32) -> f32 {
    if syn.syn_weight > 0.0 {
        syn.syn_weight
    } else {
        count as f32
    }
}

fn default_delay_ms() -> f32 {
    // Research doc §3.2: FlyWire does not publish conduction delays;
    // the ingest loader uses a constant fallback of 2.0 ms. The
    // distance-scaled estimator requires soma coordinates, which are
    // optional in the release and absent from the fixture.
    2.0
}
