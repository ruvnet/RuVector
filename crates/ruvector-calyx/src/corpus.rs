//! The `.calyx` corpus format — a dependency-free bridge from real datasets.
//!
//! Real lenses (a code-search model, a text embedder, BM25) are produced
//! **offline, once** by a converter (see `tools/build_codesearchnet.py`) and
//! serialized here. This crate stays pure Rust with no model/network
//! dependency: it *loads precomputed multi-lens vectors + ground truth* and
//! runs the association layer. Same pattern as ann-benchmarks' precomputed HDF5.
//!
//! ## Binary layout (little-endian, version 1)
//!
//! ```text
//! magic     : 8 bytes = b"CALYXB1\n"
//! u32         n_lenses
//!   per lens: u16 name_len, name;  u16 ver_len, ver;
//!             u8 kind; u32 dims; f32 cost_us; f32 latency_us
//! u32         n_records
//!   per rec : u64 id;  (dims×f32 per lens, in header order);
//!             u16 n_anchors; per anchor: u8 kind, f32 weight, u16 ref_len, ref
//! u32         n_queries
//!   per q   : per lens: u8 present, if present dims×f32;
//!             u32 n_relevant; n_relevant × u64 rel_id
//! ```
//!
//! `n_relevant == 0` marks an *unanswerable* query (the correct behaviour is to
//! abstain) — which is what makes conformal risk meaningful on real data.

use std::io::{self, Read, Write};

use crate::constellation::{Constellation, ConstellationStore};
use crate::engine::Query;
use crate::ledger::{Anchor, AnchorKind};
use crate::lens::{LensKind, LensManifest};

const MAGIC: &[u8; 8] = b"CALYXB1\n";

/// A query paired with its relevance ground truth (`relevant` empty = the query
/// is unanswerable and abstaining is correct).
#[derive(Debug, Clone)]
pub struct RealQuery {
    pub query: Query,
    pub relevant: Vec<u64>,
}

impl RealQuery {
    pub fn is_answerable(&self) -> bool {
        !self.relevant.is_empty()
    }
}

/// A loaded corpus: the lens registry + records, plus labelled queries.
pub struct Corpus {
    pub store: ConstellationStore,
    pub queries: Vec<RealQuery>,
}

// ── LensKind / AnchorKind wire codes ────────────────────────────────────────
fn lens_kind_code(k: LensKind) -> u8 {
    match k {
        LensKind::DenseSemantic => 0,
        LensKind::Lexical => 1,
        LensKind::Structural => 2,
        LensKind::Temporal => 3,
        LensKind::Sensor => 4,
    }
}
fn lens_kind_from(c: u8) -> LensKind {
    match c {
        1 => LensKind::Lexical,
        2 => LensKind::Structural,
        3 => LensKind::Temporal,
        4 => LensKind::Sensor,
        _ => LensKind::DenseSemantic,
    }
}
fn anchor_kind_code(k: AnchorKind) -> u8 {
    match k {
        AnchorKind::AcceptedAnswer => 0,
        AnchorKind::PassedTest => 1,
        AnchorKind::SensorLabel => 2,
        AnchorKind::Citation => 3,
        AnchorKind::Reward => 4,
    }
}
fn anchor_kind_from(c: u8) -> AnchorKind {
    match c {
        0 => AnchorKind::AcceptedAnswer,
        2 => AnchorKind::SensorLabel,
        3 => AnchorKind::Citation,
        4 => AnchorKind::Reward,
        _ => AnchorKind::PassedTest,
    }
}

// ── little-endian primitives ────────────────────────────────────────────────
fn wu16<W: Write>(w: &mut W, v: u16) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn wu32<W: Write>(w: &mut W, v: u32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn wu64<W: Write>(w: &mut W, v: u64) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn wf32<W: Write>(w: &mut W, v: f32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn wstr<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    wu16(w, s.len() as u16)?;
    w.write_all(s.as_bytes())
}

fn ru8<R: Read>(r: &mut R) -> io::Result<u8> { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(b[0]) }
fn ru16<R: Read>(r: &mut R) -> io::Result<u16> { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(u16::from_le_bytes(b)) }
fn ru32<R: Read>(r: &mut R) -> io::Result<u32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn ru64<R: Read>(r: &mut R) -> io::Result<u64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
fn rf32<R: Read>(r: &mut R) -> io::Result<f32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(f32::from_le_bytes(b)) }
fn rstr<R: Read>(r: &mut R) -> io::Result<String> {
    let n = ru16(r)? as usize;
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad utf8"))
}
fn rvec<R: Read>(r: &mut R, dims: usize) -> io::Result<Vec<f32>> {
    let mut v = Vec::with_capacity(dims);
    for _ in 0..dims {
        v.push(rf32(r)?);
    }
    Ok(v)
}

/// Serialize a corpus to a writer in `.calyx` v1.
pub fn save<W: Write>(w: &mut W, store: &ConstellationStore, queries: &[RealQuery]) -> io::Result<()> {
    w.write_all(MAGIC)?;
    let lenses = store.lenses();
    wu32(w, lenses.len() as u32)?;
    for m in lenses {
        wstr(w, &m.name)?;
        wstr(w, &m.version)?;
        w.write_all(&[lens_kind_code(m.kind)])?;
        wu32(w, m.dims as u32)?;
        wf32(w, m.cost_us)?;
        wf32(w, m.latency_us)?;
    }
    wu32(w, store.records().len() as u32)?;
    for rec in store.records() {
        wu64(w, rec.id)?;
        for m in lenses {
            let slot = rec.slot(&m.name).unwrap_or(&[]);
            for i in 0..m.dims {
                wf32(w, slot.get(i).copied().unwrap_or(0.0))?;
            }
        }
        wu16(w, rec.anchors.len() as u16)?;
        for a in &rec.anchors {
            w.write_all(&[anchor_kind_code(a.kind)])?;
            wf32(w, a.weight)?;
            wstr(w, &a.reference)?;
        }
    }
    wu32(w, queries.len() as u32)?;
    for q in queries {
        for m in lenses {
            match q.query.slots.get(&m.name) {
                Some(v) => {
                    w.write_all(&[1u8])?;
                    for i in 0..m.dims {
                        wf32(w, v.get(i).copied().unwrap_or(0.0))?;
                    }
                }
                None => w.write_all(&[0u8])?,
            }
        }
        wu32(w, q.relevant.len() as u32)?;
        for &id in &q.relevant {
            wu64(w, id)?;
        }
    }
    Ok(())
}

/// Deserialize a corpus from a reader.
pub fn load<R: Read>(r: &mut R) -> io::Result<Corpus> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic (not a .calyx v1 file)"));
    }
    let n_lenses = ru32(r)? as usize;
    let mut lenses = Vec::with_capacity(n_lenses);
    for _ in 0..n_lenses {
        let name = rstr(r)?;
        let ver = rstr(r)?;
        let kind = lens_kind_from(ru8(r)?);
        let dims = ru32(r)? as usize;
        let cost = rf32(r)?;
        let lat = rf32(r)?;
        lenses.push(LensManifest::new(name, ver, kind, dims, cost, lat));
    }
    let mut store = ConstellationStore::new(lenses.clone());

    let n_records = ru32(r)? as usize;
    for _ in 0..n_records {
        let id = ru64(r)?;
        let mut c = Constellation::new(id);
        for m in &lenses {
            c.slots.insert(m.name.clone(), rvec(r, m.dims)?);
        }
        let n_anchors = ru16(r)? as usize;
        for _ in 0..n_anchors {
            let kind = anchor_kind_from(ru8(r)?);
            let weight = rf32(r)?;
            let reference = rstr(r)?;
            c.anchors.push(Anchor::new(kind, reference, weight));
        }
        store
            .insert(c)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    }

    let n_queries = ru32(r)? as usize;
    let mut queries = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let mut query = Query::new();
        for m in &lenses {
            if ru8(r)? == 1 {
                query.slots.insert(m.name.clone(), rvec(r, m.dims)?);
            }
        }
        let n_rel = ru32(r)? as usize;
        let mut relevant = Vec::with_capacity(n_rel);
        for _ in 0..n_rel {
            relevant.push(ru64(r)?);
        }
        queries.push(RealQuery { query, relevant });
    }
    Ok(Corpus { store, queries })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> (ConstellationStore, Vec<RealQuery>) {
        let lenses = vec![
            LensManifest::new("code", "unixcoder-base", LensKind::DenseSemantic, 3, 8.0, 8.0),
            LensManifest::new("doc", "bge-small", LensKind::DenseSemantic, 3, 3.0, 3.0),
            LensManifest::new("lexical", "bm25", LensKind::Lexical, 3, 1.0, 1.0),
        ];
        let mut store = ConstellationStore::new(lenses);
        store
            .insert(
                Constellation::new(10)
                    .with_slot("code", vec![1.0, 0.0, 0.0])
                    .with_slot("doc", vec![0.9, 0.1, 0.0])
                    .with_slot("lexical", vec![1.0, 0.0, 0.0])
                    .with_anchor(Anchor::new(AnchorKind::PassedTest, "has_test", 0.9)),
            )
            .unwrap();
        store
            .insert(
                Constellation::new(11)
                    .with_slot("code", vec![0.0, 1.0, 0.0])
                    .with_slot("doc", vec![0.0, 0.9, 0.1])
                    .with_slot("lexical", vec![0.0, 1.0, 0.0]),
            )
            .unwrap();
        let queries = vec![
            RealQuery {
                query: Query::new()
                    .with_slot("code", vec![0.95, 0.05, 0.0])
                    .with_slot("doc", vec![0.9, 0.1, 0.0])
                    .with_slot("lexical", vec![1.0, 0.0, 0.0]),
                relevant: vec![10],
            },
            RealQuery {
                // Unanswerable: gold not in pool. Query has only text lens.
                query: Query::new().with_slot("doc", vec![0.0, 0.0, 1.0]),
                relevant: vec![],
            },
        ];
        (store, queries)
    }

    #[test]
    fn round_trips() {
        let (store, queries) = tiny();
        let mut buf = Vec::new();
        save(&mut buf, &store, &queries).unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let c = load(&mut cur).unwrap();

        assert_eq!(c.store.lenses().len(), 3);
        assert_eq!(c.store.len(), 2);
        assert_eq!(c.queries.len(), 2);
        // Record slots survived.
        assert_eq!(c.store.get(10).unwrap().slot("code").unwrap(), &[1.0, 0.0, 0.0]);
        // Anchor survived.
        assert_eq!(c.store.get(10).unwrap().anchors.len(), 1);
        // Query relevance + partial lens presence survived.
        assert_eq!(c.queries[0].relevant, vec![10]);
        assert!(c.queries[1].relevant.is_empty());
        assert!(!c.queries[1].query.slots.contains_key("code"));
        assert!(c.queries[1].query.slots.contains_key("doc"));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut cur = std::io::Cursor::new(b"NOTCALYX".to_vec());
        assert!(load(&mut cur).is_err());
    }
}
