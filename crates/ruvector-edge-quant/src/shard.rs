//! `QuantShard`: one tenant shard's RaBitQ codes (ADR-351 §3 rv-quant).
//!
//! Storage is struct-of-arrays in insertion order, exactly the layout of
//! `ruvector_rabitq::RabitqIndex` (row `i` of `packed` is
//! `packed[i * n_words ..][..n_words]`, bit `63 - (j % 64)` of word `j / 64`
//! is `sign(rotated_unit[j]) >= 0`), built with that crate's rotation,
//! normalisation and popcount scan kernel. The shard owns the arrays itself
//! because the persist v2 loader must install decoded codes directly (no
//! f32 rebuild), and because the edge needs `u64` row keys and deletes.
//!
//! Keys are the store's row keys (the DO-SQLite rowid behind the `String`
//! vector id); the f32 originals stay in the store and are fetched only for
//! rerank candidates (see [`crate::query`]).

use crate::budget::{self, Budget};
use crate::error::{BudgetResource, QuantError, Result};
use ruvector_edge_store::{distance, Metric};
use ruvector_rabitq::rotation::normalize_inplace;
use ruvector_rabitq::{RandomRotation, RandomRotationKind};
use std::collections::BTreeMap;

/// ADR-351 §7: dimension `1..=1536`.
pub const MAX_DIM: usize = 1536;

/// Static configuration of a shard (persisted in the v2 header).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantConfig {
    /// Vector dimension.
    pub dim: usize,
    /// Collection metric (codes are metric-independent; scoring is not).
    pub metric: Metric,
    /// Rotation construction.
    pub rotation: RandomRotationKind,
    /// Rotation seed (the rotation is regenerated from it on load).
    pub seed: u64,
    /// Hard limits.
    pub budget: Budget,
}

impl QuantConfig {
    /// Randomised-Hadamard rotation with default budgets.
    ///
    /// Hadamard is the edge default: `O(D log D)` apply, `3 · next_pow2(D)`
    /// f32 of storage and a near-free rebuild on cold load. At 50k × 384 it
    /// measured the same recall as Haar (1.000 vs 0.9985 at 500 rerank
    /// candidates) with 10× faster encode and 4.5× faster cold load. The
    /// dense Haar rotation (`ruvector-rabitq`'s default, `rbpx0001`
    /// parity) stays opt-in: its `D³` rebuild exceeds the default load
    /// budget above 512 dims and its `4·D²` matrix alone is 9.4 MB at 1536.
    pub fn new(dim: usize, metric: Metric, seed: u64) -> Self {
        QuantConfig {
            dim,
            metric,
            rotation: RandomRotationKind::HadamardSigned,
            seed,
            budget: Budget::default(),
        }
    }
}

/// Result of an upsert batch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UpsertStats {
    /// Rows added under new keys.
    pub inserted: u32,
    /// Rows re-encoded under existing keys.
    pub replaced: u32,
}

/// One tenant shard of RaBitQ codes.
pub struct QuantShard {
    pub(crate) cfg: QuantConfig,
    pub(crate) n_words: usize,
    pub(crate) last_word_mask: u64,
    pub(crate) rotation: RandomRotation,
    pub(crate) cos_lut: Vec<f32>,
    pub(crate) keys: Vec<u64>,
    pub(crate) norms: Vec<f32>,
    pub(crate) packed: Vec<u64>,
    pub(crate) index: BTreeMap<u64, u32>,
    /// Rows charged for `index` since it was last (re)built: inserts add,
    /// removals do not subtract (a B-tree keeps its nodes after removals
    /// until it is compacted, which resets this to the live row count).
    pub(crate) index_rows: u64,
}

/// Mask of the valid (high) bits in a code's last word.
pub(crate) fn last_word_mask(dim: usize) -> u64 {
    let valid = dim - 64 * (budget::n_words(dim) - 1);
    if valid == 64 {
        !0u64
    } else {
        !0u64 << (64 - valid)
    }
}

/// `cos(π · (1 − B/D))` for `B ∈ 0..=D` (same table as `RabitqIndex`).
fn cos_lut(dim: usize) -> Vec<f32> {
    let d = dim as f32;
    (0..=dim)
        .map(|b| (std::f32::consts::PI * (1.0 - b as f32 / d)).cos())
        .collect()
}

/// Deterministically regenerate a rotation.
pub(crate) fn build_rotation(dim: usize, kind: RandomRotationKind, seed: u64) -> RandomRotation {
    match kind {
        RandomRotationKind::HaarDense => RandomRotation::random(dim, seed),
        RandomRotationKind::HadamardSigned => RandomRotation::hadamard(dim, seed),
    }
}

/// Validate a config (dimension range).
pub(crate) fn validate_config(cfg: &QuantConfig) -> Result<()> {
    if cfg.dim == 0 || cfg.dim > MAX_DIM {
        return Err(QuantError::InvalidConfig("dim must be 1..=1536"));
    }
    Ok(())
}

/// Encode one vector into `slot` (zeroed, `n_words` long); returns its norm.
/// `unit` and `rotated` are caller scratch of length `dim`.
///
/// In the normal range this is `ruvector-rabitq`'s f32 path, so codes and
/// norms stay bit-identical to `RabitqIndex`. When the f32 sum of squares
/// overflows (`|v| ≳ 1.8e19`) or underflows, the norm and the unit vector
/// are computed in f64 instead, so a large row can never become an
/// all-ones code with an infinite norm. Callers reject rows whose f64 norm
/// is not a finite f32 ([`QuantShard::upsert`]), so the returned norm is
/// finite for every stored row.
pub(crate) fn encode_into(
    rotation: &RandomRotation,
    v: &[f32],
    unit: &mut Vec<f32>,
    rotated: &mut [f32],
    slot: &mut [u64],
) -> f32 {
    let n32: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    unit.clear();
    let norm = if n32.is_finite() && n32 > 1e-10 {
        unit.extend_from_slice(v);
        normalize_inplace(unit);
        n32
    } else {
        let n64 = distance::norm(v);
        if n64 > 0.0 {
            unit.extend(v.iter().map(|&x| (f64::from(x) / n64) as f32));
        } else {
            unit.extend_from_slice(v);
        }
        n64 as f32
    };
    rotation.apply_into(unit, rotated);
    for (i, &x) in rotated.iter().enumerate() {
        if x >= 0.0 {
            slot[i / 64] |= 1u64 << (63 - (i % 64));
        }
    }
    norm
}

impl QuantShard {
    /// An empty shard. Regenerates the rotation, charged against the load
    /// budget (it is the same work a cold load does).
    pub fn new(cfg: QuantConfig) -> Result<Self> {
        validate_config(&cfg)?;
        let units = budget::rotation_build_units(cfg.dim, cfg.rotation);
        budget::check(BudgetResource::LoadUnits, cfg.budget.max_load_units, units)?;
        let bytes = budget::resident_bytes(0, cfg.dim, cfg.rotation);
        budget::check(
            BudgetResource::ResidentBytes,
            cfg.budget.max_resident_bytes,
            bytes,
        )?;
        let rotation = build_rotation(cfg.dim, cfg.rotation, cfg.seed);
        Ok(Self::from_parts(
            cfg,
            rotation,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        ))
    }

    /// Assemble a shard from validated arrays (used by the v2 loader).
    pub(crate) fn from_parts(
        cfg: QuantConfig,
        rotation: RandomRotation,
        keys: Vec<u64>,
        norms: Vec<f32>,
        packed: Vec<u64>,
        index: BTreeMap<u64, u32>,
    ) -> Self {
        QuantShard {
            n_words: budget::n_words(cfg.dim),
            last_word_mask: last_word_mask(cfg.dim),
            cos_lut: cos_lut(cfg.dim),
            cfg,
            rotation,
            keys,
            norms,
            packed,
            index_rows: index.len() as u64,
            index,
        }
    }

    /// Configuration.
    pub fn config(&self) -> &QuantConfig {
        &self.cfg
    }
    /// Replace the budget (limits are policy, not persisted state).
    pub fn set_budget(&mut self, budget: Budget) {
        self.cfg.budget = budget;
    }
    /// Rows.
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    /// `true` with no rows.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    /// Dimension.
    pub fn dim(&self) -> usize {
        self.cfg.dim
    }
    /// Row keys in storage order.
    pub fn keys(&self) -> &[u64] {
        &self.keys
    }
    /// Original L2 norms in storage order.
    pub fn norms(&self) -> &[f32] {
        &self.norms
    }
    /// Packed codes, row-major.
    pub fn packed(&self) -> &[u64] {
        &self.packed
    }
    /// 64-bit words per code.
    pub fn n_words(&self) -> usize {
        self.n_words
    }
    /// The rotation.
    pub fn rotation(&self) -> &RandomRotation {
        &self.rotation
    }
    /// `true` if `key` is present.
    pub fn contains(&self, key: u64) -> bool {
        self.index.contains_key(&key)
    }
    /// Resident bytes (the figure charged against the budget and reported
    /// to the isolate `ResidentRegistry`): the *capacity* of the code, norm
    /// and key arrays (what the heap actually holds, not their length),
    /// the key index per row, and the rotation + cos-LUT.
    pub fn resident_bytes(&self) -> u64 {
        budget::fixed_bytes(self.cfg.dim, self.cfg.rotation)
            .saturating_add(self.vec_capacity_bytes())
            .saturating_add(self.index_rows * budget::KEY_INDEX_BYTES_PER_ROW)
    }

    /// Rows the arrays hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.keys
            .capacity()
            .min(self.norms.capacity())
            .min(self.packed.capacity() / self.n_words)
    }

    fn vec_capacity_bytes(&self) -> u64 {
        (self.keys.capacity() as u64 * 8)
            .saturating_add(self.norms.capacity() as u64 * 4)
            .saturating_add(self.packed.capacity() as u64 * 8)
    }

    /// Row capacity to hold `n_after` rows, or `413 resident_bytes`.
    ///
    /// Grows by ×1.5 (at least [`budget::MIN_GROW_ROWS`]) for amortised
    /// O(1) inserts, clamped so that the *peak* fits the budget: the new
    /// arrays plus the old `packed` buffer, which is live while `Vec`
    /// reallocation copies it.
    fn plan_capacity(&self, n_after: u64, index_after: u64) -> Result<u64> {
        let (dim, kind) = (self.cfg.dim, self.cfg.rotation);
        let limit = self.cfg.budget.max_resident_bytes;
        let index = index_after.saturating_mul(budget::KEY_INDEX_BYTES_PER_ROW);
        let fixed = budget::fixed_bytes(dim, kind).saturating_add(index);
        let cur = self.capacity() as u64;
        if n_after <= cur {
            let steady = fixed.saturating_add(self.vec_capacity_bytes());
            budget::check(BudgetResource::ResidentBytes, limit, steady)?;
            return Ok(cur);
        }
        let per_row = budget::vec_row_bytes(dim);
        let base = fixed.saturating_add(self.packed.capacity() as u64 * 8);
        let peak = base.saturating_add(n_after.saturating_mul(per_row));
        budget::check(BudgetResource::ResidentBytes, limit, peak)?;
        let fits = (limit - base) / per_row; // ≥ n_after after the check
        let want = n_after.max(cur + cur / 2).max(budget::MIN_GROW_ROWS);
        Ok(want.min(fits))
    }

    fn validate_vector(&self, v: &[f32]) -> Result<()> {
        if v.len() != self.cfg.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.cfg.dim,
                actual: v.len(),
            });
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(QuantError::NonFinite);
        }
        // The stored norm is an f32 and a cold load rejects a non-finite
        // one, so the row's f64 norm must be representable: refusing it
        // here (400) keeps one write from making every snapshot unloadable.
        let n = distance::norm(v);
        if !(n as f32).is_finite() {
            return Err(QuantError::NonFinite);
        }
        if self.cfg.metric == Metric::Cosine && n == 0.0 {
            return Err(QuantError::NonFinite);
        }
        Ok(())
    }

    /// Insert or replace rows. All-or-nothing: every row is validated and
    /// the batch's growth is checked against the vector, resident-byte and
    /// work budgets before any code is written. Within a batch the last
    /// occurrence of a key wins.
    pub fn upsert(&mut self, rows: &[(u64, &[f32])]) -> Result<UpsertStats> {
        for (_, v) in rows {
            self.validate_vector(v)?;
        }
        let mut new_keys: Vec<u64> = rows
            .iter()
            .map(|(k, _)| *k)
            .filter(|k| !self.index.contains_key(k))
            .collect();
        new_keys.sort_unstable();
        new_keys.dedup();
        let added = new_keys.len() as u64;
        drop(new_keys);
        let n_after = self.keys.len() as u64 + added;
        let b = self.cfg.budget;
        budget::check(BudgetResource::Vectors, b.max_vectors, n_after)?;
        let target = self.plan_capacity(n_after, self.index_rows + added)? as usize;
        let units = budget::encode_units(rows.len() as u64, self.cfg.dim, self.cfg.rotation);
        budget::check(BudgetResource::LoadUnits, b.max_load_units, units)?;

        let (nw, dim) = (self.n_words, self.cfg.dim);
        let mut unit = Vec::with_capacity(dim);
        let mut rotated = vec![0.0f32; dim];
        let mut code = vec![0u64; nw];
        let mut stats = UpsertStats::default();
        // Exact reservations: the capacity is what the budget charged.
        let len = self.keys.len();
        self.keys.reserve_exact(target - len);
        self.norms.reserve_exact(target - len);
        self.packed.reserve_exact((target - len) * nw);
        for (key, v) in rows {
            code.iter_mut().for_each(|w| *w = 0);
            let norm = encode_into(&self.rotation, v, &mut unit, &mut rotated, &mut code);
            match self.index.get(key) {
                Some(&pos) => {
                    let p = pos as usize;
                    self.packed[p * nw..(p + 1) * nw].copy_from_slice(&code);
                    self.norms[p] = norm;
                    stats.replaced += 1;
                }
                None => {
                    self.index.insert(*key, self.keys.len() as u32);
                    self.index_rows += 1;
                    self.keys.push(*key);
                    self.norms.push(norm);
                    self.packed.extend_from_slice(&code);
                    stats.inserted += 1;
                }
            }
        }
        Ok(stats)
    }

    /// Remove rows by key (absent keys are ignored). Returns rows removed.
    /// The last row moves into each hole (O(1) per delete).
    pub fn delete(&mut self, keys: &[u64]) -> usize {
        let nw = self.n_words;
        let mut removed = 0;
        for key in keys {
            let Some(pos) = self.index.remove(key) else {
                continue;
            };
            let p = pos as usize;
            let last = self.keys.len() - 1;
            if p != last {
                let moved = self.keys[last];
                self.keys[p] = moved;
                self.norms[p] = self.norms[last];
                self.packed.copy_within(last * nw..(last + 1) * nw, p * nw);
                self.index.insert(moved, pos);
            }
            self.keys.truncate(last);
            self.norms.truncate(last);
            self.packed.truncate(last * nw);
            removed += 1;
        }
        // Compact once more than half of what is charged is slack, so
        // resident bytes (capacity + index charge) track the live rows.
        let len = self.keys.len();
        let slack = len.max(budget::MIN_GROW_ROWS as usize);
        if removed > 0 && (self.capacity() - len > slack || self.index_rows as usize - len > slack)
        {
            self.keys.shrink_to(len);
            self.norms.shrink_to(len);
            self.packed.shrink_to(len * nw);
            // Sorted bulk build: densely packed nodes.
            self.index = self
                .keys
                .iter()
                .enumerate()
                .map(|(i, &k)| (k, i as u32))
                .collect();
            self.index_rows = len as u64;
        }
        removed
    }
}
