//! Insertion and in-place update: level draw from caller randomness, beam
//! search over codes, heuristic neighbour selection (Malkov & Yashunin,
//! alg. 4) and pruned back-links. Deterministic for a given op sequence
//! and random stream, so a replay rebuilds a bit-identical graph.

use super::{HnswIndex, Visited, ABSENT, NONE};
use crate::error::IndexError;
use crate::grow;
use crate::heap::Cand;
use crate::metric::validate_vector;
use crate::quant::PreparedQuery;
use crate::rng::{level_from_u64, LevelRng};

impl HnswIndex {
    /// Insert `v` at a new `iid`, drawing its level from `rng` (ADR §6.1:
    /// `SplitMix64::mix(seq ^ salt)` per op keeps replays bit-identical).
    ///
    /// `iid` may skip ahead of the last slot (deletes leave holes in the
    /// store's iid space); skipped slots stay absent, cost `dim` bytes
    /// each, and make [`Self::to_chunks`] fail with `GappedIid` until the
    /// store compacts ([`Self::compact`]). Errors: invalid vector, iid
    /// outside `iid_base..iid_base + max_slots`, or an occupied slot (use
    /// [`Self::update`] / [`Self::upsert`] for an existing iid).
    pub fn insert(
        &mut self,
        iid: u32,
        v: &[f32],
        rng: &mut dyn LevelRng,
    ) -> Result<(), IndexError> {
        validate_vector(self.quant.metric(), self.quant.dim(), v)?;
        let s = self.slot(iid)?;
        if self.is_present(s) {
            return Err(IndexError::Occupied(iid));
        }
        self.ensure_slots(s as usize + 1);
        let level = level_from_u64(
            rng.next_u64(),
            self.params.m as usize,
            self.params.max_level,
        );
        let d = self.quant.dim();
        self.quant
            .encode(v, &mut self.codes[s as usize * d..(s as usize + 1) * d]);
        self.levels[s as usize] = level;
        if level > 0 {
            let m = self.params.m as usize;
            let need = self.upper.len() + level as usize * m;
            let target = self.upper_target(need);
            grow::fit(&mut self.upper, need, target);
            self.upper_off[s as usize] = (self.upper.len() / m) as u32;
            self.upper.resize(need, NONE);
        }
        self.present += 1;
        self.live += 1;
        if self.entry == NONE {
            self.entry = s;
            self.top = level;
            return Ok(());
        }
        let pq = self.quant.prepare(v);
        self.connect(s, &pq);
        if level > self.top {
            self.entry = s;
            self.top = level;
        }
        Ok(())
    }

    /// Move the existing node at `iid` to `v` in place: its code is
    /// re-encoded, its level kept (no randomness is drawn, so replay stays
    /// deterministic) and its out-links re-selected on every layer, with
    /// back-links from the new neighbours. A tombstoned node is revived.
    /// Old in-links stay as valid (if less useful) edges until a purge.
    /// The store keeps an id's iid across upserts, so this is the M2b
    /// upsert path. Errors: invalid vector or no node at `iid`.
    pub fn update(&mut self, iid: u32, v: &[f32]) -> Result<(), IndexError> {
        validate_vector(self.quant.metric(), self.quant.dim(), v)?;
        let s = self.slot(iid)?;
        if !self.is_present(s) {
            return Err(IndexError::NotFound(iid));
        }
        let d = self.quant.dim();
        self.quant
            .encode(v, &mut self.codes[s as usize * d..(s as usize + 1) * d]);
        if self.is_deleted(s) {
            self.deleted[(s / 64) as usize] &= !(1u64 << (s % 64));
            self.live += 1;
        }
        let pq = self.quant.prepare(v);
        self.connect(s, &pq);
        Ok(())
    }

    /// [`Self::update`] if `iid` holds a node, else [`Self::insert`]
    /// (which alone consumes `rng`).
    pub fn upsert(
        &mut self,
        iid: u32,
        v: &[f32],
        rng: &mut dyn LevelRng,
    ) -> Result<(), IndexError> {
        match self.slot(iid) {
            Ok(s) if self.is_present(s) => self.update(iid, v),
            _ => self.insert(iid, v, rng),
        }
    }

    /// (Re)select `s`'s neighbours on every layer it has, from a search
    /// seeded at the entry point; `s` itself is never its own candidate.
    fn connect(&mut self, s: u32, pq: &PreparedQuery) {
        let level = self.levels[s as usize];
        let mut ep = Cand {
            d: pq.distance(self.code(self.entry)),
            id: self.entry,
        };
        for l in (level as usize + 1..=self.top as usize).rev() {
            ep = self.greedy(pq, ep, l);
        }
        let mut visited = Visited::new(self.levels.len());
        let mut scratch = Vec::new();
        let ef = self.params.ef_construction as usize;
        for l in (0..=level.min(self.top) as usize).rev() {
            visited.clear();
            let mut cands = self.search_layer(pq, &[ep], ef, l, &mut visited, false);
            cands.retain(|c| c.id != s);
            let chosen = self.select(&cands, self.params.m as usize, &mut scratch);
            let ids: Vec<u32> = chosen.iter().map(|c| c.id).collect();
            self.set_links(s, l, &ids);
            for &nb in &ids {
                self.link_back(nb, s, l, &mut scratch);
            }
            if let Some(best) = cands.first() {
                ep = *best;
            }
        }
    }

    pub(crate) fn ensure_slots(&mut self, n: usize) {
        if n <= self.levels.len() {
            return;
        }
        let t = self.slot_target(n);
        self.fit_slots(n, t);
        self.codes.resize(n * self.quant.dim(), 0);
        self.levels.resize(n, ABSENT);
        self.links0.resize(n * self.params.m0 as usize, NONE);
        self.upper_off.resize(n, NONE);
        self.deleted.resize(n.div_ceil(64), 0);
    }

    /// Heuristic selection from `cands` (ascending by distance to the base):
    /// keep a candidate only if it is closer to the base than to every
    /// neighbour already kept.
    pub(crate) fn select(&self, cands: &[Cand], cap: usize, scratch: &mut Vec<f32>) -> Vec<Cand> {
        let mut out: Vec<Cand> = Vec::with_capacity(cap);
        for c in cands {
            if out.len() >= cap {
                break;
            }
            if out.is_empty() {
                out.push(*c);
                continue;
            }
            let pc = self.quant.prepare_code(self.code(c.id), scratch);
            if out.iter().all(|s| pc.distance(self.code(s.id)) > c.d) {
                out.push(*c);
            }
        }
        out
    }

    /// Add `new` to `nb`'s list at `layer` (no-op if already there),
    /// re-selecting when full.
    fn link_back(&mut self, nb: u32, new: u32, layer: usize, scratch: &mut Vec<f32>) {
        let links = self.links(nb, layer);
        let len = links.iter().position(|&x| x == NONE).unwrap_or(links.len());
        if links[..len].contains(&new) {
            return;
        }
        if len < links.len() {
            let mut ids: Vec<u32> = links[..len].to_vec();
            ids.push(new);
            self.set_links(nb, layer, &ids);
            return;
        }
        let cap = links.len();
        let pn = self.quant.prepare_code(self.code(nb), scratch);
        let mut cands: Vec<Cand> = links
            .iter()
            .chain(std::iter::once(&new))
            .map(|&id| Cand {
                d: pn.distance(self.code(id)),
                id,
            })
            .collect();
        cands.sort_unstable();
        let chosen = self.select(&cands, cap, scratch);
        let ids: Vec<u32> = chosen.iter().map(|c| c.id).collect();
        self.set_links(nb, layer, &ids);
    }
}
