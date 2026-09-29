//! Tombstone purge and dense-iid compaction, both in place.
//!
//! Two passes, so the expensive part can run in alarm slices (ADR §6.1):
//! 1. [`HnswIndex::repair_links`] (sliceable): in the *old* slot space,
//!    each live node that links to a tombstoned node gets those links
//!    replaced by the live neighbours reachable through it, re-selected by
//!    the usual heuristic. Tombstoned rows are only read, so slices are
//!    independent and inserts/deletes may interleave.
//! 2. [`HnswIndex::compact`]: one linear renumbering pass that drops
//!    never-assigned slots (and, when purging, tombstoned nodes plus any
//!    link still pointing at them). New slots are `≤` old ones and the
//!    pass runs in ascending order, so every arena is rewritten in place:
//!    the only allocation is the returned old→new map (4 B per slot).

use super::{HnswIndex, ABSENT, NONE};
use crate::heap::Cand;

impl HnswIndex {
    /// Repair up to `max_nodes` slots starting at `cursor` (start at 0);
    /// returns the next cursor, which equals [`Self::node_count`] when the
    /// pass is complete. Run it to completion before `compact(true)`.
    pub fn repair_links(&mut self, cursor: u32, max_nodes: u32) -> u32 {
        let end = cursor.saturating_add(max_nodes).min(self.node_count());
        let mut scratch = Vec::new();
        let mut cands: Vec<Cand> = Vec::new();
        for x in cursor..end {
            if !self.is_present(x) || self.is_deleted(x) {
                continue;
            }
            for layer in 0..=self.levels[x as usize] as usize {
                let row = self.links(x, layer);
                if !row.iter().any(|&y| y != NONE && self.is_deleted(y)) {
                    continue;
                }
                let mut ids: Vec<u32> = Vec::new();
                for &y in row.iter().take_while(|&&y| y != NONE) {
                    if !self.is_deleted(y) {
                        ids.push(y);
                        continue;
                    }
                    let via = self.links(y, layer);
                    ids.extend(
                        via.iter()
                            .take_while(|&&z| z != NONE)
                            .filter(|&&z| z != x && !self.is_deleted(z)),
                    );
                }
                ids.sort_unstable();
                ids.dedup();
                let pq = self.quant.prepare_code(self.code(x), &mut scratch);
                cands.clear();
                cands.extend(ids.iter().map(|&id| Cand {
                    d: pq.distance(self.code(id)),
                    id,
                }));
                cands.sort_unstable();
                let cap = row.len();
                let chosen = self.select(&cands, cap, &mut scratch);
                let keep: Vec<u32> = chosen.iter().map(|c| c.id).collect();
                self.set_links(x, layer, &keep);
            }
        }
        end
    }

    /// Renumber in place. Drops never-assigned slots, and with
    /// `purge_deleted` also every tombstoned node (their codes, links and
    /// upper rows are freed for reuse; links to them are removed, and a
    /// deleted entry point is replaced by the highest-level live node).
    ///
    /// Returns the map: `map[old_iid - iid_base]` is the new iid, or
    /// `u32::MAX` for a dropped slot; the store applies it to
    /// `vectors.iid`. Search results are identical up to renumbering when
    /// nothing is purged. Capacity is kept for reuse (see
    /// [`Self::shrink_to_fit`]).
    pub fn compact(&mut self, purge_deleted: bool) -> Vec<u32> {
        let n = self.levels.len();
        let mut map = vec![NONE; n];
        let mut next = 0u32;
        for (i, m) in map.iter_mut().enumerate() {
            let s = i as u32;
            if self.is_present(s) && !(purge_deleted && self.is_deleted(s)) {
                *m = next;
                next += 1;
            }
        }
        let remap = |row: &mut [u32], map: &[u32]| {
            let mut k = 0;
            for j in 0..row.len() {
                let x = row[j];
                if x != NONE && map[x as usize] != NONE {
                    row[k] = map[x as usize];
                    k += 1;
                }
            }
            row[k..].fill(NONE);
        };
        let (d, m, m0) = (
            self.quant.dim(),
            self.params.m as usize,
            self.params.m0 as usize,
        );
        // Upper rows move in offset order (insert order), not slot order.
        let mut owners: Vec<u32> = (0..n as u32)
            .filter(|&s| map[s as usize] != NONE && self.levels[s as usize] > 0)
            .collect();
        owners.sort_unstable_by_key(|&s| self.upper_off[s as usize]);
        let mut row_next = 0usize;
        for &s in &owners {
            let (from, rows) = (
                self.upper_off[s as usize] as usize,
                self.levels[s as usize] as usize,
            );
            self.upper
                .copy_within(from * m..(from + rows) * m, row_next * m);
            for r in 0..rows {
                let at = (row_next + r) * m;
                remap(&mut self.upper[at..at + m], &map);
            }
            self.upper_off[s as usize] = row_next as u32;
            row_next += rows;
        }
        self.upper.truncate(row_next * m);
        let mut deleted = vec![0u64; (next as usize).div_ceil(64)];
        for old in 0..n {
            let new = map[old];
            if new == NONE {
                continue;
            }
            let nw = new as usize;
            if self.is_deleted(old as u32) {
                deleted[nw / 64] |= 1u64 << (nw % 64);
            }
            self.codes.copy_within(old * d..(old + 1) * d, nw * d);
            self.levels[nw] = self.levels[old];
            self.upper_off[nw] = if self.levels[old] > 0 {
                self.upper_off[old]
            } else {
                NONE
            };
            self.links0.copy_within(old * m0..(old + 1) * m0, nw * m0);
            remap(&mut self.links0[nw * m0..(nw + 1) * m0], &map);
        }
        let nn = next as usize;
        self.codes.truncate(nn * d);
        self.levels.truncate(nn);
        self.upper_off.truncate(nn);
        self.links0.truncate(nn * m0);
        self.deleted.clear();
        self.deleted.extend_from_slice(&deleted);
        self.present = next;
        self.live = next - deleted.iter().map(|w| w.count_ones()).sum::<u32>();
        if self.entry != NONE && map[self.entry as usize] != NONE {
            self.entry = map[self.entry as usize];
        } else {
            // Highest level, lowest slot on ties.
            let best = (0..next).rev().max_by_key(|&s| self.levels[s as usize]);
            self.entry = best.unwrap_or(NONE);
            self.top = best.map_or(0, |s| self.levels[s as usize]);
        }
        debug_assert!(self.levels.iter().all(|&l| l != ABSENT));
        let base = self.params.iid_base;
        for v in map.iter_mut().filter(|v| **v != NONE) {
            *v += base;
        }
        map
    }
}
