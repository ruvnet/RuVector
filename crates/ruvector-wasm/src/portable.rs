//! Portable HNSW: no mmap, filesystem, threads, native RNG or native HNSW dependency.
//! Updates/deletes rebuild the graph; filtered queries use exact search for completeness.
use ruvector_core::{
    distance::distance,
    error::{Result, RuvectorError},
    types::{DbOptions, DistanceMetric, SearchQuery, SearchResult, VectorEntry},
};
use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap, HashSet},
};

const M: usize = 16;
const EF_CONSTRUCTION: usize = 128;
const EF_SEARCH: usize = 64;
const MAX_LEVEL: usize = 16;

#[derive(Clone)]
struct Node {
    entry: VectorEntry,
    links: Vec<Vec<usize>>,
}
#[derive(Copy, Clone)]
struct Candidate {
    index: usize,
    distance: f32,
}
impl PartialEq for Candidate {
    fn eq(&self, b: &Self) -> bool {
        self.cmp(b) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl Ord for Candidate {
    fn cmp(&self, b: &Self) -> Ordering {
        self.distance
            .total_cmp(&b.distance)
            .then(self.index.cmp(&b.index))
    }
}

pub(crate) struct PortableDB {
    dimensions: usize,
    metric: DistanceMetric,
    hnsw: bool,
    nodes: Vec<Node>,
    ids: HashMap<String, usize>,
    entry: Option<usize>,
    rng: u64,
    next_id: u64,
}
impl PortableDB {
    pub fn new(options: DbOptions) -> Result<Self> {
        if options.dimensions == 0 || options.dimensions > 65536 {
            return Err(RuvectorError::InvalidDimension("expected 1..65536".into()));
        }
        Ok(Self {
            dimensions: options.dimensions,
            metric: options.distance_metric,
            hnsw: options.hnsw_config.is_some(),
            nodes: vec![],
            ids: HashMap::new(),
            entry: None,
            rng: 0x9e3779b97f4a7c15,
            next_id: 0,
        })
    }
    pub fn index_type(&self) -> &'static str {
        if self.hnsw {
            "hnsw"
        } else {
            "flat"
        }
    }
    fn validate(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: vector.len(),
            });
        }
        if vector.iter().any(|x| !x.is_finite()) {
            return Err(RuvectorError::InvalidInput(
                "vector values must be finite".into(),
            ));
        }
        // Keep every metric's f32 accumulation finite, including squared L2
        // differences. Extreme finite floats otherwise produce NaN cosine scores.
        let safe_magnitude = f32::MAX.sqrt() / (4.0 * (self.dimensions as f32).sqrt());
        if vector.iter().any(|x| x.abs() > safe_magnitude) {
            return Err(RuvectorError::InvalidInput(
                "vector magnitude exceeds safe distance range".into(),
            ));
        }
        Ok(())
    }
    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        // Core distances preserve the published lower-is-better contract.
        // Dimension equality is checked at insertion/query boundaries.
        distance(a, b, self.metric).expect("validated dimensions")
    }
    fn candidate(&self, query: &[f32], index: usize) -> Candidate {
        Candidate {
            index,
            distance: self.dist(query, &self.nodes[index].entry.vector),
        }
    }
    fn level(&mut self) -> usize {
        // Xorshift64 with nonzero seed; geometric P(level >= l) = M^-l.
        let mut level = 0;
        while level < MAX_LEVEL {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            if self.rng % M as u64 != 0 {
                break;
            }
            level += 1;
        }
        level
    }
    fn greedy(&self, query: &[f32], start: usize, layer: usize) -> usize {
        let mut best = self.candidate(query, start);
        loop {
            let before = best.index;
            for &n in &self.nodes[before].links[layer] {
                let c = self.candidate(query, n);
                if c < best {
                    best = c;
                }
            }
            if best.index == before {
                return before;
            }
        }
    }
    fn layer_search(&self, query: &[f32], start: usize, layer: usize, ef: usize) -> Vec<Candidate> {
        let first = self.candidate(query, start);
        let mut pending = BinaryHeap::from([Reverse(first)]);
        let mut best = BinaryHeap::from([first]);
        let mut seen = HashSet::from([start]);
        while let Some(Reverse(current)) = pending.pop() {
            if best.len() >= ef && current > *best.peek().unwrap() {
                break;
            }
            for &n in &self.nodes[current.index].links[layer] {
                if !seen.insert(n) {
                    continue;
                }
                let c = self.candidate(query, n);
                if best.len() < ef || c < *best.peek().unwrap() {
                    pending.push(Reverse(c));
                    best.push(c);
                    if best.len() > ef {
                        best.pop();
                    }
                }
            }
        }
        best.into_sorted_vec()
    }
    fn select(&self, candidates: &[Candidate], limit: usize) -> Vec<usize> {
        // HNSW diversity heuristic retains links across clusters, not just a local clique.
        let mut selected = Vec::new();
        let mut rejected = Vec::new();
        for c in candidates {
            if selected.iter().all(|&n: &usize| {
                self.dist(
                    &self.nodes[c.index].entry.vector,
                    &self.nodes[n].entry.vector,
                ) >= c.distance
            }) {
                selected.push(c.index);
                if selected.len() == limit {
                    return selected;
                }
            } else {
                rejected.push(c.index);
            }
        }
        selected.extend(rejected.into_iter().take(limit - selected.len()));
        selected
    }
    fn add(&mut self, entry: VectorEntry) {
        let level = if self.hnsw { self.level() } else { 0 };
        let index = self.nodes.len();
        self.ids.insert(entry.id.clone().unwrap(), index);
        self.nodes.push(Node {
            entry,
            links: vec![vec![]; level + 1],
        });
        let Some(mut ep) = self.entry else {
            self.entry = Some(index);
            return;
        };
        if !self.hnsw {
            return;
        }
        let top = self.nodes[ep].links.len() - 1;
        let query = self.nodes[index].entry.vector.clone();
        for layer in ((level + 1)..=top).rev() {
            ep = self.greedy(&query, ep, layer);
        }
        for layer in (0..=level.min(top)).rev() {
            let candidates = self.layer_search(&query, ep, layer, EF_CONSTRUCTION);
            let limit = if layer == 0 { M * 2 } else { M };
            let neighbors = self.select(&candidates, M);
            self.nodes[index].links[layer] = neighbors.clone();
            for n in neighbors {
                self.nodes[n].links[layer].push(index);
                if self.nodes[n].links[layer].len() > limit {
                    let mut links: Vec<_> = self.nodes[n].links[layer]
                        .iter()
                        .map(|&i| self.candidate(&self.nodes[n].entry.vector, i))
                        .collect();
                    links.sort_unstable();
                    self.nodes[n].links[layer] = self.select(&links, limit);
                }
            }
            if let Some(c) = candidates.first() {
                ep = c.index;
            }
        }
        if level > top {
            self.entry = Some(index);
        }
    }
    fn rebuild(&mut self, entries: Vec<VectorEntry>) {
        self.nodes.clear();
        self.ids.clear();
        self.entry = None;
        self.rng = 0x9e3779b97f4a7c15;
        for entry in entries {
            self.add(entry);
        }
    }
    pub fn insert(&mut self, mut entry: VectorEntry) -> Result<String> {
        self.validate(&entry.vector)?;
        let id = match entry.id.clone() {
            Some(id) => id,
            None => loop {
                let id = format!("wasm_{}", self.next_id);
                self.next_id = self.next_id.wrapping_add(1);
                if !self.ids.contains_key(&id) {
                    break id;
                }
            },
        };
        entry.id = Some(id.clone());
        if let Some(&index) = self.ids.get(&id) {
            let mut entries: Vec<_> = self.nodes.iter().map(|n| n.entry.clone()).collect();
            entries[index] = entry;
            self.rebuild(entries);
        } else {
            self.add(entry);
        }
        Ok(id)
    }
    pub fn insert_batch(&mut self, entries: Vec<VectorEntry>) -> Result<Vec<String>> {
        for entry in &entries {
            self.validate(&entry.vector)?;
        }
        entries.into_iter().map(|e| self.insert(e)).collect()
    }
    pub fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        self.validate(&query.vector)?;
        let k = query.k.min(self.nodes.len());
        if k == 0 {
            return Ok(vec![]);
        }
        let has_filter = query.filter.as_ref().is_some_and(|f| !f.is_empty());
        let mut candidates = if !self.hnsw || has_filter || k == self.nodes.len() {
            self.nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| {
                    query.filter.as_ref().map_or(true, |f| {
                        f.iter().all(|(key, value)| {
                            n.entry.metadata.as_ref().and_then(|m| m.get(key)) == Some(value)
                        })
                    })
                })
                .map(|(i, _)| self.candidate(&query.vector, i))
                .collect::<Vec<_>>()
        } else {
            let mut ep = self.entry.unwrap();
            for layer in (1..self.nodes[ep].links.len()).rev() {
                ep = self.greedy(&query.vector, ep, layer);
            }
            self.layer_search(
                &query.vector,
                ep,
                0,
                query
                    .ef_search
                    .unwrap_or(EF_SEARCH)
                    .max(k)
                    .min(self.nodes.len()),
            )
        };
        candidates.sort_unstable();
        candidates.truncate(k);
        Ok(candidates
            .into_iter()
            .map(|c| {
                let e = &self.nodes[c.index].entry;
                SearchResult {
                    id: e.id.clone().unwrap(),
                    score: c.distance,
                    vector: Some(e.vector.clone()),
                    metadata: e.metadata.clone(),
                }
            })
            .collect())
    }
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        let Some(&index) = self.ids.get(id) else {
            return Ok(false);
        };
        let entries = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != index)
            .map(|(_, n)| n.entry.clone())
            .collect();
        self.rebuild(entries);
        Ok(true)
    }
    pub fn get(&self, id: &str) -> Result<Option<VectorEntry>> {
        Ok(self.ids.get(id).map(|&i| self.nodes[i].entry.clone()))
    }
    pub fn len(&self) -> Result<usize> {
        Ok(self.nodes.len())
    }
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.nodes.is_empty())
    }
}
