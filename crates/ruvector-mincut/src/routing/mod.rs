//! Exact directed routing, independently of the undirected mincut graph.
//!
//! Costs are integer units chosen by the caller (for example milliseconds).
//! Arc IDs are input positions; parallel arcs and zero costs are supported.
//! Forbidden turns are pairs of consecutive arc IDs. Access rules must be
//! resolved by the importer before construction. This is not an OSM parser.
//! Landmark lower bounds ignore turns and remain admissible under closures and
//! cost increases. Decreases/reopening discard them until explicitly rebuilt.
//! No route cache survives an update. Coordinates only affect map lookup.

pub mod rufield;
mod spatial;
pub use spatial::{MapIndex, Snap};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const INF: u64 = u64::MAX;
/// Hard limits also keep every returned integer route cost exactly representable in JS.
pub const MAX_NODES: usize = 1_000_000;
pub const MAX_ARCS: usize = 4_000_000;
pub const MAX_COST: u32 = 1_000_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingError {
    InvalidInput(&'static str),
    BudgetExceeded,
    Cancelled,
}
impl std::fmt::Display for RoutingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for RoutingError {}

#[derive(Clone, Copy, Debug)]
pub struct Arc {
    pub source: u32,
    pub target: u32,
    pub cost: u32,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    pub cost: u64,
    pub nodes: Vec<u32>,
    pub arcs: Vec<u32>,
    pub settled: usize,
}

struct Csr {
    offsets: Vec<usize>,
    arcs: Vec<usize>,
}
impl Csr {
    fn new(n: usize, arcs: &[Arc], reverse: bool) -> Self {
        let node = |a: &Arc| if reverse { a.target } else { a.source } as usize;
        let mut offsets = vec![0; n + 1];
        for a in arcs {
            offsets[node(a) + 1] += 1;
        }
        for i in 1..=n {
            offsets[i] += offsets[i - 1];
        }
        let mut cursor = offsets.clone();
        let mut ids = vec![0; arcs.len()];
        for (id, a) in arcs.iter().enumerate() {
            ids[cursor[node(a)]] = id;
            cursor[node(a)] += 1;
        }
        Self { offsets, arcs: ids }
    }
    fn adjacent(&self, v: usize) -> &[usize] {
        &self.arcs[self.offsets[v]..self.offsets[v + 1]]
    }
}

// One heap entry per state bounds queue memory even with many turn transitions.
#[derive(Default)]
struct QueryHeap {
    entries: Vec<(u64, u64, usize)>,
    positions: Vec<usize>,
}
impl QueryHeap {
    fn new(states: usize) -> Self {
        Self {
            entries: Vec::new(),
            positions: vec![usize::MAX; states],
        }
    }
    fn clear(&mut self) {
        for &(_, _, s) in &self.entries {
            self.positions[s] = usize::MAX;
        }
        self.entries.clear();
    }
    fn swap(&mut self, a: usize, b: usize) {
        self.entries.swap(a, b);
        self.positions[self.entries[a].2] = a;
        self.positions[self.entries[b].2] = b;
    }
    fn push(&mut self, item: (u64, u64, usize)) {
        let mut i = self.positions[item.2];
        if i == usize::MAX {
            i = self.entries.len();
            self.entries.push(item);
            self.positions[item.2] = i;
        } else {
            self.entries[i] = item;
        }
        while i > 0 {
            let p = (i - 1) / 2;
            if self.entries[p] <= self.entries[i] {
                break;
            }
            self.swap(i, p);
            i = p;
        }
    }
    fn pop(&mut self) -> Option<(u64, u64, usize)> {
        if self.entries.is_empty() {
            return None;
        }
        let item = self.entries[0];
        let last = self.entries.pop().unwrap();
        self.positions[item.2] = usize::MAX;
        if !self.entries.is_empty() {
            self.entries[0] = last;
            self.positions[last.2] = 0;
            let mut i = 0;
            loop {
                let left = 2 * i + 1;
                if left >= self.entries.len() {
                    break;
                }
                let right = left + 1;
                let child =
                    if right < self.entries.len() && self.entries[right] < self.entries[left] {
                        right
                    } else {
                        left
                    };
                if self.entries[i] <= self.entries[child] {
                    break;
                }
                self.swap(i, child);
                i = child;
            }
        }
        Some(item)
    }
}

/// Mutable query scratch is owned by one router, avoiding O(n) clears per query.
/// Use one router per worker; synchronous JS calls should run off the UI thread.
pub struct RoadRouter {
    n: usize,
    arcs: Vec<Arc>,
    open: Vec<bool>,
    forward: Csr,
    reverse: Csr,
    turns: Vec<(u32, u32)>,
    landmarks: Vec<(Vec<u64>, Vec<u64>)>,
    dist: Vec<u64>,
    parent: Vec<usize>,
    touched: Vec<usize>,
    heap: QueryHeap,
    map: Option<MapIndex>,
}
impl RoadRouter {
    pub fn new(
        n: usize,
        arcs: Vec<Arc>,
        mut forbidden: Vec<(u32, u32)>,
    ) -> Result<Self, RoutingError> {
        if n == 0 || n > MAX_NODES || arcs.len() > MAX_ARCS || forbidden.len() > MAX_ARCS {
            return Err(RoutingError::InvalidInput("graph size limit"));
        }
        if arcs
            .iter()
            .any(|a| a.source as usize >= n || a.target as usize >= n || a.cost > MAX_COST)
        {
            return Err(RoutingError::InvalidInput("arc endpoint or cost"));
        }
        for &(a, b) in &forbidden {
            if a as usize >= arcs.len()
                || b as usize >= arcs.len()
                || arcs[a as usize].target != arcs[b as usize].source
            {
                return Err(RoutingError::InvalidInput(
                    "turn must join consecutive arcs",
                ));
            }
        }
        forbidden.sort_unstable();
        forbidden.dedup();
        let states = if forbidden.is_empty() {
            n
        } else {
            arcs.len() + 1
        };
        let forward = Csr::new(n, &arcs, false);
        let reverse = Csr::new(n, &arcs, true);
        Ok(Self {
            n,
            open: vec![true; arcs.len()],
            arcs,
            forward,
            reverse,
            turns: forbidden,
            landmarks: Vec::new(),
            dist: vec![INF; states],
            parent: vec![usize::MAX; states],
            touched: Vec::new(),
            heap: QueryHeap::new(states),
            map: None,
        })
    }

    /// Validate typed arrays before allocation. Costs are integers, never NaN/fractions.
    pub fn from_arrays(
        n: usize,
        endpoints: &[u32],
        costs: &[u32],
        turns: &[u32],
    ) -> Result<Self, RoutingError> {
        if n == 0
            || n > MAX_NODES
            || costs.len() > MAX_ARCS
            || endpoints.len() != costs.len() * 2
            || turns.len() % 2 != 0
            || turns.len() / 2 > MAX_ARCS
        {
            return Err(RoutingError::InvalidInput("array lengths or graph size"));
        }
        let arcs = endpoints
            .chunks_exact(2)
            .zip(costs)
            .map(|(e, &cost)| Arc {
                source: e[0],
                target: e[1],
                cost,
            })
            .collect();
        Self::new(
            n,
            arcs,
            turns.chunks_exact(2).map(|t| (t[0], t[1])).collect(),
        )
    }
    pub fn node_count(&self) -> usize {
        self.n
    }
    pub fn arc_count(&self) -> usize {
        self.arcs.len()
    }
    pub fn landmark_count(&self) -> usize {
        self.landmarks.len()
    }
    pub fn set_coordinates(&mut self, lat_lon: &[f64]) -> Result<(), RoutingError> {
        if lat_lon.len() != self.n * 2 {
            return Err(RoutingError::InvalidInput("coordinate count"));
        }
        let map = MapIndex::new(lat_lon)?;
        self.map = Some(map);
        Ok(())
    }
    pub fn nearest(&self, lat: f64, lon: f64, radius_m: f64) -> Result<Option<Snap>, RoutingError> {
        self.map
            .as_ref()
            .ok_or(RoutingError::InvalidInput("coordinates not set"))?
            .nearest(lat, lon, radius_m)
    }
    /// Atomic batch update. None closes an arc. Duplicate IDs are rejected.
    pub fn update(&mut self, updates: &[(u32, Option<u32>)]) -> Result<(), RoutingError> {
        if updates.len() > self.arcs.len() {
            return Err(RoutingError::InvalidInput("update count"));
        }
        let mut ids: Vec<_> = updates.iter().map(|x| x.0).collect();
        ids.sort_unstable();
        if ids.windows(2).any(|w| w[0] == w[1])
            || updates
                .iter()
                .any(|&(id, c)| id as usize >= self.arcs.len() || c.is_some_and(|v| v > MAX_COST))
        {
            return Err(RoutingError::InvalidInput("update ID or cost"));
        }
        for &(id, cost) in updates {
            let id = id as usize;
            if let Some(cost) = cost {
                if !self.open[id] || cost < self.arcs[id].cost {
                    self.landmarks.clear();
                }
                self.arcs[id].cost = cost;
                self.open[id] = true;
            } else {
                self.open[id] = false;
            }
        }
        Ok(())
    }

    /// Explicit landmarks, with a cap on work/memory. Builds atomically; cancellation
    /// or exhaustion preserves the old index. Budget counts arc scans and heap pops.
    pub fn prepare(
        &mut self,
        ids: &[u32],
        budget: usize,
        mut cancelled: impl FnMut() -> bool,
    ) -> Result<(), RoutingError> {
        if ids.len() > 16
            || self.n * ids.len() > 8_000_000
            || ids.iter().any(|&v| v as usize >= self.n)
        {
            return Err(RoutingError::InvalidInput("landmarks"));
        }
        let mut result = Vec::with_capacity(ids.len());
        let mut work = 0;
        for &v in ids {
            let f = self.distances(v as usize, false, budget, &mut work, &mut cancelled)?;
            let r = self.distances(v as usize, true, budget, &mut work, &mut cancelled)?;
            result.push((f, r));
        }
        self.landmarks = result;
        Ok(())
    }
    fn distances(
        &self,
        source: usize,
        reverse: bool,
        budget: usize,
        work: &mut usize,
        cancelled: &mut impl FnMut() -> bool,
    ) -> Result<Vec<u64>, RoutingError> {
        let mut d = vec![INF; self.n];
        d[source] = 0;
        let mut heap = BinaryHeap::from([Reverse((0, source))]);
        let csr = if reverse {
            &self.reverse
        } else {
            &self.forward
        };
        while let Some(Reverse((cost, v))) = heap.pop() {
            tick(work, budget, cancelled)?;
            if cost != d[v] {
                continue;
            }
            for &id in csr.adjacent(v) {
                tick(work, budget, cancelled)?;
                if !self.open[id] {
                    continue;
                }
                let a = self.arcs[id];
                let u = if reverse { a.source } else { a.target } as usize;
                let next = cost + a.cost as u64;
                if next < d[u] {
                    d[u] = next;
                    heap.push(Reverse((next, u)));
                }
            }
        }
        Ok(d)
    }
    fn estimate(&self, v: usize, t: usize) -> u64 {
        let mut h = 0;
        for (f, r) in &self.landmarks {
            if f[t] != INF && f[v] != INF {
                h = h.max(f[t].saturating_sub(f[v]));
            }
            if r[v] != INF && r[t] != INF {
                h = h.max(r[v].saturating_sub(r[t]));
            }
        }
        h
    }
    /// Exact A* (or Dijkstra when use_landmarks=false), with edge-state search for
    /// turn restrictions. Reopening states handles inconsistent directed bounds.
    pub fn route(
        &mut self,
        source: u32,
        target: u32,
        use_landmarks: bool,
        budget: usize,
        mut cancelled: impl FnMut() -> bool,
    ) -> Result<Option<Route>, RoutingError> {
        if source as usize >= self.n || target as usize >= self.n {
            return Err(RoutingError::InvalidInput("query endpoint"));
        }
        for v in self.touched.drain(..) {
            self.dist[v] = INF;
            self.parent[v] = usize::MAX;
        }
        self.heap.clear();
        if cancelled() {
            return Err(RoutingError::Cancelled);
        }
        let edge_states = !self.turns.is_empty();
        let start = if edge_states {
            self.arcs.len()
        } else {
            source as usize
        };
        self.dist[start] = 0;
        self.touched.push(start);
        self.heap.push((0, 0, start));
        let mut work = 0;
        let mut settled = 0;
        while let Some((_, cost, state)) = self.heap.pop() {
            tick(&mut work, budget, &mut cancelled)?;
            if cost != self.dist[state] {
                continue;
            }
            settled += 1;
            let v = if state == start {
                source as usize
            } else if edge_states {
                self.arcs[state].target as usize
            } else {
                state
            };
            if v == target as usize {
                let mut arcs = Vec::new();
                let mut s = state;
                while s != start {
                    let id = if edge_states { s } else { self.parent[s] };
                    arcs.push(id as u32);
                    s = if edge_states {
                        self.parent[s]
                    } else {
                        self.arcs[id].source as usize
                    };
                }
                arcs.reverse();
                let mut nodes = Vec::with_capacity(arcs.len() + 1);
                nodes.push(source);
                nodes.extend(arcs.iter().map(|&a| self.arcs[a as usize].target));
                return Ok(Some(Route {
                    cost,
                    nodes,
                    arcs,
                    settled,
                }));
            }
            for offset in self.forward.offsets[v]..self.forward.offsets[v + 1] {
                tick(&mut work, budget, &mut cancelled)?;
                let id = self.forward.arcs[offset];
                if !self.open[id]
                    || (edge_states
                        && state != start
                        && self.turns.binary_search(&(state as u32, id as u32)).is_ok())
                {
                    continue;
                }
                let a = self.arcs[id];
                let next = cost + a.cost as u64;
                let s = if edge_states { id } else { a.target as usize };
                if next < self.dist[s] {
                    if self.dist[s] == INF {
                        self.touched.push(s);
                    }
                    self.dist[s] = next;
                    self.parent[s] = if edge_states { state } else { id };
                    let h = if use_landmarks {
                        self.estimate(a.target as usize, target as usize)
                    } else {
                        0
                    };
                    self.heap.push((next + h, next, s));
                }
            }
        }
        Ok(None)
    }
}
fn tick(
    work: &mut usize,
    budget: usize,
    cancelled: &mut impl FnMut() -> bool,
) -> Result<(), RoutingError> {
    if *work >= budget {
        return Err(RoutingError::BudgetExceeded);
    }
    if *work % 1024 == 0 && cancelled() {
        return Err(RoutingError::Cancelled);
    }
    *work += 1;
    Ok(())
}
