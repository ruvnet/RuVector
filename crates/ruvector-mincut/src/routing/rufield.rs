//! RuField spatial-awareness adapter for exact routing.
//!
//! The adapter consumes the network-safe subset of a RuField `FieldEvent`.
//! Signature verification stays at the trust boundary: callers must pass
//! `verified=true` for non-synthetic events only after verifying the detached
//! RuField receipt. This module never treats a signature-shaped string as proof.
//! Zones and cells must be explicitly mapped to routing nodes. RuView's current
//! CSI `space_cell` is an uncalibrated field peak, so it is not projected into
//! geographic coordinates here.

use super::{RoadRouter, Route, RoutingError, MAX_COST};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, VecDeque};

const MAX_EVENT_BYTES: usize = 256 * 1024;
const MAX_TRACKED_EVENTS: usize = 4096;
const MAX_ACTIVE_SOURCES: usize = 65_536;
const MAX_BINDINGS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub enum PrivacyClass {
    P0,
    P1,
    P2,
    P3,
    P4,
    P5,
}

#[derive(Debug, Deserialize)]
struct Provenance {
    #[serde(default)]
    synthetic: bool,
}
#[derive(Debug, Deserialize)]
struct Sensor {
    device_id: String,
}
#[derive(Debug, Deserialize)]
struct Observation {
    zone_id: Option<String>,
    space_cell: Option<[i32; 3]>,
    confidence: f32,
    #[serde(default)]
    features: BTreeMap<String, f32>,
    privacy_class: PrivacyClass,
}
#[derive(Debug, Deserialize)]
struct FieldEvent {
    event_id: String,
    timestamp_ns: u64,
    #[serde(default)]
    sensor: Option<Sensor>,
    observation: Observation,
    provenance: Provenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AwarenessPolicy {
    /// Extra route-cost units at confidence=1 and risk=1.
    pub max_penalty: u32,
    /// Close arcs adjacent to a node at or above this normalized risk.
    /// Set above one to disable closures.
    pub close_at_millionths: u32,
    pub ttl_ns: u64,
    pub max_lateness_ns: u64,
}
impl Default for AwarenessPolicy {
    fn default() -> Self {
        Self {
            max_penalty: 60_000,
            close_at_millionths: 950_000,
            ttl_ns: 2_000_000_000,
            max_lateness_ns: 500_000_000,
        }
    }
}
impl AwarenessPolicy {
    fn validate(self) -> Result<Self, RoutingError> {
        if self.max_penalty > MAX_COST
            || self.close_at_millionths > 1_000_001
            || self.ttl_ns == 0
            || self.ttl_ns > 3_600_000_000_000
            || self.max_lateness_ns > self.ttl_ns
        {
            return Err(RoutingError::InvalidInput("RuField policy"));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AwarenessUpdate {
    pub node: u32,
    pub risk_millionths: u32,
    pub changed_arcs: usize,
    pub duplicate: bool,
}

#[derive(Clone, Copy)]
struct ActiveRisk {
    timestamp_ns: u64,
    expires_ns: u64,
    risk: u32,
}

/// Binds privacy-safe, verified RuField observations to routing costs.
///
/// Invariants:
/// * duplicate event IDs are idempotent;
/// * per-source timestamps never move backward beyond the configured allowance;
/// * updates are derived from immutable base costs, so penalties never compound;
/// * expiry restores the exact original arc cost/open state;
/// * landmark bounds remain safe because `RoadRouter::update` invalidates them
///   after a decrease or reopening.
pub struct RuFieldRouter {
    router: RoadRouter,
    policy: AwarenessPolicy,
    base_costs: Vec<u32>,
    base_open: Vec<bool>,
    incident: Vec<Vec<usize>>,
    zones: HashMap<String, u32>,
    cells: HashMap<[i32; 3], u32>,
    active: HashMap<u32, HashMap<String, ActiveRisk>>,
    node_risk: HashMap<u32, u32>,
    active_sources: usize,
    seen: HashMap<String, (u64, u32)>,
    seen_order: VecDeque<(String, u64)>,
}

impl RuFieldRouter {
    pub fn new(router: RoadRouter, policy: AwarenessPolicy) -> Result<Self, RoutingError> {
        let policy = policy.validate()?;
        let base_costs = router.arcs.iter().map(|a| a.cost).collect();
        let base_open = router.open.clone();
        let mut incident = vec![Vec::new(); router.n];
        for (id, arc) in router.arcs.iter().enumerate() {
            incident[arc.source as usize].push(id);
            if arc.target != arc.source {
                incident[arc.target as usize].push(id);
            }
        }
        Ok(Self {
            router,
            policy,
            base_costs,
            base_open,
            incident,
            zones: HashMap::new(),
            cells: HashMap::new(),
            active: HashMap::new(),
            node_risk: HashMap::new(),
            active_sources: 0,
            seen: HashMap::new(),
            seen_order: VecDeque::new(),
        })
    }
    pub fn router(&self) -> &RoadRouter {
        &self.router
    }
    pub fn route(
        &mut self,
        source: u32,
        target: u32,
        use_landmarks: bool,
        budget: usize,
        cancelled: impl FnMut() -> bool,
    ) -> Result<Option<Route>, RoutingError> {
        self.router
            .route(source, target, use_landmarks, budget, cancelled)
    }
    pub fn bind_zone(&mut self, zone: String, node: u32) -> Result<(), RoutingError> {
        if zone.is_empty()
            || zone.len() > 256
            || node as usize >= self.router.n
            || self.zones.len() >= MAX_BINDINGS
        {
            return Err(RoutingError::InvalidInput("RuField zone binding"));
        }
        self.zones.insert(zone, node);
        Ok(())
    }
    pub fn bind_cell(&mut self, cell: [i32; 3], node: u32) -> Result<(), RoutingError> {
        if node as usize >= self.router.n || self.cells.len() >= MAX_BINDINGS {
            return Err(RoutingError::InvalidInput("RuField cell binding"));
        }
        self.cells.insert(cell, node);
        Ok(())
    }
    pub fn active_nodes(&self) -> usize {
        self.node_risk.len()
    }

    pub fn ingest_json(
        &mut self,
        json: &[u8],
        verified: bool,
        now_ns: u64,
    ) -> Result<AwarenessUpdate, RoutingError> {
        if json.is_empty() || json.len() > MAX_EVENT_BYTES {
            return Err(RoutingError::InvalidInput("RuField event size"));
        }
        let event: FieldEvent = serde_json::from_slice(json)
            .map_err(|_| RoutingError::InvalidInput("RuField event JSON"))?;
        if event.event_id.is_empty() || event.event_id.len() > 256 {
            return Err(RoutingError::InvalidInput("RuField event ID"));
        }
        if let Some(&(timestamp, node)) = self.seen.get(&event.event_id) {
            if timestamp != event.timestamp_ns {
                return Err(RoutingError::InvalidInput("RuField event ID collision"));
            }
            return Ok(AwarenessUpdate {
                node,
                risk_millionths: 0,
                changed_arcs: 0,
                duplicate: true,
            });
        }
        if !event.provenance.synthetic && !verified {
            return Err(RoutingError::InvalidInput("unverified RuField event"));
        }
        if event.observation.privacy_class > PrivacyClass::P2 {
            return Err(RoutingError::InvalidInput("RuField privacy class"));
        }
        if event.timestamp_ns > now_ns.saturating_add(self.policy.max_lateness_ns)
            || now_ns.saturating_sub(event.timestamp_ns) > self.policy.ttl_ns
        {
            return Err(RoutingError::InvalidInput("RuField event time"));
        }
        let node = event
            .observation
            .zone_id
            .as_ref()
            .and_then(|z| self.zones.get(z))
            .copied()
            .or_else(|| {
                event
                    .observation
                    .space_cell
                    .and_then(|c| self.cells.get(&c).copied())
            })
            .ok_or(RoutingError::InvalidInput("unmapped RuField observation"))?;
        let risk = risk_millionths(&event.observation)?;
        let source = source_id(&event)?;
        let is_new_source = !self
            .active
            .get(&node)
            .is_some_and(|sources| sources.contains_key(&source));
        if is_new_source && self.active_sources >= MAX_ACTIVE_SOURCES {
            return Err(RoutingError::InvalidInput(
                "too many active RuField sources",
            ));
        }
        if let Some(previous) = self
            .active
            .get(&node)
            .and_then(|sources| sources.get(&source))
        {
            if event
                .timestamp_ns
                .saturating_add(self.policy.max_lateness_ns)
                < previous.timestamp_ns
            {
                return Err(RoutingError::InvalidInput("stale RuField event"));
            }
        }
        self.active.entry(node).or_default().insert(
            source,
            ActiveRisk {
                timestamp_ns: event.timestamp_ns,
                expires_ns: event.timestamp_ns.saturating_add(self.policy.ttl_ns),
                risk,
            },
        );
        self.active_sources += usize::from(is_new_source);
        self.remember(event.event_id, event.timestamp_ns, node);
        let changed_arcs = self.recompute_node(node)?;
        Ok(AwarenessUpdate {
            node,
            risk_millionths: risk,
            changed_arcs,
            duplicate: false,
        })
    }

    pub fn expire(&mut self, now_ns: u64) -> Result<usize, RoutingError> {
        let expired: Vec<(u32, String)> = self
            .active
            .iter()
            .flat_map(|(&node, sources)| {
                sources.iter().filter_map(move |(source, risk)| {
                    (risk.expires_ns <= now_ns).then_some((node, source.clone()))
                })
            })
            .collect();
        let mut affected = Vec::new();
        for (node, source) in expired {
            affected.push(node);
            if let Some(sources) = self.active.get_mut(&node) {
                if sources.remove(&source).is_some() {
                    self.active_sources -= 1;
                }
                if sources.is_empty() {
                    self.active.remove(&node);
                }
            }
        }
        affected.sort_unstable();
        affected.dedup();
        let mut changed = 0;
        for node in affected {
            changed += self.recompute_node(node)?;
        }
        while self
            .seen_order
            .front()
            .is_some_and(|x| now_ns.saturating_sub(x.1) > self.policy.ttl_ns)
        {
            let (id, timestamp) = self.seen_order.pop_front().unwrap();
            if self.seen.get(&id).is_some_and(|x| x.0 == timestamp) {
                self.seen.remove(&id);
            }
        }
        Ok(changed)
    }

    fn remember(&mut self, id: String, timestamp: u64, node: u32) {
        self.seen.insert(id.clone(), (timestamp, node));
        self.seen_order.push_back((id, timestamp));
        while self.seen.len() > MAX_TRACKED_EVENTS {
            if let Some((id, ts)) = self.seen_order.pop_front() {
                if self.seen.get(&id).is_some_and(|x| x.0 == ts) {
                    self.seen.remove(&id);
                }
            }
        }
    }
    fn recompute_node(&mut self, node: u32) -> Result<usize, RoutingError> {
        let risk = self
            .active
            .get(&node)
            .into_iter()
            .flat_map(|sources| sources.values().map(|active| active.risk))
            .max()
            .unwrap_or(0);
        let previous = self.node_risk.get(&node).copied().unwrap_or(0);
        if risk == 0 {
            self.node_risk.remove(&node);
        } else {
            self.node_risk.insert(node, risk);
        }
        if risk == previous {
            return Ok(0);
        }
        self.refresh_node(node)
    }
    fn refresh_node(&mut self, node: u32) -> Result<usize, RoutingError> {
        let ids = self.incident[node as usize].clone();
        let mut updates = Vec::with_capacity(ids.len());
        for id in ids {
            let a = self.router.arcs[id];
            let risk = self
                .node_risk
                .get(&a.source)
                .copied()
                .unwrap_or(0)
                .max(self.node_risk.get(&a.target).copied().unwrap_or(0));
            let value = if !self.base_open[id] || risk >= self.policy.close_at_millionths {
                None
            } else {
                let penalty = (self.policy.max_penalty as u64 * risk as u64 / 1_000_000) as u32;
                Some(self.base_costs[id].saturating_add(penalty).min(MAX_COST))
            };
            if self.router.open[id] != value.is_some() || value.is_some_and(|v| v != a.cost) {
                updates.push((id as u32, value));
            }
        }
        let count = updates.len();
        if count > 0 {
            self.router.update(&updates)?;
        }
        Ok(count)
    }
}

fn source_id(event: &FieldEvent) -> Result<String, RoutingError> {
    let source = event
        .sensor
        .as_ref()
        .map(|sensor| sensor.device_id.clone())
        .or_else(|| {
            event
                .observation
                .zone_id
                .as_ref()
                .map(|zone| format!("zone:{zone}"))
        })
        .or_else(|| {
            event
                .observation
                .space_cell
                .map(|cell| format!("cell:{cell:?}"))
        })
        .ok_or(RoutingError::InvalidInput("RuField source"))?;
    if source.is_empty() || source.len() > 256 {
        return Err(RoutingError::InvalidInput("RuField source"));
    }
    Ok(source)
}

fn finite_unit(value: f32) -> Result<f32, RoutingError> {
    if !value.is_finite() {
        return Err(RoutingError::InvalidInput("RuField numeric value"));
    }
    Ok(value.clamp(0.0, 1.0))
}
fn risk_millionths(observation: &Observation) -> Result<u32, RoutingError> {
    let confidence = finite_unit(observation.confidence)?;
    let feature = |name: &str| observation.features.get(name).copied().unwrap_or(0.0);
    let presence = finite_unit(feature("presence"))?;
    let motion = finite_unit(feature("motion_energy"))?;
    let transient = finite_unit(feature("transient"))?;
    // Weighted maximum keeps a strong independent signal visible while confidence
    // gates the full observation. Integer output makes route effects reproducible.
    let raw = presence.max(0.8 * motion).max(0.6 * transient) * confidence;
    Ok((raw * 1_000_000.0).round() as u32)
}
