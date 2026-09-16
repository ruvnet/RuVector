use napi::bindgen_prelude::*;
use napi_derive::napi;
use ruvector_mincut::routing::{RoadRouter as Router, RoutingError};

fn error(e: RoutingError) -> Error {
    Error::from_reason(e.to_string())
}
fn integer(v: f64, max: u32) -> Result<u32> {
    if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v > max as f64 {
        return Err(Error::from_reason("expected bounded nonnegative integer"));
    }
    Ok(v as u32)
}
#[napi(object)]
pub struct RoadRoute {
    pub cost: f64,
    pub nodes: Vec<u32>,
    pub arcs: Vec<u32>,
    pub settled: u32,
}
#[napi(object)]
pub struct RoadSnap {
    pub node: u32,
    pub distance_m: f64,
}
/// Directed routing with integer costs and stable input-position arc IDs.
#[napi]
pub struct RoadRouter {
    inner: Option<Router>,
}
#[napi]
impl RoadRouter {
    #[napi(constructor)]
    pub fn new(
        nodes: f64,
        endpoints: Uint32Array,
        costs: Uint32Array,
        turns: Uint32Array,
    ) -> Result<Self> {
        Ok(Self {
            inner: Some(
                Router::from_arrays(
                    integer(nodes, 1_000_000)? as usize,
                    &endpoints,
                    &costs,
                    &turns,
                )
                .map_err(error)?,
            ),
        })
    }
    #[napi]
    pub fn prepare(&mut self, landmarks: Uint32Array, budget: f64) -> Result<()> {
        self.inner
            .as_mut()
            .ok_or_else(|| Error::from_reason("router cleared"))?
            .prepare(&landmarks, integer(budget, 200_000_000)? as usize, || false)
            .map_err(error)
    }
    #[napi]
    pub fn route(
        &mut self,
        source: f64,
        target: f64,
        use_landmarks: bool,
        budget: f64,
    ) -> Result<Option<RoadRoute>> {
        let (s, t, b) = (
            integer(source, u32::MAX)?,
            integer(target, u32::MAX)?,
            integer(budget, 200_000_000)?,
        );
        Ok(self
            .inner
            .as_mut()
            .ok_or_else(|| Error::from_reason("router cleared"))?
            .route(s, t, use_landmarks, b as usize, || false)
            .map_err(error)?
            .map(|r| RoadRoute {
                cost: r.cost as f64,
                nodes: r.nodes,
                arcs: r.arcs,
                settled: r.settled as u32,
            }))
    }
    /// u32::MAX closes an arc; all other costs must be <= 1e9.
    #[napi]
    pub fn update(&mut self, ids: Uint32Array, costs: Uint32Array) -> Result<()> {
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| Error::from_reason("router cleared"))?;
        if ids.len() != costs.len() || ids.len() > inner.arc_count() {
            return Err(Error::from_reason("update count"));
        }
        let updates: Vec<_> = ids
            .iter()
            .zip(costs.iter())
            .map(|(&id, &c)| (id, (c != u32::MAX).then_some(c)))
            .collect();
        inner.update(&updates).map_err(error)
    }
    #[napi]
    pub fn set_coordinates(&mut self, lat_lon: Float64Array) -> Result<()> {
        self.inner
            .as_mut()
            .ok_or_else(|| Error::from_reason("router cleared"))?
            .set_coordinates(&lat_lon)
            .map_err(error)
    }
    #[napi]
    pub fn nearest(&self, lat: f64, lon: f64, radius_m: f64) -> Result<Option<RoadSnap>> {
        Ok(self
            .inner
            .as_ref()
            .ok_or_else(|| Error::from_reason("router cleared"))?
            .nearest(lat, lon, radius_m)
            .map_err(error)?
            .map(|r| RoadSnap {
                node: r.node,
                distance_m: r.distance_m,
            }))
    }
    #[napi]
    pub fn clear(&mut self) {
        self.inner = None;
    }
}

use ruvector_mincut::routing::rufield::{AwarenessPolicy, RuFieldRouter as FieldRouter};

#[napi(object)]
pub struct RuFieldPolicy {
    pub max_penalty: Option<u32>,
    pub close_at_millionths: Option<u32>,
    pub ttl_ns: Option<f64>,
    pub max_lateness_ns: Option<f64>,
}
#[napi(object)]
pub struct RuFieldUpdate {
    pub node: u32,
    pub risk_millionths: u32,
    pub changed_arcs: u32,
    pub duplicate: bool,
}
/// Exact route planner whose costs respond to verified, privacy-safe RuField events.
#[napi]
pub struct RuFieldRoadRouter {
    inner: Option<FieldRouter>,
}
#[napi]
impl RuFieldRoadRouter {
    #[napi(constructor)]
    pub fn new(
        nodes: f64,
        endpoints: Uint32Array,
        costs: Uint32Array,
        turns: Uint32Array,
        policy: Option<RuFieldPolicy>,
    ) -> Result<Self> {
        let p = policy.unwrap_or(RuFieldPolicy {
            max_penalty: None,
            close_at_millionths: None,
            ttl_ns: None,
            max_lateness_ns: None,
        });
        let defaults = AwarenessPolicy::default();
        let policy = AwarenessPolicy {
            max_penalty: p.max_penalty.unwrap_or(defaults.max_penalty),
            close_at_millionths: p
                .close_at_millionths
                .unwrap_or(defaults.close_at_millionths),
            ttl_ns: p
                .ttl_ns
                .map(|v| integer64(v, 3_600_000_000_000))
                .transpose()?
                .unwrap_or(defaults.ttl_ns),
            max_lateness_ns: p
                .max_lateness_ns
                .map(|v| integer64(v, 3_600_000_000_000))
                .transpose()?
                .unwrap_or(defaults.max_lateness_ns),
        };
        let road = Router::from_arrays(
            integer(nodes, 1_000_000)? as usize,
            &endpoints,
            &costs,
            &turns,
        )
        .map_err(error)?;
        Ok(Self {
            inner: Some(FieldRouter::new(road, policy).map_err(error)?),
        })
    }
    #[napi]
    pub fn bind_zone(&mut self, zone: String, node: f64) -> Result<()> {
        self.get()?
            .bind_zone(zone, integer(node, u32::MAX)?)
            .map_err(error)
    }
    #[napi]
    pub fn bind_cell(&mut self, x: i32, y: i32, z: i32, node: f64) -> Result<()> {
        self.get()?
            .bind_cell([x, y, z], integer(node, u32::MAX)?)
            .map_err(error)
    }
    #[napi]
    pub fn ingest_rufield(
        &mut self,
        json: String,
        verified: bool,
        now_ns: f64,
    ) -> Result<RuFieldUpdate> {
        let u = self
            .get()?
            .ingest_json(
                json.as_bytes(),
                verified,
                integer64(now_ns, 9_007_199_254_740_991)?,
            )
            .map_err(error)?;
        Ok(RuFieldUpdate {
            node: u.node,
            risk_millionths: u.risk_millionths,
            changed_arcs: u.changed_arcs as u32,
            duplicate: u.duplicate,
        })
    }
    #[napi]
    pub fn expire(&mut self, now_ns: f64) -> Result<u32> {
        Ok(self
            .get()?
            .expire(integer64(now_ns, 9_007_199_254_740_991)?)
            .map_err(error)? as u32)
    }
    #[napi]
    pub fn route(
        &mut self,
        source: f64,
        target: f64,
        use_landmarks: bool,
        budget: f64,
    ) -> Result<Option<RoadRoute>> {
        let (s, t, b) = (
            integer(source, u32::MAX)?,
            integer(target, u32::MAX)?,
            integer(budget, 200_000_000)?,
        );
        Ok(self
            .get()?
            .route(s, t, use_landmarks, b as usize, || false)
            .map_err(error)?
            .map(|r| RoadRoute {
                cost: r.cost as f64,
                nodes: r.nodes,
                arcs: r.arcs,
                settled: r.settled as u32,
            }))
    }
    #[napi]
    pub fn active_nodes(&self) -> Result<u32> {
        Ok(self
            .inner
            .as_ref()
            .ok_or_else(|| Error::from_reason("router cleared"))?
            .active_nodes() as u32)
    }
    #[napi]
    pub fn clear(&mut self) {
        self.inner = None;
    }
}
impl RuFieldRoadRouter {
    fn get(&mut self) -> Result<&mut FieldRouter> {
        self.inner
            .as_mut()
            .ok_or_else(|| Error::from_reason("router cleared"))
    }
}
fn integer64(v: f64, max: u64) -> Result<u64> {
    if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v > max as f64 {
        Err(Error::from_reason("expected bounded nonnegative integer"))
    } else {
        Ok(v as u64)
    }
}
