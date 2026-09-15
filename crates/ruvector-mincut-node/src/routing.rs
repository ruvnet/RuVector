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
