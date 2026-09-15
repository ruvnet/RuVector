use ruvector_mincut::routing::RoadRouter as Router;
use serde::Serialize;
use wasm_bindgen::prelude::*;
fn integer(v: f64, max: u32) -> Result<u32, JsError> {
    if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v > max as f64 {
        return Err(JsError::new("expected bounded nonnegative integer"));
    }
    Ok(v as u32)
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Route {
    cost: f64,
    nodes: Vec<u32>,
    arcs: Vec<u32>,
    settled: usize,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Snap {
    node: u32,
    distance_m: f64,
}
/// Directed integer-cost routing. Arc IDs are input positions. Run in a Worker.
#[wasm_bindgen]
pub struct WasmRoadRouter {
    inner: Router,
}
#[wasm_bindgen]
impl WasmRoadRouter {
    #[wasm_bindgen(constructor)]
    pub fn new(
        nodes: f64,
        endpoints: &[u32],
        costs: &[u32],
        turns: &[u32],
    ) -> Result<WasmRoadRouter, JsError> {
        Ok(Self {
            inner: Router::from_arrays(
                integer(nodes, 1_000_000)? as usize,
                endpoints,
                costs,
                turns,
            )
            .map_err(|e| JsError::new(&e.to_string()))?,
        })
    }
    pub fn prepare(&mut self, landmarks: &[u32], budget: f64) -> Result<(), JsError> {
        self.inner
            .prepare(landmarks, integer(budget, 200_000_000)? as usize, || false)
            .map_err(|e| JsError::new(&e.to_string()))
    }
    pub fn route(
        &mut self,
        source: f64,
        target: f64,
        use_landmarks: bool,
        budget: f64,
    ) -> Result<JsValue, JsError> {
        let route = self
            .inner
            .route(
                integer(source, u32::MAX)?,
                integer(target, u32::MAX)?,
                use_landmarks,
                integer(budget, 200_000_000)? as usize,
                || false,
            )
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&route.map(|r| Route {
            cost: r.cost as f64,
            nodes: r.nodes,
            arcs: r.arcs,
            settled: r.settled,
        }))
        .map_err(|e| JsError::new(&e.to_string()))
    }
    /// u32::MAX closes an arc. Other costs must be <= 1e9.
    pub fn update(&mut self, ids: &[u32], costs: &[u32]) -> Result<(), JsError> {
        if ids.len() != costs.len() || ids.len() > self.inner.arc_count() {
            return Err(JsError::new("update count"));
        }
        let updates: Vec<_> = ids
            .iter()
            .zip(costs)
            .map(|(&id, &c)| (id, (c != u32::MAX).then_some(c)))
            .collect();
        self.inner
            .update(&updates)
            .map_err(|e| JsError::new(&e.to_string()))
    }
    #[wasm_bindgen(js_name=setCoordinates)]
    pub fn set_coordinates(&mut self, lat_lon: &[f64]) -> Result<(), JsError> {
        self.inner
            .set_coordinates(lat_lon)
            .map_err(|e| JsError::new(&e.to_string()))
    }
    pub fn nearest(&self, lat: f64, lon: f64, radius_m: f64) -> Result<JsValue, JsError> {
        let snap = self
            .inner
            .nearest(lat, lon, radius_m)
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&snap.map(|r| Snap {
            node: r.node,
            distance_m: r.distance_m,
        }))
        .map_err(|e| JsError::new(&e.to_string()))
    }
}
