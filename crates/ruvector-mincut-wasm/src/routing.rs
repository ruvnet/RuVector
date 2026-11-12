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

use ruvector_mincut::routing::rufield::{AwarenessPolicy, RuFieldRouter as FieldRouter};
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FieldUpdate {
    node: u32,
    risk_millionths: u32,
    changed_arcs: usize,
    duplicate: bool,
}
/// RuField-aware route planner. Verify live event receipts before ingest.
#[wasm_bindgen]
pub struct WasmRuFieldRouter {
    inner: FieldRouter,
}
#[wasm_bindgen]
impl WasmRuFieldRouter {
    #[wasm_bindgen(constructor)]
    pub fn new(
        nodes: f64,
        endpoints: &[u32],
        costs: &[u32],
        turns: &[u32],
        max_penalty: f64,
        close_at_millionths: f64,
        ttl_ns: f64,
        max_lateness_ns: f64,
    ) -> Result<Self, JsError> {
        let road =
            Router::from_arrays(integer(nodes, 1_000_000)? as usize, endpoints, costs, turns)
                .map_err(|e| JsError::new(&e.to_string()))?;
        let policy = AwarenessPolicy {
            max_penalty: integer(max_penalty, 1_000_000_000)?,
            close_at_millionths: integer(close_at_millionths, 1_000_001)?,
            ttl_ns: integer64(ttl_ns, 3_600_000_000_000)?,
            max_lateness_ns: integer64(max_lateness_ns, 3_600_000_000_000)?,
        };
        Ok(Self {
            inner: FieldRouter::new(road, policy).map_err(|e| JsError::new(&e.to_string()))?,
        })
    }
    #[wasm_bindgen(js_name=bindZone)]
    pub fn bind_zone(&mut self, zone: String, node: f64) -> Result<(), JsError> {
        self.inner
            .bind_zone(zone, integer(node, u32::MAX)?)
            .map_err(|e| JsError::new(&e.to_string()))
    }
    #[wasm_bindgen(js_name=bindCell)]
    pub fn bind_cell(&mut self, x: i32, y: i32, z: i32, node: f64) -> Result<(), JsError> {
        self.inner
            .bind_cell([x, y, z], integer(node, u32::MAX)?)
            .map_err(|e| JsError::new(&e.to_string()))
    }
    #[wasm_bindgen(js_name=ingestRuField)]
    pub fn ingest_rufield(
        &mut self,
        json: &str,
        verified: bool,
        now_ns: f64,
    ) -> Result<JsValue, JsError> {
        let u = self
            .inner
            .ingest_json(
                json.as_bytes(),
                verified,
                integer64(now_ns, 9_007_199_254_740_991)?,
            )
            .map_err(|e| JsError::new(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&FieldUpdate {
            node: u.node,
            risk_millionths: u.risk_millionths,
            changed_arcs: u.changed_arcs,
            duplicate: u.duplicate,
        })
        .map_err(|e| JsError::new(&e.to_string()))
    }
    pub fn expire(&mut self, now_ns: f64) -> Result<usize, JsError> {
        self.inner
            .expire(integer64(now_ns, 9_007_199_254_740_991)?)
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
    #[wasm_bindgen(js_name=activeNodes)]
    pub fn active_nodes(&self) -> usize {
        self.inner.active_nodes()
    }
}
fn integer64(v: f64, max: u64) -> Result<u64, JsError> {
    if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v > max as f64 {
        Err(JsError::new("expected bounded nonnegative integer"))
    } else {
        Ok(v as u64)
    }
}
