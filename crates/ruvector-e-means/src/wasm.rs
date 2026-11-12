use crate::{fit, predict, Config, Model};
use wasm_bindgen::prelude::*;

/// Compatibility transport and an independent parity oracle for typed output.
#[wasm_bindgen(js_name = fitJson)]
pub fn fit_json(data: &[f32], dimensions: usize, config_json: &str) -> Result<String, JsError> {
    let config: Config =
        serde_json::from_str(config_json).map_err(|e| JsError::new(&e.to_string()))?;
    let model = fit(data, dimensions, &config).map_err(|e| JsError::new(&e.to_string()))?;
    serde_json::to_string(&model).map_err(|e| JsError::new(&e.to_string()))
}

/// A temporary owner for one fit result. JavaScript takes independent output
/// arrays and then frees this handle, releasing any vectors not taken.
#[wasm_bindgen]
pub struct FittedModel {
    model: Model,
}

#[wasm_bindgen]
impl FittedModel {
    /// wasm-bindgen copies the returned vector into a caller-owned Float32Array
    /// before freeing its Rust allocation. No view into WASM memory escapes.
    #[wasm_bindgen(js_name = takeCentroids)]
    pub fn take_centroids(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.model.centroids)
    }

    /// Cluster indices fit u32 because wasm32 arrays and sample counts do.
    #[wasm_bindgen(js_name = takeAssignments)]
    pub fn take_assignments(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.model.assignments)
            .into_iter()
            .map(|index| index as u32)
            .collect()
    }

    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> usize {
        self.model.dimensions
    }

    #[wasm_bindgen(getter)]
    pub fn mse(&self) -> f64 {
        self.model.mse
    }

    #[wasm_bindgen(getter, js_name = batchSize)]
    pub fn batch_size(&self) -> usize {
        self.model.batch_size
    }

    #[wasm_bindgen(getter, js_name = peakStep)]
    pub fn peak_step(&self) -> f64 {
        self.model.peak_step
    }

    #[wasm_bindgen(getter)]
    pub fn updates(&self) -> usize {
        self.model.updates
    }

    #[wasm_bindgen(getter, js_name = revivedClusters)]
    pub fn revived_clusters(&self) -> usize {
        self.model.revived_clusters
    }
}

/// Only the small configuration crosses JSON; result vectors stay numeric.
#[wasm_bindgen(js_name = fitTyped)]
pub fn fit_typed(
    data: &[f32],
    dimensions: usize,
    config_json: &str,
) -> Result<FittedModel, JsError> {
    let config: Config =
        serde_json::from_str(config_json).map_err(|e| JsError::new(&e.to_string()))?;
    fit(data, dimensions, &config)
        .map(|model| FittedModel { model })
        .map_err(|e| JsError::new(&e.to_string()))
}

#[wasm_bindgen(js_name = predictFlat)]
pub fn predict_flat(
    data: &[f32],
    dimensions: usize,
    centroids: &[f32],
) -> Result<Vec<u32>, JsError> {
    predict(data, dimensions, centroids)
        .map(|v| v.into_iter().map(|j| j as u32).collect())
        .map_err(|e| JsError::new(&e.to_string()))
}
