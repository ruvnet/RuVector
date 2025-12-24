mod app;
mod components;
mod panels;
mod theme;

pub use app::LeviathanApp;
pub use theme::LeviathanTheme;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// WASM entry point
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn start_app(canvas_id: &str) -> Result<(), JsValue> {
    // Set up console error panic hook for better error messages
    console_error_panic_hook::set_once();

    // Initialize logging
    tracing_wasm::set_as_global_default();

    // Create web options
    let web_options = eframe::WebOptions::default();

    // Start the app
    wasm_bindgen_futures::spawn_local(async move {
        eframe::WebRunner::new()
            .start(
                canvas_id,
                web_options,
                Box::new(|cc| Ok(Box::new(LeviathanApp::new(cc)))),
            )
            .await
            .expect("Failed to start eframe");
    });

    Ok(())
}

/// Native entry point (for testing)
#[cfg(not(target_arch = "wasm32"))]
pub fn start_native() -> Result<(), eframe::Error> {
    env_logger::init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("Leviathan UI - Windows 95 Cyberpunk Edition"),
        ..Default::default()
    };

    eframe::run_native(
        "leviathan-ui",
        native_options,
        Box::new(|cc| Ok(Box::new(LeviathanApp::new(cc)))),
    )
}
