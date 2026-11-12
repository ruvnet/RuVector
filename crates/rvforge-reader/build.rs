fn main() {
    // Tauri's build step reads tauri.conf.json and generates the asset
    // registry. It is only needed for the desktop shell; the core library
    // builds without it.
    #[cfg(feature = "desktop")]
    tauri_build::build();
}
