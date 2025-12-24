use leviathan_ui::start_native;

fn main() -> Result<(), eframe::Error> {
    env_logger::init();

    println!("🚀 Starting Leviathan UI (Native)...");
    println!("   Windows 95 + Cyberpunk Edition");
    println!();

    start_native()
}
