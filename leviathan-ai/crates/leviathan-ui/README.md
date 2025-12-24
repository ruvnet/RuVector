# Leviathan UI

Windows 95 aesthetic + Cyberpunk corporate clean UI for Leviathan AI, built with egui and WASM.

## Features

- **Win95 Aesthetic**: Classic 3D beveled buttons, sharp corners, sunken text fields
- **Cyberpunk Accents**: Neon cyan/magenta glow effects, dark backgrounds, monospace fonts
- **WASM Ready**: Runs in browser with full WebAssembly support
- **Multiple Panels**:
  - Dashboard: System metrics and overview
  - Swarm: Agent visualization and management
  - Audit: DAG explorer with lineage tracking
  - Terminal: Command input/output interface
  - Config: System configuration with property sheets

## Quick Start

### Web (WASM) - Recommended

```bash
# Build and serve (automatic)
./serve.sh

# Or manually:
./build.sh
python3 -m http.server -d web 8080
# Then open http://localhost:8080
```

### Native (Desktop)

```bash
# Run example
cargo run --example native_demo

# Or run library directly (requires main binary)
cargo run --release
```

## Building

### For Web (WASM)

```bash
# Install wasm-pack if not already installed
cargo install wasm-pack

# Build for web
wasm-pack build --target web --out-dir web/pkg

# Or use the build script
./build.sh
```

### For Native (Desktop)

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Run directly
cargo run --example native_demo
```

## Development

```bash
# Check code
cargo check

# Run tests
cargo test

# Format
cargo fmt

# Lint
cargo clippy
```

## Design Philosophy

This UI combines the nostalgic clarity of Windows 95 with modern cyberpunk aesthetics:

- **Win95 Elements**:
  - 3D raised/sunken borders
  - Sharp corners (no rounding)
  - Classic window chrome
  - Taskbar and system tray
  - Status bar with sections

- **Cyberpunk Touches**:
  - Dark color palette (#1a1a2e, #16213e)
  - Neon accents (cyan #00fff5, magenta #ff00ff)
  - Subtle scanline effects
  - Glow on hover states
  - Monospace fonts throughout
  - Matrix green for success states

## Theme Colors

```rust
bg_primary:      #1a1a2e  // Main dark background
bg_secondary:    #16213e  // Panel backgrounds
bg_tertiary:     #0f0f1e  // Deeper dark areas
accent_cyan:     #00fff5  // Primary neon accent
accent_magenta:  #ff00ff  // Secondary accent
accent_purple:   #8a2be2  // Tertiary accent
success:         #00ff41  // Matrix green
warning:         #ffb400  // Amber warning
error:           #ff2850  // Neon red
```

## License

MIT
