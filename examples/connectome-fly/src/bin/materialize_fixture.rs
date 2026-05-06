//! Write the built-in 100-neuron FlyWire-format fixture to a
//! directory. Use as a quick end-to-end smoke for `ui_server`:
//!
//!   cargo run --release --bin materialize_fixture -- /tmp/flywire-fixture
//!   CONNECTOME_FLYWIRE_DIR=/tmp/flywire-fixture \
//!     cargo run --release --bin ui_server
//!
//! The fixture is the same one used by `tests/flywire_streaming.rs`,
//! so if `ui_server` can stand it up, the ingest path holds.

use std::path::PathBuf;

use connectome_fly::connectome::flywire::fixture;

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("flywire-fixture"));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let paths = fixture::write_fixture(&dir).expect("write fixture");
    println!("wrote FlyWire v783 fixture to {:?}", dir);
    println!("  neurons.tsv        = {:?}", paths.neurons);
    println!("  connections.tsv    = {:?}", paths.connections);
    println!("  classification.tsv = {:?}", paths.classification);
}
