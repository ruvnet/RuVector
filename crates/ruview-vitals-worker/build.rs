//! Build script — compile `proto/vitals.proto` via tonic + the bundled
//! protoc binary so we don't depend on a system install.

// `set_var` is `unsafe` in Rust 2024+; the build script runs single-
// threaded, so this is sound. The crate-wide `unsafe_code = "deny"`
// lint is overridden here, not in lib/bin code.
#![allow(unsafe_code)]

use std::env;

fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path()
        .expect("protoc-bin-vendored should ship a protoc for this host");
    // SAFETY: build.rs runs single-threaded.
    unsafe {
        env::set_var("PROTOC", protoc);
    }

    println!("cargo:rerun-if-changed=proto/vitals.proto");
    println!("cargo:rerun-if-changed=build.rs");

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/vitals.proto"], &["proto"])
        .expect("tonic-build failed to compile vitals.proto");
}
