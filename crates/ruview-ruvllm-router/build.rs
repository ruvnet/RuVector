#[allow(unsafe_code)]
fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc");
    // SAFETY: set before any threads start in build.rs
    unsafe { std::env::set_var("PROTOC", protoc) };
    tonic_build::configure()
        .build_server(true)
        .build_client(true)   // router needs client stubs to forward to backends
        .compile_protos(&["proto/llm.proto"], &["proto"])
        .expect("proto compile");
}
