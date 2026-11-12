fn main() {
    println!("cargo:rustc-check-cfg=cfg(ruvllm_stable_avx512)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");

    // AVX-512 stdarch intrinsics and target features became stable in 1.89.
    // Query Cargo's selected compiler, not the host CPU or rustc from PATH:
    // cross-compiles and older default-feature builds must retain fallbacks.
    let version =
        rustc_version::version().expect("cannot determine selected Rust compiler version");
    if version >= rustc_version::Version::new(1, 89, 0) {
        println!("cargo:rustc-cfg=ruvllm_stable_avx512");
    }
}
