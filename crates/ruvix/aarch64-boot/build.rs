fn main() {
    println!("cargo:rerun-if-changed=linker.ld");
    println!("cargo:rustc-link-arg=-Tlinker.ld");
}
