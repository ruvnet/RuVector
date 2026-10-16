fn main() {
    std::process::exit(ruvector_language::cli::main_with(
        ruvector_language_native::load_models,
    ));
}
