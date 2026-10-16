fn main() {
    std::process::exit(ruvector_language::cli::main_with(|_| {
        Err(ruvector_language::Error::new(
            "MODEL_UNAVAILABLE",
            "Use the native rvg binary to load RuVector KGE models",
        ))
    }));
}
