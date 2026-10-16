//! Process boundary only. The runtime itself has no I/O or ambient clock.
use crate::{compiler, parser, runtime, Error, Predictor, Program, Receipt, Result, RunRequest};
use serde::de::DeserializeOwned;
use std::collections::BTreeMap;
use std::io::Read;

pub type ModelLoader = fn(&str) -> Result<Box<dyn Predictor>>;
const MAX_BYTES: u64 = 16 * 1024 * 1024;
const HELP: &str = "RuGraph 0.1.0\n\
Usage:\n\
  rvg check PROGRAM\n\
  rvg compile PROGRAM\n\
  rvg run PROGRAM --snapshot SNAPSHOT --context CONTEXT [--models MODELS]\n\
  rvg run --request REQUEST [--models MODELS]\n\
  rvg replay --request REQUEST --receipt RECEIPT [--models MODELS]\n\
  rvg audit --request REQUEST --audit-options OPTIONS\n\
  rvg verify RECEIPT\n\
PROGRAM accepts .rvg text or strict program JSON. REQUEST is RunRequest JSON.\n\
Use - for stdin in one input position. Results are JSON on stdout; errors are\n\
JSON on stderr with exit code 2. Clock and capabilities come from CONTEXT.\n";

fn read(path: &str) -> Result<String> {
    let reader: Box<dyn Read> = if path == "-" {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(path).map_err(|e| Error::new("IO_ERROR", e.to_string()))?)
    };
    let mut bytes = Vec::new();
    reader
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::new("IO_ERROR", e.to_string()))?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(Error::new("INPUT_LIMIT", "Input exceeds 16 MiB"));
    }
    String::from_utf8(bytes).map_err(|_| Error::new("INVALID_INPUT", "Input must be UTF-8"))
}

fn decode<T: DeserializeOwned>(text: &str) -> Result<T> {
    serde_json::from_value(parser::strict_json(text)?)
        .map_err(|e| Error::new("INVALID_INPUT", e.to_string()))
}

fn program(text: &str) -> Result<Program> {
    if text.trim_start().starts_with('{') {
        parser::parse_json(text)
    } else {
        parser::parse(text)
    }
}

fn json<T: serde::Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(value).map_err(|e| Error::new("SERIALIZATION", e.to_string()))
}

fn required<'a>(options: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str> {
    options
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| Error::new("USAGE", format!("Missing {key}")))
}

/// Execute the public CLI without spawning a shell or making provider calls.
pub fn run_cli(args: &[String], loader: ModelLoader) -> Result<String> {
    if args.len() == 1 && args[0] == "--version" {
        return Ok("0.1.0".into());
    }
    if args.is_empty() || args == ["--help"] || args == ["help"] {
        return Ok(HELP.into());
    }
    let command = &args[0];
    let mut options = BTreeMap::new();
    let mut positional = Vec::new();
    let mut index = 1;
    while index < args.len() {
        let arg = &args[index];
        if arg.starts_with("--") {
            if ![
                "--audit-options",
                "--snapshot",
                "--context",
                "--request",
                "--receipt",
                "--models",
            ]
            .contains(&arg.as_str())
            {
                return Err(Error::new("USAGE", format!("Unknown option {arg}")));
            }
            let value = args
                .get(index + 1)
                .filter(|v| !v.starts_with("--"))
                .ok_or_else(|| Error::new("USAGE", format!("Missing value for {arg}")))?;
            if options.insert(arg.clone(), value.clone()).is_some() {
                return Err(Error::new("USAGE", format!("Duplicate option {arg}")));
            }
            index += 2;
        } else {
            positional.push(arg.clone());
            index += 1;
        }
    }
    let stdin_count = positional
        .iter()
        .chain(options.values())
        .filter(|p| p.as_str() == "-")
        .count();
    if stdin_count > 1 {
        return Err(Error::new("USAGE", "Only one input may use stdin"));
    }
    if command != "audit" && options.contains_key("--audit-options") {
        return Err(Error::new(
            "USAGE",
            "--audit-options is only valid for audit",
        ));
    }
    match command.as_str() {
        "audit" => {
            if !positional.is_empty() || options.len() != 2 {
                return Err(Error::new(
                    "USAGE",
                    "audit requires only --request and --audit-options",
                ));
            }
            let request_path = required(&options, "--request")?;
            let options_path = required(&options, "--audit-options")?;
            let settings: crate::audit::AuditOptions = decode(&read(options_path)?)?;
            let request = parser::parse_request_json(&read(request_path)?)?;
            json(&crate::audit::run(&request, &settings)?)
        }
        "check" | "compile" | "verify" => {
            if positional.len() != 1 || !options.is_empty() {
                return Err(Error::new("USAGE", "Expected exactly one input path"));
            }
            let source = read(&positional[0])?;
            if command == "verify" {
                let receipt: Receipt = decode(&source)?;
                runtime::verify_receipt(&receipt)?;
                return json(
                    &serde_json::json!({"valid":true,"receipt_hash":receipt.receipt_hash,
                    "scope":"integrity only; not authenticity or replay"}),
                );
            }
            let parsed = program(&source)?;
            compiler::check(&parsed)?;
            if command == "compile" {
                compiler::canonical_json(&parsed)
            } else {
                json(&serde_json::json!({"valid":true,"name":parsed.name,"version":parsed.version}))
            }
        }
        "run" | "replay" => {
            if command == "replay" {
                required(&options, "--receipt")?;
                required(&options, "--request")?;
            }
            if command == "run" && options.contains_key("--receipt") {
                return Err(Error::new("USAGE", "--receipt is only valid for replay"));
            }
            let request = if let Some(path) = options.get("--request") {
                if !positional.is_empty()
                    || options.contains_key("--snapshot")
                    || options.contains_key("--context")
                {
                    return Err(Error::new(
                        "USAGE",
                        "--request cannot be combined with program, snapshot or context",
                    ));
                }
                parser::parse_request_json(&read(path)?)?
            } else {
                if command == "replay" || positional.len() != 1 {
                    return Err(Error::new("USAGE", "Expected program path or --request"));
                }
                RunRequest {
                    program: program(&read(&positional[0])?)?,
                    snapshot: decode(&read(required(&options, "--snapshot")?)?)?,
                    context: decode(&read(required(&options, "--context")?)?)?,
                }
            };
            let predictor = options
                .get("--models")
                .map(|path| read(path).and_then(|data| loader(&data)))
                .transpose()?;
            let receipt = runtime::execute(
                &request.program,
                &request.snapshot,
                &request.context,
                predictor.as_deref(),
            )?;
            if command == "replay" {
                let previous: Receipt = decode(&read(required(&options, "--receipt")?)?)?;
                runtime::verify_receipt(&previous)?;
                if previous != receipt {
                    return Err(Error::new(
                        "REPLAY_MISMATCH",
                        "Inputs, model state or output differ from the receipt",
                    ));
                }
            }
            json(&receipt)
        }
        _ => Err(Error::new("USAGE", format!("Unknown command {command}"))),
    }
}

/// Entry point shared by the minimal and native binaries.
pub fn main_with(loader: ModelLoader) -> i32 {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match run_cli(&args, loader) {
        Ok(output) => {
            println!("{output}");
            0
        }
        Err(error) => {
            eprintln!(
                "{}",
                serde_json::to_string(&error).unwrap_or_else(|_| "{\"code\":\"ERROR\"}".into())
            );
            2
        }
    }
}
