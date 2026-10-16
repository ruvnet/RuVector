//! Offline model commands. Outputs use create_new and never overwrite models.
use crate::{
    artifact::ModelBundle,
    decode,
    training::{self, Candidate, EvaluationRequest, TrainingRequest},
};
use ruvector_language::error::{Error, Result};
use serde::Serialize;
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::Path,
};

const MAX_BYTES: u64 = 16 * 1024 * 1024;
const USAGE: &str = "rvg-model train REQUEST.json CANDIDATE.json\nrvg-model evaluate MODELS.json MODEL EVALUATION.json REPORT.json\nrvg-model promote CANDIDATE.json MINIMUM_MRR MODELS.json\nOutputs must not already exist. Training never mutates a live model.";

fn read<T: serde::de::DeserializeOwned>(path: &str) -> Result<T> {
    let file = File::open(path).map_err(|e| Error::new("IO", e.to_string()))?;
    if file
        .metadata()
        .map_err(|e| Error::new("IO", e.to_string()))?
        .len()
        > MAX_BYTES
    {
        return Err(Error::new("LIMIT", "model input exceeds 16 MiB"));
    }
    let mut bytes = Vec::new();
    file.take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::new("IO", e.to_string()))?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(Error::new("LIMIT", "model input exceeds 16 MiB"));
    }
    let json =
        std::str::from_utf8(&bytes).map_err(|e| Error::new("MODEL_FORMAT", e.to_string()))?;
    decode(json)
}

fn write<T: Serialize>(path: &str, value: &T) -> Result<()> {
    let bytes =
        serde_json::to_vec_pretty(value).map_err(|e| Error::new("SERIALIZE", e.to_string()))?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(Error::new("LIMIT", "model output exceeds 16 MiB"));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(Path::new(path))
        .map_err(|e| Error::new("IO", e.to_string()))?;
    let result = file
        .write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all());
    if let Err(error) = result {
        drop(file);
        // Only remove the file this invocation exclusively created.
        let _ = std::fs::remove_file(path);
        return Err(Error::new("IO", error.to_string()));
    }
    Ok(())
}

pub fn run(args: &[String]) -> Result<()> {
    match args {
        [command, input, output] if command == "train" => {
            let request: TrainingRequest = read(input)?;
            write(output, &training::train_candidate(&request)?)
        }
        [command, input, name, evaluation, output] if command == "evaluate" => {
            let bundle: ModelBundle = read(input)?;
            // Validate the complete bundle, including duplicate model identifiers.
            crate::NativePredictor::new(bundle.clone())?;
            let model = bundle
                .models
                .iter()
                .find(|m| m.name == *name)
                .ok_or_else(|| Error::new("MODEL", "model not found"))?;
            let request: EvaluationRequest = read(evaluation)?;
            write(output, &training::evaluate_model(model, &request)?)
        }
        [command, input, threshold, output] if command == "promote" => {
            let candidate: Candidate = read(input)?;
            let minimum_mrr = threshold.parse::<f32>().map_err(|_| {
                Error::new("USAGE", "MRR threshold must be a finite number from 0 to 1")
            })?;
            write(output, &training::promote(&candidate, minimum_mrr)?)
        }
        _ => Err(Error::new("USAGE", USAGE)),
    }
}

pub fn main() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.as_slice() == ["--help"] || args.as_slice() == ["help"] {
        println!("{USAGE}");
        return 0;
    }
    match run(&args) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!(
                "{}",
                serde_json::to_string(&error).unwrap_or_else(|_| "{\"code\":\"ERROR\"}".into())
            );
            1
        }
    }
}
