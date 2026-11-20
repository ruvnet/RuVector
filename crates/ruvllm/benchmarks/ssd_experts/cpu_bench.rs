mod metrics;
use lattice_inference::generation::GenerateConfig;
use lattice_inference::model::qwen35::{Qwen35Model, RawGenEvent};
use serde_json::json;
use std::path::Path;
use std::time::Instant;

const PROMPT: &str = "Write a detailed technical guide to how a solid state drive stores and retrieves data. Begin with the physical structure of a NAND cell and continue through the read path.\n\n";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: cpu_bench MODEL_DIR TOKEN_COUNT REPEATS".into());
    }
    let target: usize = args[2].parse()?;
    let repeats: usize = args[3].parse()?;
    if target < 2 || repeats == 0 {
        return Err("need at least two tokens and one run".into());
    }
    let load = Instant::now();
    let mut model = Qwen35Model::from_safetensors(Path::new(&args[1]))?;
    let load_s = load.elapsed().as_secs_f64();
    let eos = model.config().eos_token_id;
    model.set_eos_token_id(u32::MAX);
    let mut cfg = GenerateConfig::default();
    cfg.max_new_tokens = target;
    cfg.top_k = 1;
    cfg.temperature = 0.0;
    cfg.top_p = 1.0;
    cfg.repetition_penalty = 1.0;
    cfg.stop_token_ids.clear();
    cfg.stop_strings.clear();
    cfg.enable_mtp = Some(false);
    cfg.seed = Some(1);
    let prompt = std::env::var("RUVLLM_BENCH_PROMPT").unwrap_or_else(|_| PROMPT.to_owned());
    for trial in 0..repeats {
        let before = model.bench_cpu_cache();
        let mut times = Vec::with_capacity(target);
        let os_before = metrics::sample();
        let mut os_at_first = None;
        let mut probe_time = std::time::Duration::ZERO;
        let start = Instant::now();
        let out = model.generate_streaming_with_observer(
            &prompt,
            &cfg,
            |_| true,
            || false,
            |event| {
                if let RawGenEvent::RawToken { .. } = event {
                    times.push(start.elapsed());
                    if times.len() == 1 {
                        let t = Instant::now();
                        os_at_first = Some(metrics::sample());
                        probe_time = t.elapsed();
                    }
                }
            },
        )?;
        if times.len() != target || out.token_ids.len() != target {
            return Err("short decode".into());
        }
        let decode_s = (times[target - 1] - times[0] - probe_time).as_secs_f64();
        println!(
            "{}",
            json!({"trial":trial, "prompt_tokens":out.prompt_tokens, "ids":out.token_ids,
            "os_before":os_before, "os_at_first":os_at_first, "os_after":metrics::sample(), "cache_before":before, "cache_after":model.bench_cpu_cache(), "eos_id":eos, "load_s":load_s, "ttft_s":times[0].as_secs_f64(),
            "decode_s":decode_s, "decode_tps":(target-1) as f64/decode_s})
        );
    }
    Ok(())
}
