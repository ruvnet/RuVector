"""Instrument disposable copies pinned to RuVector 979c31a and Lattice 0.11.1."""
from pathlib import Path
import re
import subprocess

def replace_once(text, needle, change):
    if text.count(needle) != 1:
        raise ValueError(f'unexpected source around {needle[:65]!r}')
    return text.replace(needle,change)

def instrument(repo, crate):
    subprocess.run(['patch','-p1','-i',str(Path(__file__).resolve().with_name('moe_generation_prefill.patch'))],
                   cwd=crate,check=True)
    decoder=crate/'src/decoder/qwen_metal.rs'
    select='        let candidate_id = match &mut self.readback {'
    observe='''        if let Some(path) = std::env::var_os("RUVLLM_METAL_LOGITS") {
            use sha2::{Digest, Sha256};
            use std::io::Write;
            let Readback::Dense(values) = &self.readback else {
                return Err(InferenceError::Inference("the benchmark requires full Metal logits".into()));
            };
            assert_eq!(values.len(), self.state.engine.config.vocab_size);
            let mut hash = Sha256::new();
            let mut wire = [0u8; 4096];
            for block in values.chunks(1024) {
                for (value, bytes) in block.iter().zip(wire.chunks_exact_mut(4)) {
                    bytes.copy_from_slice(&value.to_le_bytes());
                }
                hash.update(&wire[..block.len()*4]);
            }
            let mut out = std::fs::OpenOptions::new().append(true).create(true).open(path)?;
            writeln!(out, "{:x}", hash.finalize())?;
        }
'''
    decoder.write_text(replace_once(decoder.read_text(),select,observe+select))
    cache=crate/'src/forward/moe_expert_cache.rs'
    code=cache.read_text()
    # Reuse upstream's real counters and synchronous slot resolver for
    # priming before the first GPU dispatch, exclusively in this build.
    pattern=(r'(?m)^( *)#\[cfg\(test\)\]\n(?=\s*(?:hit_count:|miss_count:|eviction_count:|'
             r'pub\(crate\) fn (?:num_slots|hit_miss_eviction_counts|resolve)\(|fn load_into\(|'
             r'\{\s*self\.(?:hit_count|miss_count|eviction_count) \+=))')
    code,n=re.subn(pattern,'',code)
    if n!=19: raise ValueError(f'unexpected cache instrumentation sites: {n}')
    cache.write_text(code)
    metal=crate/'src/forward/metal_qwen35.rs'
    code=metal.read_text()
    anchor='        pub fn from_q4_dir(\n'
    extra='''        pub fn parity_cache(&self) -> Vec<[u64; 7]> {
            let mut rows = Vec::new();
            for (_, common) in &self.engine.layer_weights {
                if let MetalFfnWeights::Moe(moe) = &common.ffn {
                    let RoutedExpertStorage::Cached {gate_up, down} = &moe.routed else {
                        panic!("benchmark requires a Q4 expert cache")
                    };
                    let gu = gate_up.borrow();
                    let dn = down.borrow();
                    assert_eq!(gu.num_slots(), dn.num_slots());
                    let (h,m,e) = gu.hit_miss_eviction_counts();
                    let (dh,dm,de) = dn.hit_miss_eviction_counts();
                    rows.push([gu.num_slots() as u64,h as u64,m as u64,e as u64,dh as u64,dm as u64,de as u64]);
                }
            }
            rows
        }

        pub fn parity_prime(&self, experts: usize) -> Result<(),String> {
            let rows = self.parity_cache();
            if rows.is_empty() || rows.iter().any(|r| r[0] != experts as u64) {
                return Err(format!("resident request was capped: requested {experts}, actual {:?}", rows.iter().map(|r|r[0]).collect::<Vec<_>>()));
            }
            for (_,common) in &self.engine.layer_weights {
                if let MetalFfnWeights::Moe(moe) = &common.ffn {
                    if let RoutedExpertStorage::Cached {gate_up,down} = &moe.routed {
                        let mut gu=gate_up.borrow_mut();
                        let mut dn=down.borrow_mut();
                        for expert in 0..experts { gu.resolve(expert); dn.resolve(expert); }
                    }
                }
            }
            Ok(())
        }

'''
    metal.write_text(replace_once(code,anchor,extra+anchor))
    backend=repo/'crates/ruvllm/src/backends/lattice_backend.rs'
    code=backend.read_text()
    anchor='        (metal, Quantization::Q4, sum_file_sizes(model_dir, "q4"))'
    code=replace_once(code,anchor,'''        let expected = std::env::var("LATTICE_MOE_EXPERT_CACHE_SLOTS").unwrap().parse::<u64>().unwrap();
        let rows = metal.parity_cache();
        if rows.len() != cfg.num_active_layers() || rows.iter().any(|r| r[0]!=expected) {
            return Err(format!("benchmark requires {expected} slots in every layer, actual {:?}", rows.iter().map(|r|r[0]).collect::<Vec<_>>()));
        }
        if std::env::var_os("RUVLLM_PRIME_ALL").is_some() {
            metal.parity_prime(cfg.num_experts.ok_or("checkpoint has no experts")?)?;
        }
'''+anchor)
    anchor='                Reply::Stream(tx) => {\n                    let start = Instant::now();'
    extra='''                Reply::Stream(tx) => {
                    let measure = || -> serde_json::Value {
                        let binary = std::env::var_os("RUVLLM_DARWIN_METRICS").unwrap();
                        let out = std::process::Command::new(binary).arg(std::process::id().to_string()).output().unwrap();
                        assert!(out.status.success(), "Darwin measurement unavailable");
                        serde_json::from_slice(&out.stdout).unwrap()
                    };
                    let before = metal.parity_cache();
                    let os_before = measure();
                    let mut ticks = Vec::new();
                    let mut os_at_first = serde_json::Value::Null;
                    let mut probe_time = 0.0;
                    let start = Instant::now();'''
    code=replace_once(code,anchor,extra)
    anchor='                            last_id = id;'
    code=replace_once(code,anchor,'''                            ticks.push(start.elapsed().as_secs_f64());
                            if ticks.len()==1 { let probe=Instant::now(); os_at_first=measure(); probe_time=probe.elapsed().as_secs_f64(); }
'''+anchor)
    anchor='                    let duration_ms = start.elapsed().as_millis() as u64;'
    code=replace_once(code,anchor,'''                    let os_after = measure();
                    let decode = ticks.last().zip(ticks.first()).map(|(b,a)| b-a-probe_time);
                    eprintln!("PARITY {}", serde_json::json!({"ids":out.token_ids, "captured_callbacks":ticks.len(),
                        "prompt_tokens":out.prompt_tokens, "ttft_s":ticks.first(), "decode_s":decode,
                        "cache_before":before, "cache_after":metal.parity_cache(), "os_before":os_before,
                        "os_at_first":os_at_first, "os_after":os_after}));
'''+anchor)
    backend.write_text(code)
    # The stock example otherwise runs an intervening blocking generation.
    bench=repo/'crates/ruvllm/examples/lattice_bench.rs'
    code=bench.read_text()
    anchor='        results.push(stream_result);'
    bench.write_text(replace_once(code,anchor,anchor+'\n        if std::env::var_os("RUVLLM_PARITY_STREAM_ONLY").is_some() { continue; }'))
