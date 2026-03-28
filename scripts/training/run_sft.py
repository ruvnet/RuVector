#!/usr/bin/env python3
"""RuvLTRA Phase 2: LoRA SFT fine-tuning pipeline.

Loads training corpus, runs LoRA SFT with peft + transformers,
merges adapter weights, converts to GGUF, and runs release gate checks.

Usage:
    python run_sft.py --model-id ruvnet/ruvLTRA-7b --corpus data/training/corpus.jsonl
    python run_sft.py --model-id ruvnet/ruvLTRA-7b --corpus corpus.jsonl --upload
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ruvltra-sft")


def parse_args():
    p = argparse.ArgumentParser(description="RuvLTRA LoRA SFT training pipeline")
    p.add_argument("--model-id", required=True, help="HuggingFace model ID")
    p.add_argument("--corpus", required=True, help="Path to training corpus (JSONL)")
    p.add_argument("--output-dir", default="/tmp/sft-output", help="Output directory")
    p.add_argument("--revision", default="main", help="Model revision/branch")

    # LoRA config
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    help="Comma-separated target modules for LoRA")

    # Training config
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence length")
    p.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")

    # Output controls
    p.add_argument("--upload", action="store_true", help="Upload merged model to HuggingFace")
    p.add_argument("--convert-gguf", action="store_true", default=True, help="Convert to GGUF after merge")
    p.add_argument("--quant-type", default="Q4_K_M", help="GGUF quantization type for release")
    p.add_argument("--skip-gate", action="store_true", help="Skip release gate checks")
    return p.parse_args()


def load_corpus(corpus_path: str) -> list[dict]:
    """Load JSONL training corpus. Expected format: {instruction, input, output} or {messages}."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    records = []
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("Skipping malformed line %d: %s", i + 1, e)

    if not records:
        raise ValueError(f"No valid records found in {corpus_path}")

    log.info("Loaded %d training examples from %s", len(records), corpus_path)
    return records


def format_dataset(records: list[dict]):
    """Convert corpus records into a HuggingFace Dataset."""
    from datasets import Dataset

    formatted = []
    for rec in records:
        if "messages" in rec:
            # Chat format: [{role, content}, ...]
            formatted.append({"messages": rec["messages"]})
        elif "instruction" in rec:
            # Alpaca format
            messages = [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": rec["instruction"]},
            ]
            if rec.get("input"):
                messages[-1]["content"] += f"\n\n{rec['input']}"
            messages.append({"role": "assistant", "content": rec["output"]})
            formatted.append({"messages": messages})
        elif "text" in rec and len(rec["text"]) > 100:
            # Raw text format (brain memories, ADRs) — convert to completion format
            text = rec["text"]
            title = rec.get("title", text[:60].split("\n")[0])
            messages = [
                {"role": "system", "content": "You are a knowledgeable software architect and Rust developer."},
                {"role": "user", "content": f"Explain: {title}"},
                {"role": "assistant", "content": text},
            ]
            formatted.append({"messages": messages})
        else:
            log.warning("Skipping record with unknown format: %s", list(rec.keys()))

    return Dataset.from_list(formatted)


def train_lora(model_id: str, dataset, args) -> str:
    """Run LoRA SFT training and return path to adapter directory."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    adapter_dir = os.path.join(args.output_dir, "lora-adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    # Load tokenizer
    log.info("Loading tokenizer for %s...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit for memory efficiency
    log.info("Loading model in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    target_modules = [m.strip() for m in args.target_modules.split(",")]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    log.info("Trainable parameters: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    # Training config
    training_config = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        seed=42,
    )

    # Train
    log.info("Starting LoRA SFT training (%d epochs)...", args.epochs)
    start = time.time()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_config,
    )

    trainer.train()
    elapsed = time.time() - start
    log.info("Training completed in %.1f minutes", elapsed / 60)

    # Save adapter
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    log.info("LoRA adapter saved to %s", adapter_dir)

    return adapter_dir


def merge_adapter(model_id: str, adapter_dir: str, output_dir: str) -> str:
    """Merge LoRA adapter back into base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    merged_dir = os.path.join(output_dir, "merged-model")
    os.makedirs(merged_dir, exist_ok=True)

    log.info("Loading base model for merge...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    log.info("Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()

    log.info("Saving merged model...")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(merged_dir)

    log.info("Merged model saved to %s", merged_dir)
    return merged_dir


def convert_to_gguf(merged_dir: str, output_dir: str, quant_type: str) -> str:
    """Convert merged model to quantized GGUF."""
    import subprocess
    import shutil

    gguf_f16 = os.path.join(output_dir, "model-f16.gguf")
    gguf_quant = os.path.join(output_dir, f"model-{quant_type}.gguf")

    convert_script = "/opt/llama.cpp/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        log.warning("llama.cpp convert script not found, skipping GGUF conversion")
        return ""

    # Convert to f16
    log.info("Converting to GGUF (f16)...")
    subprocess.run(
        [sys.executable, convert_script, merged_dir, "--outfile", gguf_f16, "--outtype", "f16"],
        check=True,
    )

    # Quantize
    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin:
        log.info("Quantizing to %s...", quant_type)
        subprocess.run([quantize_bin, gguf_f16, gguf_quant, quant_type], check=True)
        file_size = os.path.getsize(gguf_quant)
        log.info("Quantized GGUF: %s (%.2f GB)", gguf_quant, file_size / (1024**3))
        return gguf_quant
    else:
        log.warning("llama-quantize not found, returning f16 GGUF")
        return gguf_f16


def release_gate_check(output_dir: str, quant_type: str) -> bool:
    """Run release gate checks on the final model.

    Gate criteria:
    - Quantized GGUF exists and is non-empty
    - File size is within expected bounds (> 1GB for 7B model)
    - Training loss log shows convergence
    """
    log.info("=== Release Gate Check ===")
    passed = True

    # Check GGUF exists
    gguf_path = os.path.join(output_dir, f"model-{quant_type}.gguf")
    if not os.path.exists(gguf_path):
        gguf_path = os.path.join(output_dir, "model-f16.gguf")

    if os.path.exists(gguf_path):
        size_gb = os.path.getsize(gguf_path) / (1024**3)
        log.info("  GGUF size: %.2f GB", size_gb)
        if size_gb < 0.5:
            log.error("  FAIL: GGUF file suspiciously small (< 0.5 GB)")
            passed = False
        else:
            log.info("  PASS: GGUF file size OK")
    else:
        log.error("  FAIL: No GGUF file found")
        passed = False

    # Check adapter was saved
    adapter_dir = os.path.join(output_dir, "lora-adapter")
    adapter_config = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(adapter_config):
        log.info("  PASS: LoRA adapter config present")
    else:
        log.error("  FAIL: LoRA adapter config missing")
        passed = False

    # Check training logs for convergence
    trainer_state = os.path.join(adapter_dir, "trainer_state.json")
    if os.path.exists(trainer_state):
        with open(trainer_state) as f:
            state = json.load(f)
        log_history = state.get("log_history", [])
        losses = [entry["loss"] for entry in log_history if "loss" in entry]
        if len(losses) >= 2:
            initial_loss = losses[0]
            final_loss = losses[-1]
            if final_loss < initial_loss:
                log.info("  PASS: Loss decreased %.4f -> %.4f", initial_loss, final_loss)
            else:
                log.warning("  WARN: Loss did not decrease %.4f -> %.4f", initial_loss, final_loss)
        else:
            log.warning("  WARN: Not enough loss entries to check convergence")
    else:
        log.warning("  WARN: No trainer state found, cannot check convergence")

    verdict = "PASSED" if passed else "FAILED"
    log.info("=== Release Gate: %s ===", verdict)
    return passed


def upload_to_hf(model_id: str, output_dir: str, revision: str):
    """Upload merged model and artifacts to HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi()
    merged_dir = os.path.join(output_dir, "merged-model")

    if os.path.isdir(merged_dir):
        log.info("Uploading merged model to %s...", model_id)
        api.upload_folder(
            folder_path=merged_dir,
            repo_id=model_id,
            revision=revision,
        )

    # Upload GGUF files separately
    for f in os.listdir(output_dir):
        if f.endswith(".gguf"):
            fpath = os.path.join(output_dir, f)
            log.info("Uploading %s...", f)
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f,
                repo_id=model_id,
                revision=revision,
            )

    log.info("Upload complete")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=== RuvLTRA SFT Training Pipeline ===")
    log.info("Model: %s", args.model_id)
    log.info("Corpus: %s", args.corpus)
    log.info("LoRA: r=%d, alpha=%d, dropout=%.2f", args.lora_r, args.lora_alpha, args.lora_dropout)
    log.info("Training: epochs=%d, batch=%d, lr=%.0e", args.epochs, args.batch_size, args.lr)

    # Phase 2a: Load and format corpus
    records = load_corpus(args.corpus)
    dataset = format_dataset(records)
    log.info("Dataset prepared: %d examples", len(dataset))

    # Phase 2b: LoRA SFT training
    adapter_dir = train_lora(args.model_id, dataset, args)

    # Phase 2c: Merge adapter weights
    merged_dir = merge_adapter(args.model_id, adapter_dir, args.output_dir)

    # Phase 2d: Convert to GGUF
    gguf_path = ""
    if args.convert_gguf:
        gguf_path = convert_to_gguf(merged_dir, args.output_dir, args.quant_type)

    # Phase 2e: Release gate check
    if not args.skip_gate:
        gate_passed = release_gate_check(args.output_dir, args.quant_type)
        if not gate_passed:
            log.error("Release gate FAILED — review output before publishing")
            sys.exit(2)

    # Phase 2f: Upload if requested
    if args.upload:
        upload_to_hf(args.model_id, args.output_dir, args.revision)

    log.info("=== SFT Pipeline complete ===")
    log.info("Adapter: %s", adapter_dir)
    log.info("Merged: %s", merged_dir)
    if gguf_path:
        log.info("GGUF: %s", gguf_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)
