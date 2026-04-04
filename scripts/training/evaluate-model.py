#!/usr/bin/env python3
"""
Evaluate a trained deobfuscation model on held-out test data.

Metrics:
  - Exact match accuracy (full name correct)
  - Prefix match accuracy (first N chars correct)
  - Character-level accuracy (per-character correct rate)
  - Top-K accuracy (correct answer in top K predictions)

Usage:
    python evaluate-model.py --model model-v2/best_model.pt --test training-data-v2.jsonl
    python evaluate-model.py --model model-v2/best_model.pt --test training-data-v2.jsonl --split 0.1
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn

# Import model definition from training script
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

# Inline the constants and model to avoid import issues
VOCAB_SIZE = 256
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
MAX_CONTEXT = 64
MAX_NAME = 32
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
FFN_DIM = 512


class DeobfuscationModel(nn.Module):
    """Mirror of the training model for inference. Supports configurable arch."""

    def __init__(self, embed_dim=128, num_heads=4, num_layers=3, ffn_dim=512):
        super().__init__()
        total_seq = MAX_CONTEXT + MAX_NAME
        self.max_context = MAX_CONTEXT
        self.max_name = MAX_NAME

        self.char_embed = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_TOKEN)
        self.pos_embed = nn.Embedding(total_seq, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
        # Support both old ('output') and new ('output_proj') naming
        self.output_proj = nn.Linear(embed_dim, VOCAB_SIZE)

    def forward(self, input_tokens):
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.char_embed(input_tokens) + self.pos_embed(positions)
        pad_mask = input_tokens == PAD_TOKEN
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.layer_norm(x)
        name_out = x[:, -self.max_name:, :]
        logits = self.output_proj(name_out)
        return logits


def encode(text, max_len):
    """Encode text to byte-level tensor."""
    encoded = [min(b, VOCAB_SIZE - 1) for b in text.encode("utf-8")[:max_len]]
    padded = encoded + [PAD_TOKEN] * (max_len - len(encoded))
    return torch.tensor(padded, dtype=torch.long)


def encode_target(text, max_len):
    """Encode target with SOS/EOS."""
    encoded = [min(b, VOCAB_SIZE - 1) for b in text.encode("utf-8")[:max_len - 2]]
    tokens = [SOS_TOKEN] + encoded + [EOS_TOKEN]
    padded = tokens + [PAD_TOKEN] * (max_len - len(tokens))
    return torch.tensor(padded, dtype=torch.long)


def decode_tokens(tokens):
    """Decode byte-level tokens back to string."""
    chars = []
    for t in tokens:
        t = t.item() if hasattr(t, "item") else t
        if t == PAD_TOKEN or t == EOS_TOKEN:
            break
        if t == SOS_TOKEN:
            continue
        if 32 <= t < 127:
            chars.append(chr(t))
    return "".join(chars)


def prepare_input(sample):
    """Prepare a single sample for inference."""
    minified = sample["minified"]
    context_strings = sample.get("context_strings", [])
    properties = sample.get("properties", [])

    context_text = " ".join(context_strings[:8]) + " | " + " ".join(properties[:8])
    context_tokens = encode(context_text, MAX_CONTEXT)
    minified_tokens = encode(minified, MAX_NAME)
    input_tokens = torch.cat([context_tokens, minified_tokens])
    return input_tokens


def evaluate(model_path, test_path, split_ratio=0.1, top_k=5, device_str="cpu"):
    """Run evaluation."""
    device = torch.device(device_str)

    # Load model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Read architecture config from checkpoint if available
    config = checkpoint.get("config", {})
    embed_dim = config.get("embed_dim", EMBED_DIM)
    num_heads = config.get("num_heads", NUM_HEADS)
    num_layers = config.get("num_layers", NUM_LAYERS)
    ffn_dim = config.get("ffn_dim", FFN_DIM)

    model = DeobfuscationModel(embed_dim, num_heads, num_layers, ffn_dim).to(device)

    if "model_state_dict" in checkpoint:
        # Handle old checkpoints that use 'output' instead of 'output_proj'
        state_dict = checkpoint["model_state_dict"]
        if "output.weight" in state_dict and "output_proj.weight" not in state_dict:
            state_dict["output_proj.weight"] = state_dict.pop("output.weight")
            state_dict["output_proj.bias"] = state_dict.pop("output.bias")
        model.load_state_dict(state_dict)
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")
        print(f"  Checkpoint val_loss: {checkpoint.get('val_loss', '?'):.4f}")
        print(f"  Checkpoint val_acc: {checkpoint.get('val_acc', '?'):.4f}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Load test data
    print(f"\nLoading test data from {test_path}")
    samples = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Use last N% as test set (same split as training)
    total = len(samples)
    test_size = max(100, int(total * split_ratio))
    test_samples = samples[-test_size:]
    print(f"  Total samples: {total}, test samples: {test_size}")

    # Evaluate
    exact_match = 0
    prefix_3_match = 0
    prefix_5_match = 0
    char_correct = 0
    char_total = 0
    top_k_match = 0

    kind_correct = defaultdict(int)
    kind_total = defaultdict(int)
    failures = []

    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            original = sample["original"]
            kind = sample.get("kind", "var")

            input_tokens = prepare_input(sample).unsqueeze(0).to(device)
            logits = model(input_tokens)  # (1, MAX_NAME, VOCAB_SIZE)

            # Greedy decode
            preds = logits.argmax(dim=-1)[0]  # (MAX_NAME,)
            predicted = decode_tokens(preds)

            # Exact match
            kind_total[kind] += 1
            if predicted == original:
                exact_match += 1
                kind_correct[kind] += 1
            else:
                failures.append({
                    "minified": sample["minified"],
                    "original": original,
                    "predicted": predicted,
                    "kind": kind,
                    "context": sample.get("context_strings", [])[:3],
                })

            # Prefix matches
            if predicted[:3] == original[:3]:
                prefix_3_match += 1
            if predicted[:5] == original[:5]:
                prefix_5_match += 1

            # Character-level accuracy
            for j in range(min(len(predicted), len(original))):
                if predicted[j] == original[j]:
                    char_correct += 1
                char_total += 1
            # Count missing or extra chars as errors
            char_total += abs(len(predicted) - len(original))

            # Top-K: check if correct name is in top K at each position
            target_tokens = encode_target(original, MAX_NAME)
            match_in_topk = True
            for j in range(min(len(original) + 2, MAX_NAME)):
                if target_tokens[j] == PAD_TOKEN:
                    break
                topk_vals = torch.topk(logits[0, j], top_k).indices
                if target_tokens[j] not in topk_vals:
                    match_in_topk = False
                    break
            if match_in_topk:
                top_k_match += 1

    # Print results
    n = len(test_samples)
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Test samples:         {n}")
    print(f"  Exact match:          {exact_match}/{n} = {100*exact_match/n:.1f}%")
    print(f"  Prefix-3 match:       {prefix_3_match}/{n} = {100*prefix_3_match/n:.1f}%")
    print(f"  Prefix-5 match:       {prefix_5_match}/{n} = {100*prefix_5_match/n:.1f}%")
    print(f"  Char-level accuracy:  {100*char_correct/max(char_total,1):.1f}%")
    print(f"  Top-{top_k} accuracy:       {top_k_match}/{n} = {100*top_k_match/n:.1f}%")

    print("\nAccuracy by kind:")
    for kind in sorted(kind_total.keys()):
        total_k = kind_total[kind]
        correct_k = kind_correct[kind]
        print(f"  {kind:10s}: {correct_k}/{total_k} = {100*correct_k/max(total_k,1):.1f}%")

    # Show some failures
    print(f"\nSample failures (showing first 15):")
    for f in failures[:15]:
        ctx_str = ", ".join(f["context"][:3]) if f["context"] else "none"
        print(f"  {f['minified']:8s} -> predicted '{f['predicted']:20s}' "
              f"expected '{f['original']:20s}' [{f['kind']}] ctx=[{ctx_str}]")

    # Failure analysis
    print("\nFailure analysis:")
    len_buckets = defaultdict(lambda: [0, 0])  # [correct, total]
    for sample in test_samples:
        orig_len = len(sample["original"])
        bucket = "short(1-5)" if orig_len <= 5 else "medium(6-12)" if orig_len <= 12 else "long(13+)"
        input_tokens = prepare_input(sample).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tokens)
        predicted = decode_tokens(logits.argmax(dim=-1)[0])
        len_buckets[bucket][1] += 1
        if predicted == sample["original"]:
            len_buckets[bucket][0] += 1

    for bucket in ["short(1-5)", "medium(6-12)", "long(13+)"]:
        if bucket in len_buckets:
            c, t = len_buckets[bucket]
            print(f"  {bucket:15s}: {c}/{t} = {100*c/max(t,1):.1f}%")

    return exact_match / n


def main():
    parser = argparse.ArgumentParser(description="Evaluate deobfuscation model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--test", required=True, help="Path to test data JSONL")
    parser.add_argument("--split", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K accuracy K")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()

    accuracy = evaluate(
        model_path=args.model,
        test_path=args.test,
        split_ratio=args.split,
        top_k=args.top_k,
        device_str=args.device,
    )

    # Exit code: 0 if accuracy >= 80%, 1 otherwise
    sys.exit(0 if accuracy >= 0.80 else 1)


if __name__ == "__main__":
    main()
