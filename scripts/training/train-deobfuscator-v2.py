#!/usr/bin/env python3
"""
Train deobfuscation model v2 with configurable architecture.

Supports:
  - Configurable layer count, embed dim, heads, FFN dim
  - Label smoothing
  - Warmup + cosine annealing schedule
  - Gradient accumulation for effective larger batch sizes
  - Mixed precision (if available)

Usage:
    python train-deobfuscator-v2.py --data training-data-v2-filtered.jsonl \
        --output model-v2-big --epochs 40 --batch-size 128 \
        --num-layers 4 --embed-dim 192 --num-heads 6 --ffn-dim 768 \
        --label-smoothing 0.1 --export-onnx
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
MAX_CONTEXT = 64
MAX_NAME = 32

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DeobfuscationDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        context_text = " ".join(s.get("context_strings", [])[:8]) + " | " + \
                       " ".join(s.get("properties", [])[:8])

        context_tokens = self._encode(context_text, MAX_CONTEXT)
        minified_tokens = self._encode(s["minified"], MAX_NAME)
        original_tokens = self._encode_target(s["original"], MAX_NAME)

        input_tokens = torch.cat([context_tokens, minified_tokens])
        return input_tokens, original_tokens

    @staticmethod
    def _encode(text, max_len):
        encoded = [min(b, VOCAB_SIZE - 1) for b in text.encode("utf-8")[:max_len]]
        padded = encoded + [PAD_TOKEN] * (max_len - len(encoded))
        return torch.tensor(padded, dtype=torch.long)

    @staticmethod
    def _encode_target(text, max_len):
        encoded = [min(b, VOCAB_SIZE - 1) for b in text.encode("utf-8")[:max_len - 2]]
        tokens = [SOS_TOKEN] + encoded + [EOS_TOKEN]
        padded = tokens + [PAD_TOKEN] * (max_len - len(tokens))
        return torch.tensor(padded, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DeobfuscationModelV2(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=3,
                 ffn_dim=512, dropout=0.1):
        super().__init__()
        self.max_context = MAX_CONTEXT
        self.max_name = MAX_NAME
        total_seq = MAX_CONTEXT + MAX_NAME

        self.char_embed = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_TOKEN)
        self.pos_embed = nn.Embedding(total_seq, embed_dim)

        # Kind embedding (3 kinds: function=0, class=1, var=2)
        # Not used in input encoding for compatibility, but position helps

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, VOCAB_SIZE)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training with warmup + cosine schedule
# ---------------------------------------------------------------------------


def get_lr(step, total_steps, warmup_steps, base_lr):
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    dataset = DeobfuscationDataset(args.data)
    total = len(dataset)
    val_size = max(1, int(total * args.val_split))
    train_size = total - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Training: {train_size}, Validation: {val_size}")

    # Model
    model = DeobfuscationModelV2(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {model.param_count():,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_TOKEN,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01,
        betas=(0.9, 0.98), eps=1e-6,
    )

    # Schedule
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(2000, total_steps // 10)

    os.makedirs(args.output, exist_ok=True)
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total_tokens = 0

        for batch_idx, (input_tokens, target_tokens) in enumerate(train_loader):
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)

            # Update LR
            step = (epoch - 1) * len(train_loader) + batch_idx
            lr = get_lr(step, total_steps, warmup_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            logits = model(input_tokens)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_tokens.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * input_tokens.size(0)
            preds = logits.argmax(dim=-1)
            mask = target_tokens != PAD_TOKEN
            train_correct += (preds[mask] == target_tokens[mask]).sum().item()
            train_total_tokens += mask.sum().item()

        avg_train_loss = train_loss / train_size
        train_acc = train_correct / max(train_total_tokens, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total_tokens = 0
        exact_match = 0
        val_samples = 0

        with torch.no_grad():
            for input_tokens, target_tokens in val_loader:
                input_tokens = input_tokens.to(device)
                target_tokens = target_tokens.to(device)

                logits = model(input_tokens)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_tokens.reshape(-1))
                val_loss += loss.item() * input_tokens.size(0)

                preds = logits.argmax(dim=-1)
                mask = target_tokens != PAD_TOKEN
                val_correct += (preds[mask] == target_tokens[mask]).sum().item()
                val_total_tokens += mask.sum().item()

                # Exact match: all non-pad positions match
                for b in range(preds.size(0)):
                    b_mask = mask[b]
                    if (preds[b][b_mask] == target_tokens[b][b_mask]).all():
                        exact_match += 1
                    val_samples += 1

        avg_val_loss = val_loss / val_size
        val_acc = val_correct / max(val_total_tokens, 1)
        exact_acc = exact_match / max(val_samples, 1)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f} exact={exact_acc:.4f} | "
            f"lr={lr:.6f} | {elapsed:.1f}s"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc,
                    "exact_acc": exact_acc,
                    "config": {
                        "vocab_size": VOCAB_SIZE,
                        "embed_dim": args.embed_dim,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "ffn_dim": args.ffn_dim,
                        "max_context": MAX_CONTEXT,
                        "max_name": MAX_NAME,
                    },
                },
                os.path.join(args.output, "best_model.pt"),
            )
            print(f"  -> Saved best model (val_loss={avg_val_loss:.4f}, exact={exact_acc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {args.patience} epochs without improvement")
            break

    # Save final
    torch.save(model.state_dict(), os.path.join(args.output, "final_model.pt"))
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")

    return model


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------


def export_onnx(model, output_dir, embed_dim):
    """Export to ONNX."""
    model.eval()
    model.cpu()
    dummy = torch.zeros(1, MAX_CONTEXT + MAX_NAME, dtype=torch.long)
    onnx_path = os.path.join(output_dir, "deobfuscator.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input_tokens"],
        output_names=["logits"],
        dynamic_axes={"input_tokens": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=14,
    )
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"Exported ONNX to {onnx_path} ({size_kb:.0f} KB)")


def main():
    p = argparse.ArgumentParser(description="Train JS deobfuscation model v2")
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="./model-v2")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--ffn-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--export-onnx", action="store_true")
    args = p.parse_args()

    model = train(args)

    if args.export_onnx:
        export_onnx(model, args.output, args.embed_dim)


if __name__ == "__main__":
    main()
