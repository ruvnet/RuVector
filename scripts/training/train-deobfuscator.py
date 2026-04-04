#!/usr/bin/env python3
"""
Train a small character-level transformer for JS deobfuscation.

Input:  minified name + context tokens (strings + properties)
Output: predicted original name (character-level generation)

Model: 6M params, 3-layer transformer encoder, 4 heads, 128 embed dim.
Trains in ~2 hours on an NVIDIA L4 GPU.

Usage:
    python train-deobfuscator.py --data training-data.jsonl --output ./model
    python train-deobfuscator.py --data training-data.jsonl --output ./model --epochs 50 --batch-size 128
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256        # byte-level character vocabulary
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
MAX_CONTEXT = 64        # max context characters
MAX_NAME = 32           # max name characters (both minified and original)
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
FFN_DIM = 512

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DeobfuscationDataset(Dataset):
    """Load JSONL training data for deobfuscation."""

    def __init__(self, path: str):
        self.samples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        minified = sample["minified"]
        original = sample["original"]
        context_strings = sample.get("context_strings", [])
        properties = sample.get("properties", [])

        # Build context: join context_strings and properties with separators.
        context_text = " ".join(context_strings[:8]) + " | " + " ".join(properties[:8])

        # Encode to byte-level tokens.
        context_tokens = self._encode(context_text, MAX_CONTEXT)
        minified_tokens = self._encode(minified, MAX_NAME)
        original_tokens = self._encode_target(original, MAX_NAME)

        # Input: [context_tokens] + [minified_tokens]
        input_tokens = torch.cat([context_tokens, minified_tokens])

        return input_tokens, original_tokens

    @staticmethod
    def _encode(text: str, max_len: int) -> torch.Tensor:
        """Encode text to byte-level tensor, padded to max_len."""
        encoded = [min(b, VOCAB_SIZE - 1) for b in text.encode("utf-8")[:max_len]]
        padded = encoded + [PAD_TOKEN] * (max_len - len(encoded))
        return torch.tensor(padded, dtype=torch.long)

    @staticmethod
    def _encode_target(text: str, max_len: int) -> torch.Tensor:
        """Encode target with SOS/EOS markers."""
        encoded = [min(b, VOCAB_SIZE - 1) for b in text.encode("utf-8")[: max_len - 2]]
        tokens = [SOS_TOKEN] + encoded + [EOS_TOKEN]
        padded = tokens + [PAD_TOKEN] * (max_len - len(tokens))
        return torch.tensor(padded, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DeobfuscationModel(nn.Module):
    """Small transformer encoder for character-level name prediction."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        ffn_dim: int = FFN_DIM,
        max_context: int = MAX_CONTEXT,
        max_name: int = MAX_NAME,
    ):
        super().__init__()
        self.max_context = max_context
        self.max_name = max_name
        total_seq = max_context + max_name

        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
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
        self.output = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tokens: (batch, max_context + max_name) long tensor

        Returns:
            logits: (batch, max_name, vocab_size) predictions for original name
        """
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.char_embed(input_tokens) + self.pos_embed(positions)

        # Create padding mask.
        pad_mask = input_tokens == PAD_TOKEN

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.layer_norm(x)

        # Take the last max_name positions as the prediction window.
        name_out = x[:, -self.max_name :, :]
        logits = self.output(name_out)

        return logits

    def param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    data_path: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    val_split: float = 0.1,
    device_str: str = "auto",
):
    """Train the deobfuscation model."""

    # Device selection.
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # Load dataset.
    dataset = DeobfuscationDataset(data_path)
    total = len(dataset)
    val_size = max(1, int(total * val_split))
    train_size = total - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Model.
    model = DeobfuscationModel().to(device)
    print(f"Model parameters: {model.param_count():,}")

    # Loss and optimizer.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine annealing LR schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Output directory.
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for input_tokens, target_tokens in train_loader:
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)

            logits = model(input_tokens)  # (B, max_name, vocab)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_tokens.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * input_tokens.size(0)

            # Accuracy: non-pad positions.
            preds = logits.argmax(dim=-1)
            mask = target_tokens != PAD_TOKEN
            train_correct += (preds[mask] == target_tokens[mask]).sum().item()
            train_total += mask.sum().item()

        scheduler.step()
        avg_train_loss = train_loss / train_size
        train_acc = train_correct / max(train_total, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

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
                val_total += mask.sum().item()

        avg_val_loss = val_loss / val_size
        val_acc = val_correct / max(val_total, 1)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f} | "
            f"{elapsed:.1f}s"
        )

        # Save best model.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc,
                    "config": {
                        "vocab_size": VOCAB_SIZE,
                        "embed_dim": EMBED_DIM,
                        "num_heads": NUM_HEADS,
                        "num_layers": NUM_LAYERS,
                        "ffn_dim": FFN_DIM,
                        "max_context": MAX_CONTEXT,
                        "max_name": MAX_NAME,
                    },
                },
                checkpoint_path,
            )
            print(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")

    # Save final model.
    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Models saved to {output_dir}/")

    return model


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------


def export_onnx(model: nn.Module, output_dir: str):
    """Export trained model to ONNX format."""
    model.eval()
    model.cpu()

    dummy_input = torch.zeros(1, MAX_CONTEXT + MAX_NAME, dtype=torch.long)
    onnx_path = os.path.join(output_dir, "deobfuscator.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input_tokens"],
        output_names=["logits"],
        dynamic_axes={
            "input_tokens": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"Exported ONNX model to {onnx_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train JS deobfuscation model")
    parser.add_argument("--data", required=True, help="Path to training data JSONL")
    parser.add_argument("--output", default="./model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX after training")
    args = parser.parse_args()

    model = train(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device_str=args.device,
    )

    if args.export_onnx:
        export_onnx(model, args.output)


if __name__ == "__main__":
    main()
