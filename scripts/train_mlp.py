"""Train a compact MLP on V2 features for F1 winner prediction.

The transformer is too parameter-heavy for ~192 training sequences.
This MLP uses the same rich V2 features but with far fewer parameters
and stronger regularization, which should generalize better.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import numpy as np
import pickle
from tqdm import tqdm
import time
import copy
from typing import Optional

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
USE_AMP = torch.cuda.is_available()
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "final"


class F1WinnerMLP(nn.Module):
    """Compact MLP for F1 winner prediction.

    Strategy: flatten the full context + global candidates info,
    use a small MLP with heavy dropout to predict winner among 20 drivers.
    """

    def __init__(
        self,
        d_context_raw: int,
        context_window: int,
        d_candidate_raw: int,
        num_drivers: int,
        num_constructors: int,
        num_circuits: int,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.4,
    ):
        super().__init__()
        self.context_window = context_window
        self.d_candidate_raw = d_candidate_raw

        # Embeddings for categorical features
        self.driver_embed = nn.Embedding(num_drivers + 1, 16, padding_idx=0)
        self.constructor_embed = nn.Embedding(num_constructors + 1, 8, padding_idx=0)
        self.circuit_embed = nn.Embedding(num_circuits + 1, 8, padding_idx=0)

        # Input: context_flat + global_candidate_stats + per-candidate features
        # Context: context_window * d_context_raw
        # Global candidate: mean+std of numeric features (d_candidate_raw - 3)*2
        # Per-candidate: [driver_emb(16) + constructor_emb(8) + circuit_emb(8) + numeric(35)] * 20
        self.ctx_dim = context_window * d_context_raw
        self.global_cand_dim = (d_candidate_raw - 3) * 2  # mean + std

        # Process candidates through a shared network
        self.candidate_shared = nn.Sequential(
            nn.Linear(16 + 8 + 8 + (d_candidate_raw - 3), 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Aggregate all candidates: mean + max pool
        self.cand_pool_dim = 64 * 2  # mean + max

        # Total input to prediction MLP
        input_dim = self.ctx_dim + self.global_cand_dim + self.cand_pool_dim

        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 20))  # 20 drivers
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, nonlinearity='relu')

    def forward(self, context, candidates, time_gaps=None):
        """
        Args:
            context: (batch, context_window, d_context_raw)
            candidates: (batch, 20, d_candidate_raw)
        Returns:
            logits: (batch, 20)
        """
        batch_size = candidates.shape[0]

        # 1. Flatten context
        ctx_flat = context.reshape(batch_size, -1)  # (B, context_window * d_context_raw)

        # 2. Global candidate stats (mean+std of numeric features)
        cand_numeric = candidates[:, :, 3:]  # (B, 20, d_candidate_raw-3)
        cand_mean = cand_numeric.mean(dim=1)  # (B, d_candidate_raw-3)
        cand_std = cand_numeric.std(dim=1)    # (B, d_candidate_raw-3)
        global_feats = torch.cat([cand_mean, cand_std], dim=-1)

        # 3. Per-candidate embeddings
        driver_idx = candidates[:, :, 0].long()
        constructor_idx = candidates[:, :, 1].long()
        circuit_idx = candidates[:, :, 2].long()

        d_emb = self.driver_embed(driver_idx)       # (B, 20, 16)
        c_emb = self.constructor_embed(constructor_idx)  # (B, 20, 8)
        circ_emb = self.circuit_embed(circuit_idx)   # (B, 20, 8)

        cand_feats = torch.cat([
            d_emb, c_emb, circ_emb, cand_numeric
        ], dim=-1)  # (B, 20, 16+8+8+35)

        # Process each candidate through shared network
        cand_processed = self.candidate_shared(cand_feats)  # (B, 20, 64)

        # Pool across candidates
        cand_mean_pool = cand_processed.mean(dim=1)  # (B, 64)
        cand_max_pool = cand_processed.max(dim=1)[0]  # (B, 64)
        cand_pooled = torch.cat([cand_mean_pool, cand_max_pool], dim=-1)  # (B, 128)

        # 4. Concatenate all features
        combined = torch.cat([ctx_flat, global_feats, cand_pooled], dim=-1)

        # 5. MLP prediction
        logits = self.mlp(combined)  # (B, 20)

        return logits


def load_data():
    """Load V2 training data (all years 2014-2024 for max data)."""
    train = torch.load(PROCESSED / "features_train_v2.pt", weights_only=False)
    val = torch.load(PROCESSED / "features_val_v2.pt", weights_only=False)
    with open(PROCESSED / "metadata_v2.pkl", "rb") as f:
        metadata = pickle.load(f)
    return train, val, metadata


def train_epoch(model, loader, optimizer, scaler, criterion, use_amp):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for context, candidates, winners, time_gaps in pbar:
        context = context.to(DEVICE)
        candidates = candidates.to(DEVICE)
        winners = winners.to(DEVICE)

        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp):
            logits = model(context, candidates)
            loss = criterion(logits, winners)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    top3_correct = 0

    for context, candidates, winners, time_gaps in loader:
        context = context.to(DEVICE)
        candidates = candidates.to(DEVICE)
        winners = winners.to(DEVICE)

        with autocast('cuda', enabled=USE_AMP):
            logits = model(context, candidates)
            loss = criterion(logits, winners)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == winners).sum().item()
        total += len(winners)

        # Top-3
        _, top3 = torch.topk(logits, 3, dim=-1)
        top3_correct += sum(w in top3[i] for i, w in enumerate(winners))

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": correct / total,
        "top3_accuracy": top3_correct / total,
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRAINING MLP ON V2 FEATURES")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    train_data, val_data, metadata = load_data()
    print(f"Train: {len(train_data['winners'])} seqs, "
          f"Val: {len(val_data['winners'])} seqs")

    # Create model
    model = F1WinnerMLP(
        d_context_raw=metadata["d_context_raw"],
        context_window=metadata["context_window"],
        d_candidate_raw=metadata["d_candidate_raw"],
        num_drivers=metadata["num_drivers"],
        num_constructors=metadata["num_constructors"],
        num_circuits=metadata["num_circuits"],
        hidden_dims=[256, 128, 64],
        dropout=0.4,
    ).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model MLP: {params:,} params")

    # Data loaders
    train_dataset = TensorDataset(
        train_data["context"], train_data["candidates"],
        train_data["winners"], train_data.get("time_gaps",
            torch.zeros(len(train_data["winners"]), metadata["context_window"])),
    )
    val_dataset = TensorDataset(
        val_data["context"], val_data["candidates"],
        val_data["winners"], val_data.get("time_gaps",
            torch.zeros(len(val_data["winners"]), metadata["context_window"])),
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Class weights (sqrt inverse freq for stronger minority weighting)
    winner_counts = torch.bincount(train_data["winners"], minlength=20).float()
    weights = 1.0 / torch.sqrt(winner_counts + 1)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE), label_smoothing=0.15)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=USE_AMP)

    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    best_state = None

    print("\nTraining...")
    for epoch in range(80):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, criterion, USE_AMP)
        val_metrics = validate(model, val_loader, criterion)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}/80 | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.3f} | "
              f"Top3: {val_metrics['top3_accuracy']:.3f} | "
              f"LR: {lr:.2e}")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  >>> New best!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_val_acc": best_val_acc,
    }, MODEL_DIR / "best_mlp.pt")

    # Final eval
    final_metrics = validate(model, val_loader, criterion)
    print(f"\nBest Val acc: {best_val_acc:.3f}")
    print(f"Final Top3: {final_metrics['top3_accuracy']:.3f}")
    print(f"Model saved to {MODEL_DIR / 'best_mlp.pt'}")


if __name__ == "__main__":
    main()
