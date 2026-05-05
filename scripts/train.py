"""Train F1 winner prediction transformer on 2014-2024 combined data.

Loads preprocessed features, creates model, trains with early stopping,
and saves best checkpoint to models/final/best.pt.

Training: 192 seqs (2014-2024 combined), 33 val (last 15%)
Target: ~54.5% validation accuracy
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pickle

from src.model.transformer_model import F1WinnerTransformer
from src.training.trainer import Trainer

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
USE_AMP = torch.cuda.is_available()
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "final"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRAINING: 2014-2024 COMBINED")
    print(f"Device: {DEVICE}, AMP: {USE_AMP}")
    print("=" * 60)

    # Load data
    train_data = torch.load(PROCESSED / "features_train.pt", weights_only=False)
    val_data = torch.load(PROCESSED / "features_val.pt", weights_only=False)
    with open(PROCESSED / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    print(f"Train: {len(train_data['winners'])} seqs, Val: {len(val_data['winners'])} seqs")

    # Create model
    model = F1WinnerTransformer(
        d_model=192, n_heads=6, n_encoder_layers=3, n_cross_attn_layers=2,
        d_ff=768, dropout=0.15,
        context_window=metadata["context_window"],
        num_drivers=metadata["num_drivers"],
        num_constructors=metadata["num_constructors"],
        num_circuits=metadata["num_circuits"],
        d_candidate_raw=metadata["d_candidate_raw"],
        d_context_raw=metadata["d_context_raw"],
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params")

    # Train
    trainer = Trainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        device=DEVICE,
        batch_size=32,
        learning_rate=5e-4,
        weight_decay=1e-5,
        epochs=80,
        patience=15,
        use_amp=USE_AMP,
        checkpoint_dir=MODEL_DIR,
        num_workers=0 if str(DEVICE) == "mps" else 2,
    )

    history = trainer.train(early_stopping=True)
    print(f"\nBest val acc: {trainer.best_val_acc:.3f}")
    print(f"Model saved to {MODEL_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
