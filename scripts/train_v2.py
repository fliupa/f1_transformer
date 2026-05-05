"""Train F1 Transformer V2 on 2014-2024 combined data.

V2 improvements:
- ELO ratings, momentum, track affinity features
- Context with mean+std+max stats
- Smaller model (128-dim) with candidate self-attention + gated cross-attention
- SWA, OneCycleLR, data augmentation, class weights
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pickle

from src.model.transformer_model_v2 import F1WinnerTransformerV2
from src.training.trainer_v2 import TrainerV2
from src.preprocessing.build_dataset_v2 import main as build_v2

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
    print("TRAINING V2: 2014-2024 COMBINED")
    print(f"Device: {DEVICE}, AMP: {USE_AMP}")
    print("=" * 60)

    # Build V2 dataset (if not already done)
    v2_train_path = PROCESSED / "features_train_v2.pt"
    if not v2_train_path.exists():
        print("\nBuilding V2 dataset...")
        build_v2()

    # Load V2 data
    train_data = torch.load(PROCESSED / "features_train_v2.pt", weights_only=False)
    val_data = torch.load(PROCESSED / "features_val_v2.pt", weights_only=False)
    with open(PROCESSED / "metadata_v2.pkl", "rb") as f:
        metadata = pickle.load(f)

    print(f"\nTrain: {len(train_data['winners'])} seqs, Val: {len(val_data['winners'])} seqs")
    print(f"Candidate dim: {metadata['d_candidate_raw']}, Context dim: {metadata['d_context_raw']}")

    # Create V2 model (smaller, regularized)
    model = F1WinnerTransformerV2(
        d_model=128,
        n_heads=4,
        n_encoder_layers=4,
        n_cross_attn_layers=3,
        d_ff=512,
        dropout=0.25,
        context_window=metadata["context_window"],
        num_drivers=metadata["num_drivers"],
        num_constructors=metadata["num_constructors"],
        num_circuits=metadata["num_circuits"],
        d_candidate_raw=metadata["d_candidate_raw"],
        d_context_raw=metadata["d_context_raw"],
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Model V2: {params:,} params (V1: 2,279,449)")

    # Train with V2 trainer (SWA, OneCycle, augmentation, class weights)
    trainer = TrainerV2(
        model=model,
        train_data=train_data,
        val_data=val_data,
        device=DEVICE,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=80,
        patience=15,
        use_amp=USE_AMP,
        checkpoint_dir=MODEL_DIR,
        num_workers=0 if str(DEVICE) == "mps" else 2,
        warmup_epochs=5,
        swa_start=20,
        label_smoothing=0.10,
        noise_std=0.02,
        context_dropout_prob=0.15,
        grad_accum_steps=2,
        use_class_weights=True,
    )

    history = trainer.train(early_stopping=True)
    print(f"\nBest val acc V2: {trainer.best_val_acc:.3f}")
    print(f"Model saved to {MODEL_DIR / 'best_v2.pt'}")


if __name__ == "__main__":
    main()
