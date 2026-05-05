"""Train F1 Transformer V2 on 2022-2024 data (post-regulation change era).

F1 competitive order changed dramatically with 2022 regulations.
Training on pre-2022 data introduces bias toward Hamilton/Mercedes era.
This variant trains only on the current era for better 2025 generalization.

Also uses a smaller model (d_model=96) and stronger regularization
to handle the smaller dataset (~57 train, ~10 val).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pickle
import numpy as np

from src.model.transformer_model_v2 import F1WinnerTransformerV2
from src.training.trainer_v2 import TrainerV2
from src.preprocessing.build_dataset_v2 import (
    normalize_features_v2, PROCESSED_DATA, RAW_DATA,
)
from src.preprocessing.feature_engineering_v2 import (
    build_all_features_v2, load_race_data, load_circuit_data, load_weather_data,
    NUMERIC_DRIVER_COLS_V2, CONSTRUCTOR_COLS_V2, CIRCUIT_COLS_V2,
    WEATHER_COLS_V2, RACE_CONTEXT_COLS_V2, get_feature_vector_size_v2,
)
from src.preprocessing.encoders import build_encoders, save_encoders
from src.preprocessing.sequence_builder_v2 import SequenceBuilderV2, FeatureConfigV2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
USE_AMP = torch.cuda.is_available()
MODEL_DIR = PROJECT_ROOT / "models" / "final"

TRAIN_START, TRAIN_END = 2022, 2024


def build_recent_dataset():
    """Build dataset using only 2022-2024 races."""
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    df = load_race_data()
    circuits_df = load_circuit_data()
    weather_df = load_weather_data()

    featured_df = build_all_features_v2(df, circuits_df, weather_df)

    driver_enc, constructor_enc, circuit_enc = build_encoders(featured_df)
    save_encoders(driver_enc, constructor_enc, circuit_enc, PROCESSED_DATA)

    config = FeatureConfigV2(context_window=10)
    builder = SequenceBuilderV2(
        featured_df, driver_enc, constructor_enc, circuit_enc, config,
        numeric_driver_cols=NUMERIC_DRIVER_COLS_V2,
        constructor_cols=CONSTRUCTOR_COLS_V2,
        circuit_cols=CIRCUIT_COLS_V2,
        weather_cols=WEATHER_COLS_V2,
        race_context_cols=RACE_CONTEXT_COLS_V2,
    )

    # Build sequences for 2022-2024 training
    ctx_all, cand_all, win_all, gap_all, meta_all = builder.build_sequences(
        TRAIN_START, TRAIN_END
    )

    # Build 2025 prediction sequences
    has_2025 = not featured_df[featured_df["year"] == 2025].empty
    if has_2025:
        ctx_2025, cand_2025, win_2025, gap_2025, meta_2025 = builder.build_sequences(
            2025, 2025
        )
    else:
        # Dummy
        ctx_2025 = np.zeros((0, 10, 1))
        cand_2025 = np.zeros((0, 20, 1))
        win_2025 = np.zeros((0,))
        gap_2025 = np.zeros((0, 10))
        meta_2025 = np.zeros((0, 2))

    # Split train/val
    n_total = len(win_all)
    n_val = max(5, int(n_total * 0.15))
    n_train = n_total - n_val

    # Normalize
    data_dict = {
        "train_context": ctx_all[:n_train],
        "train_candidates": cand_all[:n_train],
        "train_winners": win_all[:n_train],
        "train_time_gaps": gap_all[:n_train],
        "val_context": ctx_all[n_train:],
        "val_candidates": cand_all[n_train:],
        "val_winners": win_all[n_train:],
        "val_time_gaps": gap_all[n_train:],
        "predict_2025_context": ctx_2025,
        "predict_2025_candidates": cand_2025,
        "predict_2025_winners": win_2025,
        "predict_2025_time_gaps": gap_2025,
        "predict_2025_meta": meta_2025,
    }
    data_dict = normalize_features_v2(data_dict)

    # Save
    torch.save({
        "context": torch.from_numpy(data_dict["train_context"]),
        "candidates": torch.from_numpy(data_dict["train_candidates"]),
        "winners": torch.from_numpy(data_dict["train_winners"]),
        "time_gaps": torch.from_numpy(data_dict["train_time_gaps"]),
    }, PROCESSED_DATA / "features_train_v2_recent.pt")

    torch.save({
        "context": torch.from_numpy(data_dict["val_context"]),
        "candidates": torch.from_numpy(data_dict["val_candidates"]),
        "winners": torch.from_numpy(data_dict["val_winners"]),
        "time_gaps": torch.from_numpy(data_dict["val_time_gaps"]),
    }, PROCESSED_DATA / "features_val_v2_recent.pt")

    torch.save({
        "context": torch.from_numpy(data_dict["predict_2025_context"]),
        "candidates": torch.from_numpy(data_dict["predict_2025_candidates"]),
        "winners": torch.from_numpy(data_dict["predict_2025_winners"]),
        "time_gaps": torch.from_numpy(data_dict["predict_2025_time_gaps"]),
        "meta": torch.from_numpy(data_dict["predict_2025_meta"]),
    }, PROCESSED_DATA / "features_2025_v2_recent.pt")

    d_candidate, d_context = get_feature_vector_size_v2(config.context_window)
    metadata = {
        "d_candidate_raw": d_candidate,
        "d_context_raw": d_context,
        "context_window": config.context_window,
        "num_drivers": len(driver_enc),
        "num_constructors": len(constructor_enc),
        "num_circuits": len(circuit_enc),
        "driver_encoder": driver_enc,
        "constructor_encoder": constructor_enc,
        "circuit_encoder": circuit_enc,
        "train_seqs": n_train,
        "val_seqs": n_val,
        "predict_2025_seqs": len(win_2025),
        "version": "v2_recent",
    }
    with open(PROCESSED_DATA / "metadata_v2_recent.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nRecent dataset: {n_train} train / {n_val} val / {len(win_2025)} test")
    return metadata


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"TRAINING V2 RECENT: {TRAIN_START}-{TRAIN_END}")
    print(f"Device: {DEVICE}, AMP: {USE_AMP}")
    print("=" * 60)

    # Build dataset
    metadata = build_recent_dataset()

    # Load data
    train_data = torch.load(
        PROCESSED_DATA / "features_train_v2_recent.pt", weights_only=False
    )
    val_data = torch.load(
        PROCESSED_DATA / "features_val_v2_recent.pt", weights_only=False
    )

    print(f"Train: {len(train_data['winners'])} seqs, "
          f"Val: {len(val_data['winners'])} seqs")

    # Smaller model for smaller dataset (~57 train seqs)
    # d_model=96 instead of 128, fewer layers
    model = F1WinnerTransformerV2(
        d_model=96,
        n_heads=3,
        n_encoder_layers=2,
        n_cross_attn_layers=2,
        d_ff=384,
        dropout=0.30,  # Higher dropout for small dataset
        context_window=metadata["context_window"],
        num_drivers=metadata["num_drivers"],
        num_constructors=metadata["num_constructors"],
        num_circuits=metadata["num_circuits"],
        d_candidate_raw=metadata["d_candidate_raw"],
        d_context_raw=metadata["d_context_raw"],
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Model V2-Recent: {params:,} params")

    # Train with aggressive regularization
    trainer = TrainerV2(
        model=model,
        train_data=train_data,
        val_data=val_data,
        device=DEVICE,
        batch_size=16,
        learning_rate=5e-4,  # Lower LR for stability
        weight_decay=1e-3,   # Higher weight decay
        epochs=100,
        patience=25,         # More patience for small dataset
        use_amp=USE_AMP,
        checkpoint_dir=MODEL_DIR,
        num_workers=0 if str(DEVICE) == "mps" else 2,
        warmup_epochs=8,     # Longer warmup
        swa_start=30,        # Later SWA
        label_smoothing=0.15, # Higher label smoothing
        noise_std=0.03,      # More noise
        context_dropout_prob=0.25,  # More context dropout
        grad_accum_steps=4,  # Accumulate to effective batch=64
        use_class_weights=True,
    )

    history = trainer.train(early_stopping=True)
    print(f"\nBest val acc V2-Recent: {trainer.best_val_acc:.3f}")
    print(f"Model saved to {MODEL_DIR / 'best_v2_recent.pt'}")


if __name__ == "__main__":
    main()
