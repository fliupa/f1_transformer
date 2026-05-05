"""Build complete dataset for transformer training.

Produces:
- features_train.pt: Combined training sequences (2014-2024, last 85%)
- features_val.pt: Validation sequences (2014-2024, last 15%)
- features_2025.pt: 2025 prediction sequences
- features_finetune.pt: Fine-tuning sequences (2023-2024) for ablation
- features_finetune_val.pt: FT validation sequences
- metadata.pkl: Encoders, feature configs, normalization stats
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Optional
import pickle
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
if os.environ.get("COLAB"):
    RAW_DATA = Path("/content/drive/MyDrive/f1_transformer/data/raw")
    PROCESSED_DATA = Path("/content/drive/MyDrive/f1_transformer/data/processed")

from .encoders import build_encoders, save_encoders
from .feature_engineering import build_all_features, load_race_data, load_circuit_data, load_weather_data
from .sequence_builder import SequenceBuilder, FeatureConfig, get_feature_vector_size


def normalize_features(data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize numeric features to [0, 1] range using min-max scaling."""
    train_candidates = data_dict["train_candidates"]

    # Skip first 3 columns (driver_idx, constructor_idx, circuit_idx)
    numeric_start = 3
    train_numeric = train_candidates[:, :, numeric_start:].reshape(
        -1, train_candidates.shape[2] - numeric_start
    )

    mins = np.min(train_numeric, axis=0)
    maxs = np.max(train_numeric, axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    def _normalize(arr):
        if len(arr) == 0:
            return arr
        numeric = arr[:, :, numeric_start:]
        normalized = (numeric - mins) / ranges
        result = arr.copy()
        result[:, :, numeric_start:] = normalized
        return result

    for key in ["train_candidates", "val_candidates", "finetune_candidates",
                "finetune_val_candidates", "predict_2025_candidates"]:
        if key in data_dict and len(data_dict[key]) > 0:
            data_dict[key] = _normalize(data_dict[key])

    # Context features (all numeric)
    train_context = data_dict["train_context"]
    ctx_mins = np.min(train_context.reshape(-1, train_context.shape[2]), axis=0)
    ctx_maxs = np.max(train_context.reshape(-1, train_context.shape[2]), axis=0)
    ctx_ranges = ctx_maxs - ctx_mins
    ctx_ranges[ctx_ranges == 0] = 1.0

    def _normalize_ctx(arr):
        if len(arr) == 0:
            return arr
        return (arr - ctx_mins) / ctx_ranges

    for key in ["train_context", "val_context", "finetune_context",
                "finetune_val_context", "predict_2025_context"]:
        if key in data_dict and len(data_dict[key]) > 0:
            data_dict[key] = _normalize_ctx(data_dict[key])

    # Time gaps
    train_gaps = data_dict["train_time_gaps"]
    gap_max = max(np.max(train_gaps), 1)
    for key in ["train_time_gaps", "val_time_gaps", "finetune_time_gaps",
                "finetune_val_time_gaps", "predict_2025_time_gaps"]:
        if key in data_dict and len(data_dict[key]) > 0:
            data_dict[key] = data_dict[key] / gap_max

    print(f"Normalized features (numeric: {mins.shape[0]}, context: {ctx_mins.shape[0]})")
    return data_dict


def main():
    """Build complete dataset with combined 2014-2024 + optional FT split."""
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BUILDING F1 DATASET")
    print("=" * 60)

    # 1. Load raw data
    df = load_race_data()
    circuits_df = load_circuit_data()
    weather_df = load_weather_data()

    # 2. Feature engineering
    print("\n--- Feature Engineering ---")
    featured_df = build_all_features(df, circuits_df, weather_df)

    # 3. Build encoders
    print("\n--- Building Encoders ---")
    driver_enc, constructor_enc, circuit_enc = build_encoders(featured_df)
    save_encoders(driver_enc, constructor_enc, circuit_enc, PROCESSED_DATA)

    # 4. Build sequences
    print("\n--- Building Sequences ---")
    config = FeatureConfig(context_window=10)
    builder = SequenceBuilder(featured_df, driver_enc, constructor_enc, circuit_enc, config)

    # COMBINED 2014-2024 (main approach)
    print("\nCombined 2014-2024 sequences:")
    ctx_all, cand_all, win_all, gap_all, meta_all = builder.build_sequences(2014, 2024)

    # 2025 prediction sequences
    print("\n2025 prediction sequences:")
    ctx_2025, cand_2025, win_2025, gap_2025, meta_2025 = builder.build_sequences(2025, 2025)

    # 5. Split combined into train/val (time-based: last 15% for validation)
    n_total = len(win_all)
    n_val = max(10, int(n_total * 0.15))
    n_train = n_total - n_val

    # 6. Optional: Fine-tuning split for ablation (2023-2024 only)
    print("\nFine-tuning split (2023-2024) for ablation:")
    ctx_ft, cand_ft, win_ft, gap_ft, _ = builder.build_sequences(2023, 2024)
    n_ft = len(win_ft)
    n_ft_val = max(1, int(n_ft * 0.2))
    n_ft_train = n_ft - n_ft_val

    # 7. Normalize features
    print("\n--- Normalizing Features ---")
    data_dict = {
        # Combined train/val
        "train_context": ctx_all[:n_train], "train_candidates": cand_all[:n_train],
        "train_winners": win_all[:n_train], "train_time_gaps": gap_all[:n_train],
        "val_context": ctx_all[n_train:], "val_candidates": cand_all[n_train:],
        "val_winners": win_all[n_train:], "val_time_gaps": gap_all[n_train:],
        # FT split for ablation
        "finetune_context": ctx_ft[:n_ft_train], "finetune_candidates": cand_ft[:n_ft_train],
        "finetune_winners": win_ft[:n_ft_train], "finetune_time_gaps": gap_ft[:n_ft_train],
        "finetune_val_context": ctx_ft[n_ft_train:], "finetune_val_candidates": cand_ft[n_ft_train:],
        "finetune_val_winners": win_ft[n_ft_train:], "finetune_val_time_gaps": gap_ft[n_ft_train:],
        # 2025
        "predict_2025_context": ctx_2025, "predict_2025_candidates": cand_2025,
        "predict_2025_winners": win_2025, "predict_2025_time_gaps": gap_2025,
        "predict_2025_meta": meta_2025,
    }
    data_dict = normalize_features(data_dict)

    # 8. Save everything
    print("\n--- Saving Datasets ---")

    # Combined training set
    torch.save({
        "context": torch.from_numpy(data_dict["train_context"]),
        "candidates": torch.from_numpy(data_dict["train_candidates"]),
        "winners": torch.from_numpy(data_dict["train_winners"]),
        "time_gaps": torch.from_numpy(data_dict["train_time_gaps"]),
    }, PROCESSED_DATA / "features_train.pt")

    torch.save({
        "context": torch.from_numpy(data_dict["val_context"]),
        "candidates": torch.from_numpy(data_dict["val_candidates"]),
        "winners": torch.from_numpy(data_dict["val_winners"]),
        "time_gaps": torch.from_numpy(data_dict["val_time_gaps"]),
    }, PROCESSED_DATA / "features_val.pt")

    # Fine-tuning set (for ablation study)
    torch.save({
        "context": torch.from_numpy(data_dict["finetune_context"]),
        "candidates": torch.from_numpy(data_dict["finetune_candidates"]),
        "winners": torch.from_numpy(data_dict["finetune_winners"]),
        "time_gaps": torch.from_numpy(data_dict["finetune_time_gaps"]),
    }, PROCESSED_DATA / "features_finetune.pt")

    torch.save({
        "context": torch.from_numpy(data_dict["finetune_val_context"]),
        "candidates": torch.from_numpy(data_dict["finetune_val_candidates"]),
        "winners": torch.from_numpy(data_dict["finetune_val_winners"]),
        "time_gaps": torch.from_numpy(data_dict["finetune_val_time_gaps"]),
    }, PROCESSED_DATA / "features_finetune_val.pt")

    # 2025 prediction set
    torch.save({
        "context": torch.from_numpy(data_dict["predict_2025_context"]),
        "candidates": torch.from_numpy(data_dict["predict_2025_candidates"]),
        "winners": torch.from_numpy(data_dict["predict_2025_winners"]),
        "time_gaps": torch.from_numpy(data_dict["predict_2025_time_gaps"]),
        "meta": torch.from_numpy(data_dict["predict_2025_meta"]),
    }, PROCESSED_DATA / "features_2025.pt")

    # Metadata
    d_candidate, d_context = get_feature_vector_size(config)
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
        "finetune_seqs": n_ft_train,
        "finetune_val_seqs": n_ft_val,
        "predict_2025_seqs": len(win_2025),
        "feature_config": config,
    }
    with open(PROCESSED_DATA / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Combined training:  {n_train} train / {n_val} val")
    print(f"  FT ablation:         {n_ft_train} train / {n_ft_val} val")
    print(f"  2025 predictions:    {len(win_2025)} races")
    print(f"  Context window:      {config.context_window}")
    print(f"  Candidate features:  {d_candidate}")
    print(f"  Context features:    {d_context}")
    print(f"  Drivers:             {len(driver_enc)}")
    print(f"  Constructors:        {len(constructor_enc)}")
    print(f"  Circuits:            {len(circuit_enc)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
