"""Build dataset V2 with improved features and context distribution stats.

Uses feature_engineering_v2 + sequence_builder_v2.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict
import pickle
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
if os.environ.get("COLAB"):
    RAW_DATA = Path("/content/drive/MyDrive/f1_transformer/data/raw")
    PROCESSED_DATA = Path("/content/drive/MyDrive/f1_transformer/data/processed")

from .encoders import build_encoders, save_encoders
from .feature_engineering_v2 import (
    build_all_features_v2, load_race_data, load_circuit_data, load_weather_data,
    NUMERIC_DRIVER_COLS_V2, CONSTRUCTOR_COLS_V2, CIRCUIT_COLS_V2, WEATHER_COLS_V2, RACE_CONTEXT_COLS_V2,
    get_feature_vector_size_v2,
)
from .sequence_builder_v2 import SequenceBuilderV2, FeatureConfigV2


def normalize_features_v2(data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize numeric features to [0, 1] range (same logic as V1)."""
    train_candidates = data_dict["train_candidates"]
    numeric_start = 3

    # Candidate numeric features
    train_numeric = train_candidates[:, :, numeric_start:].reshape(-1, train_candidates.shape[2] - numeric_start)
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

    for key in ["train_candidates", "val_candidates", "predict_2025_candidates"]:
        if key in data_dict and len(data_dict[key]) > 0:
            data_dict[key] = _normalize(data_dict[key])

    # Context features
    train_context = data_dict["train_context"]
    ctx_mins = np.min(train_context.reshape(-1, train_context.shape[2]), axis=0)
    ctx_maxs = np.max(train_context.reshape(-1, train_context.shape[2]), axis=0)
    ctx_ranges = ctx_maxs - ctx_mins
    ctx_ranges[ctx_ranges == 0] = 1.0

    def _normalize_ctx(arr):
        if len(arr) == 0:
            return arr
        return (arr - ctx_mins) / ctx_ranges

    for key in ["train_context", "val_context", "predict_2025_context"]:
        if key in data_dict and len(data_dict[key]) > 0:
            data_dict[key] = _normalize_ctx(data_dict[key])

    # Time gaps
    train_gaps = data_dict["train_time_gaps"]
    gap_max = max(np.max(train_gaps), 1)
    for key in ["train_time_gaps", "val_time_gaps", "predict_2025_time_gaps"]:
        if key in data_dict and len(data_dict[key]) > 0:
            data_dict[key] = data_dict[key] / gap_max

    print(f"Normalized V2 features (numeric: {mins.shape[0]}, context: {ctx_mins.shape[0]})")
    return data_dict


def main():
    """Build V2 dataset with improved features and context stats."""
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BUILDING F1 DATASET V2 (ELO, momentum, track affinity, std/max context)")
    print("=" * 60)

    # 1. Load raw data
    df = load_race_data()
    circuits_df = load_circuit_data()
    weather_df = load_weather_data()

    # 2. V2 Feature engineering (ELO, momentum, track affinity, etc.)
    print("\n--- Feature Engineering V2 ---")
    featured_df = build_all_features_v2(df, circuits_df, weather_df)

    # 3. Build encoders
    print("\n--- Building Encoders ---")
    driver_enc, constructor_enc, circuit_enc = build_encoders(featured_df)
    save_encoders(driver_enc, constructor_enc, circuit_enc, PROCESSED_DATA)

    # 4. Build sequences with V2 builder (mean+std+max context)
    print("\n--- Building V2 Sequences ---")
    config = FeatureConfigV2(context_window=10)
    builder = SequenceBuilderV2(
        featured_df, driver_enc, constructor_enc, circuit_enc, config,
        numeric_driver_cols=NUMERIC_DRIVER_COLS_V2,
        constructor_cols=CONSTRUCTOR_COLS_V2,
        circuit_cols=CIRCUIT_COLS_V2,
        weather_cols=WEATHER_COLS_V2,
        race_context_cols=RACE_CONTEXT_COLS_V2,
    )

    # Combined 2014-2024
    print("\nCombined 2014-2024 sequences:")
    ctx_all, cand_all, win_all, gap_all, meta_all = builder.build_sequences(2014, 2024)

    # 2025 prediction sequences
    print("\n2025 prediction sequences:")
    has_2025 = not featured_df[featured_df["year"] == 2025].empty
    if not has_2025:
        print("  WARNING: No 2025 data found! Run fetch_data.py first.")
    ctx_2025, cand_2025, win_2025, gap_2025, meta_2025 = builder.build_sequences(2025, 2025)

    # 5. Split into train/val
    n_total = len(win_all)
    n_val = max(10, int(n_total * 0.15))
    n_train = n_total - n_val

    # 6. Normalize
    print("\n--- Normalizing V2 Features ---")
    data_dict = {
        "train_context": ctx_all[:n_train], "train_candidates": cand_all[:n_train],
        "train_winners": win_all[:n_train], "train_time_gaps": gap_all[:n_train],
        "val_context": ctx_all[n_train:], "val_candidates": cand_all[n_train:],
        "val_winners": win_all[n_train:], "val_time_gaps": gap_all[n_train:],
        "predict_2025_context": ctx_2025, "predict_2025_candidates": cand_2025,
        "predict_2025_winners": win_2025, "predict_2025_time_gaps": gap_2025,
        "predict_2025_meta": meta_2025,
    }
    data_dict = normalize_features_v2(data_dict)

    # 7. Save
    print("\n--- Saving V2 Datasets ---")
    torch.save({
        "context": torch.from_numpy(data_dict["train_context"]),
        "candidates": torch.from_numpy(data_dict["train_candidates"]),
        "winners": torch.from_numpy(data_dict["train_winners"]),
        "time_gaps": torch.from_numpy(data_dict["train_time_gaps"]),
    }, PROCESSED_DATA / "features_train_v2.pt")

    torch.save({
        "context": torch.from_numpy(data_dict["val_context"]),
        "candidates": torch.from_numpy(data_dict["val_candidates"]),
        "winners": torch.from_numpy(data_dict["val_winners"]),
        "time_gaps": torch.from_numpy(data_dict["val_time_gaps"]),
    }, PROCESSED_DATA / "features_val_v2.pt")

    torch.save({
        "context": torch.from_numpy(data_dict["predict_2025_context"]),
        "candidates": torch.from_numpy(data_dict["predict_2025_candidates"]),
        "winners": torch.from_numpy(data_dict["predict_2025_winners"]),
        "time_gaps": torch.from_numpy(data_dict["predict_2025_time_gaps"]),
        "meta": torch.from_numpy(data_dict["predict_2025_meta"]),
    }, PROCESSED_DATA / "features_2025_v2.pt")

    # Metadata
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
        "version": "v2",
    }
    with open(PROCESSED_DATA / "metadata_v2.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # Summary
    print("\n" + "=" * 60)
    print("V2 DATASET SUMMARY")
    print("=" * 60)
    print(f"  Training:   {n_train} seqs")
    print(f"  Validation: {n_val} seqs")
    print(f"  2025:       {len(win_2025)} races")
    print(f"  Candidate features: {d_candidate}  (V1: 32)")
    print(f"  Context features:   {d_context}  (V1: 29)")
    print(f"  Drivers:            {len(driver_enc)}")
    print(f"  Constructors:       {len(constructor_enc)}")
    print(f"  Circuits:           {len(circuit_enc)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
