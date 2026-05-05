"""Train F1 Transformer V3: context sin driver IDs, 2020-2024 data, pesos agresivos.

Key insight: V1/V2 memorizan los driver IDs del top-3/winner en el contexto,
aprendiendo "X ganó antes, predice X". Al eliminar estos IDs, el modelo
debe basarse en features reales de rendimiento (ELO, momentum, track affinity).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pickle
import numpy as np
import pandas as pd

from src.model.transformer_model_v2 import F1WinnerTransformerV2
from src.training.trainer_v2 import TrainerV2
from src.preprocessing.build_dataset_v2 import (
    normalize_features_v2, PROCESSED_DATA,
)
from src.preprocessing.feature_engineering_v2 import (
    build_all_features_v2, load_race_data, load_circuit_data, load_weather_data,
    NUMERIC_DRIVER_COLS_V2, CONSTRUCTOR_COLS_V2, CIRCUIT_COLS_V2,
    WEATHER_COLS_V2, RACE_CONTEXT_COLS_V2,
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

# ─── V3: Context sin driver IDs ────────────────────────────────────────


class SequenceBuilderV3(SequenceBuilderV2):
    """V3: Same as V2 but context excludes top-3/winner driver IDs."""

    def build_context_features(self, race_df: pd.DataFrame) -> np.ndarray:
        """Context sin driver IDs - solo features estadisticas de la carrera."""
        n_numeric = len(self.numeric_driver_cols)
        n_constructor = len(self.constructor_cols)
        n_circuit = len(self.circuit_cols)
        n_weather = len(self.weather_cols)
        n_race = len(self.race_context_cols)

        # V3: solo mean+std+max de features, sin IDs de pilotos
        d_context = (
            n_numeric * 3 + n_constructor * 3 + n_circuit + n_weather + n_race
        )

        features = np.zeros(d_context, dtype=np.float32)
        offset = 0

        # Mean of driver numeric features
        for col in self.numeric_driver_cols:
            vals = race_df[col].values if col in race_df.columns else np.array([0])
            features[offset] = np.nanmean(vals) if not np.all(np.isnan(vals)) else 0
            offset += 1

        # Std of driver numeric features
        for col in self.numeric_driver_cols:
            vals = race_df[col].values if col in race_df.columns else np.array([0])
            features[offset] = np.nanstd(vals) if not np.all(np.isnan(vals)) else 0
            offset += 1

        # Max of driver numeric features
        for col in self.numeric_driver_cols:
            vals = race_df[col].values if col in race_df.columns else np.array([0])
            features[offset] = np.nanmax(vals) if not np.all(np.isnan(vals)) else 0
            offset += 1

        # Mean of constructor features
        for col in self.constructor_cols:
            vals = race_df[col].values if col in race_df.columns else np.array([0])
            features[offset] = np.nanmean(vals) if not np.all(np.isnan(vals)) else 0
            offset += 1

        # Std of constructor features
        for col in self.constructor_cols:
            vals = race_df[col].values if col in race_df.columns else np.array([0])
            features[offset] = np.nanstd(vals) if not np.all(np.isnan(vals)) else 0
            offset += 1

        # Max of constructor features
        for col in self.constructor_cols:
            vals = race_df[col].values if col in race_df.columns else np.array([0])
            features[offset] = np.nanmax(vals) if not np.all(np.isnan(vals)) else 0
            offset += 1

        # Circuit features
        for col in self.circuit_cols:
            val = race_df[col].values[0] if col in race_df.columns else 0
            features[offset] = val if not pd.isna(val) else 0
            offset += 1

        # Weather features
        for col in self.weather_cols:
            val = race_df[col].values[0] if col in race_df.columns else 0
            features[offset] = val if not pd.isna(val) else 0
            offset += 1

        # Race context
        for col in self.race_context_cols:
            val = race_df[col].values[0] if col in race_df.columns else 0
            features[offset] = val if not pd.isna(val) else 0
            offset += 1

        return features


def get_feature_vector_size_v3(context_window: int = 10) -> tuple:
    """V3 feature sizes: candidate same as V2, context sin driver IDs."""
    n_numeric = len(NUMERIC_DRIVER_COLS_V2)
    n_constructor = len(CONSTRUCTOR_COLS_V2)
    n_circuit = len(CIRCUIT_COLS_V2)
    n_weather = len(WEATHER_COLS_V2)
    n_race = len(RACE_CONTEXT_COLS_V2)

    # Per-driver: same as V2
    d_candidate = 3 + n_numeric + n_constructor + n_circuit + n_weather + n_race

    # Per-race context: solo distribuciones estadisticas (sin top-3/winner IDs)
    d_context = (
        n_numeric * 3 +      # mean + std + max of driver features
        n_constructor * 3 +  # mean + std + max of constructor features
        n_circuit +          # circuit
        n_weather +          # weather
        n_race               # round
    )

    return d_candidate, d_context


def build_v3_dataset():
    """Build V3 dataset: 2020-2024 training, context sin driver IDs."""
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    df = load_race_data()
    circuits_df = load_circuit_data()
    weather_df = load_weather_data()

    featured_df = build_all_features_v2(df, circuits_df, weather_df)

    driver_enc, constructor_enc, circuit_enc = build_encoders(featured_df)
    save_encoders(driver_enc, constructor_enc, circuit_enc, PROCESSED_DATA)

    config = FeatureConfigV2(context_window=10)
    builder = SequenceBuilderV3(
        featured_df, driver_enc, constructor_enc, circuit_enc, config,
        numeric_driver_cols=NUMERIC_DRIVER_COLS_V2,
        constructor_cols=CONSTRUCTOR_COLS_V2,
        circuit_cols=CIRCUIT_COLS_V2,
        weather_cols=WEATHER_COLS_V2,
        race_context_cols=RACE_CONTEXT_COLS_V2,
    )

    # Train on 2020-2024 (compromise: relevance + quantity)
    ctx_all, cand_all, win_all, gap_all, meta_all = builder.build_sequences(
        2020, 2024
    )

    # 2025 prediction
    has_2025 = not featured_df[featured_df["year"] == 2025].empty
    if has_2025:
        ctx_2025, cand_2025, win_2025, gap_2025, meta_2025 = builder.build_sequences(
            2025, 2025
        )
    else:
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
    }, PROCESSED_DATA / "features_train_v3.pt")

    torch.save({
        "context": torch.from_numpy(data_dict["val_context"]),
        "candidates": torch.from_numpy(data_dict["val_candidates"]),
        "winners": torch.from_numpy(data_dict["val_winners"]),
        "time_gaps": torch.from_numpy(data_dict["val_time_gaps"]),
    }, PROCESSED_DATA / "features_val_v3.pt")

    torch.save({
        "context": torch.from_numpy(data_dict["predict_2025_context"]),
        "candidates": torch.from_numpy(data_dict["predict_2025_candidates"]),
        "winners": torch.from_numpy(data_dict["predict_2025_winners"]),
        "time_gaps": torch.from_numpy(data_dict["predict_2025_time_gaps"]),
        "meta": torch.from_numpy(data_dict["predict_2025_meta"]),
    }, PROCESSED_DATA / "features_2025_v3.pt")

    d_candidate, d_context = get_feature_vector_size_v3(config.context_window)
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
        "version": "v3",
    }
    with open(PROCESSED_DATA / "metadata_v3.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nV3 dataset: {n_train} train / {n_val} val / {len(win_2025)} test")
    print(f"d_candidate={d_candidate}, d_context={d_context} (V2 was 139)")
    return metadata


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRAINING V3: 2020-2024, Context sin Driver IDs")
    print(f"Device: {DEVICE}, AMP: {USE_AMP}")
    print("=" * 60)

    metadata = build_v3_dataset()

    train_data = torch.load(
        PROCESSED_DATA / "features_train_v3.pt", weights_only=False
    )
    val_data = torch.load(
        PROCESSED_DATA / "features_val_v3.pt", weights_only=False
    )

    print(f"Train: {len(train_data['winners'])} seqs, "
          f"Val: {len(val_data['winners'])} seqs")

    # Model: compact, strong regularization
    model = F1WinnerTransformerV2(
        d_model=128,
        n_heads=4,
        n_encoder_layers=4,
        n_cross_attn_layers=3,
        d_ff=512,
        dropout=0.30,
        context_window=metadata["context_window"],
        num_drivers=metadata["num_drivers"],
        num_constructors=metadata["num_constructors"],
        num_circuits=metadata["num_circuits"],
        d_candidate_raw=metadata["d_candidate_raw"],
        d_context_raw=metadata["d_context_raw"],
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Model V3: {params:,} params")

    # Stronger class weights via sqrt inverse frequency
    trainer = TrainerV2(
        model=model,
        train_data=train_data,
        val_data=val_data,
        device=DEVICE,
        batch_size=16,
        learning_rate=5e-4,
        weight_decay=1e-3,
        epochs=100,
        patience=25,
        use_amp=USE_AMP,
        checkpoint_dir=MODEL_DIR,
        num_workers=0 if str(DEVICE) == "mps" else 2,
        warmup_epochs=8,
        swa_start=30,
        label_smoothing=0.15,
        noise_std=0.03,
        context_dropout_prob=0.25,
        grad_accum_steps=4,
        use_class_weights=True,
    )

    trainer.checkpoint_filename = "best_v3.pt"
    history = trainer.train(early_stopping=True)
    print(f"\nBest val acc V3: {trainer.best_val_acc:.3f}")
    print(f"Model saved to {MODEL_DIR / 'best_v3.pt'}")


if __name__ == "__main__":
    main()
