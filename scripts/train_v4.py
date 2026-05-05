"""Train F1 Transformer V4: Grid-First Cascade.

Key insight: 87.6% of winners come from top-3 on the grid, 93.2% from top-5.
Instead of 20-way classification, V4 filters candidates to top-K by grid
position, reducing to a K-class problem (K=7 by default).

This dramatically simplifies the learning task with tiny data (~200 seqs).
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
TOP_K = 7  # >=95% of winners covered
PROCESSED = PROJECT_ROOT / "data" / "processed"


class SequenceBuilderV4(SequenceBuilderV2):
    """V4: Same as V2 but candidates are filtered to top-K by grid position.

    The context still uses ALL drivers (full field stats).
    Only the candidate list (prediction targets) is limited to top-K grid positions.
    """

    def __init__(self, *args, top_k: int = TOP_K, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = top_k

    def build_sequences(self, start_year: int, end_year: int):
        """Override: filter candidates to top-K by grid position."""
        N = self.config.context_window

        race_list = self.races[
            (self.races["year"] >= start_year) & (self.races["year"] <= end_year)
        ].sort_values(["year", "round"]).reset_index(drop=True)

        all_races = self.races.sort_values(["year", "round"]).reset_index(drop=True)
        race_index_map = {row["race_id"]: i for i, row in all_races.iterrows()}

        context_seqs = []
        candidate_feats = []
        winner_indices = []
        time_gaps_list = []
        metadata = []
        skipped_no_winner = 0

        for _, target_race in race_list.iterrows():
            target_year = target_race["year"]
            target_round = target_race["round"]
            target_id = target_race["race_id"]

            try:
                target_chrono_idx = race_index_map[target_id]
            except KeyError:
                continue

            prev_indices = list(range(max(0, target_chrono_idx - N), target_chrono_idx))
            if len(prev_indices) < 2:
                continue

            prev_races = all_races.iloc[prev_indices]

            # Build context from ALL drivers in past races (same as V2)
            context_rows = []
            time_gaps = np.zeros(N, dtype=np.float32)
            target_date = pd.Timestamp(target_race["event_date"])

            for j, (_, prev_race) in enumerate(prev_races.iterrows()):
                prev_df = self.get_race_drivers(prev_race["year"], prev_race["round"])
                if len(prev_df) < 2:
                    continue
                ctx = self.build_context_features(prev_df)
                context_rows.append(ctx)
                prev_date = pd.Timestamp(prev_race["event_date"])
                time_gaps[j] = max(1, (target_date - prev_date).days)

            if len(context_rows) < 2:
                continue

            if len(context_rows) < N:
                pad_size = N - len(context_rows)
                pad_vec = np.zeros_like(context_rows[0])
                context_rows = [pad_vec] * pad_size + context_rows
                time_gaps = np.concatenate([np.full(pad_size, 365.0), time_gaps[-len(context_rows):]])

            context = np.array(context_rows[-N:])
            time_gaps = time_gaps[-N:]

            # ─── V4: Build candidates for target race (top-K by grid only) ───
            candidates_df = self.get_race_drivers(target_year, target_round)
            # Already sorted by grid_position from get_race_drivers()

            # Take top-K by grid
            top_k_df = candidates_df.head(self.top_k)

            # Build features for top-K candidates
            candidates = self.build_driver_features(top_k_df)

            # Find winner index within top-K
            winner_mask = top_k_df["finish_position"] == 1
            if not winner_mask.any():
                skipped_no_winner += 1
                continue
            winner_idx = int(top_k_df[winner_mask].index[0])
            # winner_idx is the position in top_k_df (0 to K-1)

            context_seqs.append(context)
            candidate_feats.append(candidates)
            winner_indices.append(winner_idx)
            time_gaps_list.append(time_gaps)
            metadata.append([target_year, target_round])

        print(f"Built {len(context_seqs)} sequences for {start_year}-{end_year} "
              f"(top-{self.top_k} grid, skipped {skipped_no_winner} with winner outside top-{self.top_k})")

        return (
            np.array(context_seqs, dtype=np.float32),
            np.array(candidate_feats, dtype=np.float32),
            np.array(winner_indices, dtype=np.int64),
            np.array(time_gaps_list, dtype=np.float32),
            np.array(metadata, dtype=np.int64),
        )


def build_v4_dataset(top_k=TOP_K):
    """Build V4 dataset: top-K grid candidates, train 2014-2024, test 2025."""
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    print(f"Building V4 dataset (top-{top_k} grid candidates)...")
    df = load_race_data()
    circuits_df = load_circuit_data()
    weather_df = load_weather_data()

    featured_df = build_all_features_v2(df, circuits_df, weather_df)
    driver_enc, constructor_enc, circuit_enc = build_encoders(featured_df)
    save_encoders(driver_enc, constructor_enc, circuit_enc, PROCESSED_DATA)

    config = FeatureConfigV2(context_window=10)
    builder = SequenceBuilderV4(
        featured_df, driver_enc, constructor_enc, circuit_enc, config,
        top_k=top_k,
        numeric_driver_cols=NUMERIC_DRIVER_COLS_V2,
        constructor_cols=CONSTRUCTOR_COLS_V2,
        circuit_cols=CIRCUIT_COLS_V2,
        weather_cols=WEATHER_COLS_V2,
        race_context_cols=RACE_CONTEXT_COLS_V2,
    )

    # Train on 2014-2024
    ctx_all, cand_all, win_all, gap_all, meta_all = builder.build_sequences(2014, 2024)

    # 2025 prediction
    has_2025 = not featured_df[featured_df["year"] == 2025].empty
    if has_2025:
        ctx_2025, cand_2025, win_2025, gap_2025, meta_2025 = builder.build_sequences(2025, 2025)
    else:
        ctx_2025 = np.zeros((0, 10, 1)); cand_2025 = np.zeros((0, top_k, 1))
        win_2025 = np.zeros((0,)); gap_2025 = np.zeros((0, 10)); meta_2025 = np.zeros((0, 2))

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
    }, PROCESSED_DATA / "features_train_v4.pt")

    torch.save({
        "context": torch.from_numpy(data_dict["val_context"]),
        "candidates": torch.from_numpy(data_dict["val_candidates"]),
        "winners": torch.from_numpy(data_dict["val_winners"]),
        "time_gaps": torch.from_numpy(data_dict["val_time_gaps"]),
    }, PROCESSED_DATA / "features_val_v4.pt")

    torch.save({
        "context": torch.from_numpy(data_dict["predict_2025_context"]),
        "candidates": torch.from_numpy(data_dict["predict_2025_candidates"]),
        "winners": torch.from_numpy(data_dict["predict_2025_winners"]),
        "time_gaps": torch.from_numpy(data_dict["predict_2025_time_gaps"]),
        "meta": torch.from_numpy(data_dict["predict_2025_meta"]),
    }, PROCESSED_DATA / "features_2025_v4.pt")

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
        "version": "v4",
        "top_k": top_k,
    }
    with open(PROCESSED_DATA / "metadata_v4.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nV4 dataset: {n_train} train / {n_val} val / {len(win_2025)} test")
    print(f"  Top-K: {top_k}, d_candidate={d_candidate}, d_context={d_context}")
    print(f"  Winners outside top-{top_k} in training: "
          f"{sum(1 for w in win_all if w >= top_k)}/{len(win_all)}")
    return metadata


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    top_k = TOP_K

    print("=" * 60)
    print(f"TRAINING V4: Grid-First Cascade (top-{top_k})")
    print(f"Device: {DEVICE}, AMP: {USE_AMP}")
    print("=" * 60)

    metadata = build_v4_dataset(top_k=top_k)

    train_data = torch.load(PROCESSED_DATA / "features_train_v4.pt", weights_only=False)
    val_data = torch.load(PROCESSED_DATA / "features_val_v4.pt", weights_only=False)

    print(f"Train: {len(train_data['winners'])} seqs, "
          f"Val: {len(val_data['winners'])} seqs")
    print(f"Candidates: {train_data['candidates'].shape[1]} (was 20)")

    # Model: max_drivers_per_race = K instead of 20
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
        max_drivers_per_race=top_k,  # V4: K candidates instead of 20
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Model V4: {params:,} params (output: {top_k} classes)")

    # For class weights, we need to map winners to K classes
    # Winner indices should already be 0..K-1
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
        num_classes=top_k,
    )

    trainer.checkpoint_filename = "best_v4.pt"
    history = trainer.train(early_stopping=True)
    print(f"\nBest val acc V4: {trainer.best_val_acc:.3f}")
    print(f"Model saved to {MODEL_DIR / 'best_v4.pt'}")


if __name__ == "__main__":
    main()
