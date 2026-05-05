"""Quick test: predict 2024 season and compare with 2025 results."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import pickle
import numpy as np

from src.model.transformer_model_v2 import F1WinnerTransformerV2
from src.preprocessing.build_dataset_v2 import (
    normalize_features_v2, PROCESSED_DATA,
)
from src.preprocessing.feature_engineering_v2 import (
    build_all_features_v2, load_race_data, load_circuit_data, load_weather_data,
    NUMERIC_DRIVER_COLS_V2, CONSTRUCTOR_COLS_V2, CIRCUIT_COLS_V2,
    WEATHER_COLS_V2, RACE_CONTEXT_COLS_V2,
)
from src.preprocessing.encoders import build_encoders
from src.preprocessing.sequence_builder_v2 import SequenceBuilderV2, FeatureConfigV2
from scripts.train_mlp import F1WinnerMLP

PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "final"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Real 2024 winners
REAL_WINNERS_2024 = {
    1: 'VER', 2: 'VER', 3: 'SAI', 4: 'VER', 5: 'NOR', 6: 'LEC',
    7: 'VER', 8: 'RUS', 9: 'VER', 10: 'HAM', 11: 'PIA', 12: 'NOR',
    13: 'RUS', 14: 'LEC', 15: 'NOR', 16: 'VER', 17: 'SAI', 18: 'VER',
    19: 'NOR', 20: 'RUS', 21: 'VER', 22: 'NOR', 23: 'RUS', 24: 'NOR',
}

# Real 2025 winners (for comparison)
REAL_WINNERS_2025 = {
    1: 'NOR', 2: 'PIA', 3: 'VER', 4: 'RUS', 5: 'VER', 6: 'NOR',
    7: 'PIA', 8: 'NOR', 9: 'VER', 10: 'NOR', 11: 'PIA', 12: 'RUS',
    13: 'PIA', 14: 'VER', 15: 'VER', 16: 'NOR', 17: 'PIA', 18: 'VER',
    19: 'PIA', 20: 'VER', 21: 'PIA', 22: 'VER', 23: 'NOR', 24: 'NOR',
}


def get_winner_idx(candidates_row, driver_enc, real_abbr):
    driver_ids = candidates_row[:, 0].long().numpy()
    for i, idx in enumerate(driver_ids):
        try:
            if driver_enc.decode(int(idx)) == real_abbr:
                return i
        except:
            pass
    return None


def build_2024_features():
    """Build V2 features for 2024 season only (train on 2014-2023)."""
    print("Building 2024 features...")
    df = load_race_data()
    circuits_df = load_circuit_data()
    weather_df = load_weather_data()

    featured_df = build_all_features_v2(df, circuits_df, weather_df)

    driver_enc, constructor_enc, circuit_enc = build_encoders(featured_df)
    config = FeatureConfigV2(context_window=10)
    builder = SequenceBuilderV2(
        featured_df, driver_enc, constructor_enc, circuit_enc, config,
        numeric_driver_cols=NUMERIC_DRIVER_COLS_V2,
        constructor_cols=CONSTRUCTOR_COLS_V2,
        circuit_cols=CIRCUIT_COLS_V2,
        weather_cols=WEATHER_COLS_V2,
        race_context_cols=RACE_CONTEXT_COLS_V2,
    )

    # Train: 2014-2023
    ctx_train, cand_train, win_train, gap_train, _ = builder.build_sequences(2014, 2023)
    # Test: 2024
    ctx_2024, cand_2024, win_2024, gap_2024, _ = builder.build_sequences(2024, 2024)

    n_train = len(win_train)
    n_val = max(5, int(n_train * 0.15))

    # Normalize
    data_dict = {
        "train_context": ctx_train[:n_train - n_val],
        "train_candidates": cand_train[:n_train - n_val],
        "train_winners": win_train[:n_train - n_val],
        "train_time_gaps": gap_train[:n_train - n_val],
        "val_context": ctx_train[n_train - n_val:],
        "val_candidates": cand_train[n_train - n_val:],
        "val_winners": win_train[n_train - n_val:],
        "val_time_gaps": gap_train[n_train - n_val:],
        "predict_2025_context": ctx_2024,
        "predict_2025_candidates": cand_2024,
        "predict_2025_winners": win_2024,
        "predict_2025_time_gaps": gap_2024,
        "predict_2025_meta": np.zeros((len(win_2024), 2)),
    }
    data_dict = normalize_features_v2(data_dict)

    torch.save({
        "context": torch.from_numpy(data_dict["predict_2025_context"]),
        "candidates": torch.from_numpy(data_dict["predict_2025_candidates"]),
        "winners": torch.from_numpy(data_dict["predict_2025_winners"]),
        "time_gaps": torch.from_numpy(data_dict["predict_2025_time_gaps"]),
        "meta": torch.from_numpy(data_dict["predict_2025_meta"]),
    }, PROCESSED / "features_2024_v2.pt")

    print(f"Built: {n_train - n_val} train / {n_val} val / {len(win_2024)} test (2024)")
    return driver_enc


def predict_year(model, features_path, metadata, real_winners, year_label, device):
    """Predict race winners for a given year."""
    data = torch.load(features_path, weights_only=False)
    context = data["context"]
    candidates = data["candidates"]
    time_gaps = data.get("time_gaps", torch.zeros(len(context), metadata["context_window"]))
    driver_enc = metadata["driver_encoder"]

    correct = 0
    total = 0
    results = []

    for i in range(len(context)):
        round_num = i + 1
        real_abbr = real_winners.get(round_num, None)
        if real_abbr is None:
            continue

        real_idx = get_winner_idx(candidates[i], driver_enc, real_abbr)
        if real_idx is None:
            continue

        ctx = context[i:i+1].to(device)
        cand = candidates[i:i+1].to(device)
        gaps = time_gaps[i:i+1].to(device)

        with torch.no_grad():
            if isinstance(model, F1WinnerMLP):
                logits = model(ctx, cand)
            else:
                logits = model(ctx, cand, gaps)
            probs = F.softmax(logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()

        pred_name = driver_enc.decode(int(candidates[i, pred_idx, 0].item()))
        matched = pred_idx == real_idx
        correct += int(matched)
        total += 1
        results.append((round_num, pred_name, real_abbr, matched))

    acc = correct / max(total, 1)
    print(f"\n{year_label}: {correct}/{total} = {acc:.1%}")

    # Per-driver breakdown
    driver_wins = {}
    for round_num, pred, real, matched in results:
        if real not in driver_wins:
            driver_wins[real] = {"total": 0, "correct": 0}
        driver_wins[real]["total"] += 1
        driver_wins[real]["correct"] += int(matched)

    print(f"  Winners: {sorted(driver_wins.keys())}")
    for d in sorted(driver_wins.keys()):
        dw = driver_wins[d]
        print(f"    {d}: {dw['correct']}/{dw['total']}")

    return acc, results


def main():
    print("=" * 70)
    print("TESTING MODEL ON 2024 vs 2025")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Build 2024 features if needed
    features_2024_path = PROCESSED / "features_2024_v2.pt"
    features_2024_v4_path = PROCESSED / "features_2024_v4.pt"

    # Build V4 2024 features
    if not features_2024_v4_path.exists():
        from scripts.train_v4 import SequenceBuilderV4, build_v4_dataset
        print("\nBuilding 2024 features in V4 format (top-7)...")
        df = load_race_data()
        circuits_df = load_circuit_data()
        weather_df = load_weather_data()
        featured_df = build_all_features_v2(df, circuits_df, weather_df)
        driver_enc, constructor_enc, circuit_enc = build_encoders(featured_df)

        config = FeatureConfigV2(context_window=10)
        builder_v4 = SequenceBuilderV4(
            featured_df, driver_enc, constructor_enc, circuit_enc, config,
            top_k=7,
            numeric_driver_cols=NUMERIC_DRIVER_COLS_V2,
            constructor_cols=CONSTRUCTOR_COLS_V2,
            circuit_cols=CIRCUIT_COLS_V2,
            weather_cols=WEATHER_COLS_V2,
            race_context_cols=RACE_CONTEXT_COLS_V2,
        )

        # Build 2014-2023 (train) + 2024 (test) in V4 format
        ctx_train, cand_train, win_train, gap_train, _ = builder_v4.build_sequences(2014, 2023)
        ctx_2024, cand_2024, win_2024, gap_2024, meta_2024 = builder_v4.build_sequences(2024, 2024)

        n_train = len(win_train)
        n_val = max(5, int(n_train * 0.15))

        data_dict = {
            "train_context": ctx_train[:n_train - n_val],
            "train_candidates": cand_train[:n_train - n_val],
            "train_winners": win_train[:n_train - n_val],
            "train_time_gaps": gap_train[:n_train - n_val],
            "val_context": ctx_train[n_train - n_val:],
            "val_candidates": cand_train[n_train - n_val:],
            "val_winners": win_train[n_train - n_val:],
            "val_time_gaps": gap_train[n_train - n_val:],
            "predict_2025_context": ctx_2024,
            "predict_2025_candidates": cand_2024,
            "predict_2025_winners": win_2024,
            "predict_2025_time_gaps": gap_2024,
            "predict_2025_meta": meta_2024,
        }
        data_dict = normalize_features_v2(data_dict)

        torch.save({
            "context": torch.from_numpy(data_dict["predict_2025_context"]),
            "candidates": torch.from_numpy(data_dict["predict_2025_candidates"]),
            "winners": torch.from_numpy(data_dict["predict_2025_winners"]),
            "time_gaps": torch.from_numpy(data_dict["predict_2025_time_gaps"]),
            "meta": torch.from_numpy(data_dict["predict_2025_meta"]),
        }, features_2024_v4_path)
        print(f"Saved V4 2024 features: {len(win_2024)} races")

    if not features_2024_path.exists():
        build_2024_features()

    # Load metadata (V2 metadata has driver_enc etc)
    with open(PROCESSED / "metadata_v2.pkl", "rb") as f:
        metadata = pickle.load(f)

    # ============ V2 Transformer ============
    print("\n--- V2 Transformer ---")
    model_v2 = F1WinnerTransformerV2(
        d_model=128, n_heads=4, n_encoder_layers=4, n_cross_attn_layers=3,
        d_ff=512, dropout=0.0,
        context_window=metadata["context_window"],
        num_drivers=metadata["num_drivers"],
        num_constructors=metadata["num_constructors"],
        num_circuits=metadata["num_circuits"],
        d_candidate_raw=metadata["d_candidate_raw"],
        d_context_raw=metadata["d_context_raw"],
    ).to(DEVICE)
    ckpt_v2 = torch.load(MODEL_DIR / "best_v2.pt", map_location="cpu", weights_only=False)
    model_v2.load_state_dict(ckpt_v2["model_state_dict"])
    model_v2.eval()

    acc_v2_2024, _ = predict_year(
        model_v2,
        features_2024_path, metadata, REAL_WINNERS_2024, "V2 on 2024", DEVICE
    )

    acc_v2_2025, _ = predict_year(
        model_v2,
        PROCESSED / "features_2025_v2.pt", metadata, REAL_WINNERS_2025, "V2 on 2025", DEVICE
    )

    # ============ MLP ============
    print("\n--- MLP ---")
    model_mlp = F1WinnerMLP(
        d_context_raw=metadata["d_context_raw"],
        context_window=metadata["context_window"],
        d_candidate_raw=metadata["d_candidate_raw"],
        num_drivers=metadata["num_drivers"],
        num_constructors=metadata["num_constructors"],
        num_circuits=metadata["num_circuits"],
    ).to(DEVICE)
    ckpt_mlp = torch.load(MODEL_DIR / "best_mlp.pt", map_location="cpu", weights_only=False)
    model_mlp.load_state_dict(ckpt_mlp["model_state_dict"])
    model_mlp.eval()

    acc_mlp_2024, _ = predict_year(
        model_mlp,
        features_2024_path, metadata, REAL_WINNERS_2024, "MLP on 2024", DEVICE
    )

    acc_mlp_2025, _ = predict_year(
        model_mlp,
        PROCESSED / "features_2025_v2.pt", metadata, REAL_WINNERS_2025, "MLP on 2025", DEVICE
    )

    # ============ V4 Grid-First ============
    print("\n--- V4 Grid-First (top-7) ---")
    model_v4 = F1WinnerTransformerV2(
        d_model=128, n_heads=4, n_encoder_layers=4, n_cross_attn_layers=3,
        d_ff=512, dropout=0.0,
        context_window=metadata["context_window"],
        num_drivers=metadata["num_drivers"],
        num_constructors=metadata["num_constructors"],
        num_circuits=metadata["num_circuits"],
        d_candidate_raw=metadata["d_candidate_raw"],
        d_context_raw=metadata["d_context_raw"],
        max_drivers_per_race=7,
    ).to(DEVICE)
    ckpt_v4 = torch.load(MODEL_DIR / "best_v4.pt", map_location="cpu", weights_only=False)
    model_v4.load_state_dict(ckpt_v4["model_state_dict"])
    model_v4.eval()

    acc_v4_2024, _ = predict_year(
        model_v4,
        features_2024_v4_path, metadata, REAL_WINNERS_2024, "V4 on 2024", DEVICE
    )

    acc_v4_2025, _ = predict_year(
        model_v4,
        PROCESSED / "features_2025_v4.pt", metadata, REAL_WINNERS_2025, "V4 on 2025", DEVICE
    )

    # ============ Summary ============
    print(f"\n{'='*70}")
    print(f"COMPARISON: 2024 vs 2025")
    print(f"{'='*70}")
    print(f"  {'Model':<20s} {'2024':>8s} {'2025':>8s} {'Delta':>8s}")
    print(f"  {'-'*44}")
    print(f"  {'V2 Transformer':<20s} {acc_v2_2024:>7.1%} {acc_v2_2025:>7.1%} {acc_v2_2024-acc_v2_2025:>+7.1%}")
    print(f"  {'MLP':<20s} {acc_mlp_2024:>7.1%} {acc_mlp_2025:>7.1%} {acc_mlp_2024-acc_mlp_2025:>+7.1%}")
    print(f"  {'V4 Grid-First':<20s} {acc_v4_2024:>7.1%} {acc_v4_2025:>7.1%} {acc_v4_2024-acc_v4_2025:>+7.1%}")
    print(f"  (= model trained on 2014-2024, evaluated on each year)")
    print(f"  V2/MLP: 20-class problem  |  V4: 7-class (top-7 grid)")


if __name__ == "__main__":
    main()
