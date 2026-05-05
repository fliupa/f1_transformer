"""Prediction pipeline for 2025 F1 season.

Uses the combined 2014-2024 trained model to predict winners for each 2025 race.
Compares predictions against real 2025 results.
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL = MODELS_DIR / "final" / "best.pt"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
if os.environ.get("COLAB"):
    PROJECT_ROOT = Path("/content/drive/MyDrive/f1_transformer")
    PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    BEST_MODEL = MODELS_DIR / "final" / "best.pt"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(PROJECT_ROOT))
from src.model.transformer_model import F1WinnerTransformer
from src.model.transformer_model_v2 import F1WinnerTransformerV2
from scripts.train_mlp import F1WinnerMLP

# Real 2025 winners
REAL_WINNERS_2025 = {
    1: 'NOR', 2: 'PIA', 3: 'VER', 4: 'RUS', 5: 'VER', 6: 'NOR',
    7: 'PIA', 8: 'NOR', 9: 'VER', 10: 'NOR', 11: 'PIA', 12: 'RUS',
    13: 'PIA', 14: 'VER', 15: 'VER', 16: 'NOR', 17: 'PIA', 18: 'VER',
    19: 'PIA', 20: 'VER', 21: 'PIA', 22: 'VER', 23: 'NOR', 24: 'NOR',
}


def load_model_and_metadata(device="cpu", model_version="v1"):
    """Load trained model and metadata.

    Args:
        device: torch device
        model_version: "v1", "v2", or "v2_recent"
    """
    version_map = {
        "v1": ("", F1WinnerTransformer),
        "v2": ("_v2", F1WinnerTransformerV2),
        "v2_recent": ("_v2_recent", F1WinnerTransformerV2),
        "v3": ("_v3", F1WinnerTransformerV2),
        "mlp": ("_v2", F1WinnerMLP),  # MLP uses V2 data
    }
    suffix, ModelClass = version_map.get(model_version, ("", F1WinnerTransformer))
    metadata_path = PROCESSED_DATA / f"metadata{suffix}.pkl"
    if model_version == "mlp":
        model_path = MODELS_DIR / "final" / "best_mlp.pt"
    else:
        model_path = MODELS_DIR / "final" / f"best{suffix}.pt"

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    if ModelClass == F1WinnerMLP:
        model = F1WinnerMLP(
            d_context_raw=metadata["d_context_raw"],
            context_window=metadata["context_window"],
            d_candidate_raw=metadata["d_candidate_raw"],
            num_drivers=metadata["num_drivers"],
            num_constructors=metadata["num_constructors"],
            num_circuits=metadata["num_circuits"],
        )
    elif ModelClass == F1WinnerTransformerV2:
        # V2 models: read architecture params from metadata or use defaults
        v2_configs = {
            "v2": dict(d_model=128, n_heads=4, n_encoder_layers=4, n_cross_attn_layers=3, d_ff=512, dropout=0.25),
            "v2_recent": dict(d_model=96, n_heads=3, n_encoder_layers=2, n_cross_attn_layers=2, d_ff=384, dropout=0.30),
            "v3": dict(d_model=128, n_heads=4, n_encoder_layers=4, n_cross_attn_layers=3, d_ff=512, dropout=0.30),
        }
        cfg = v2_configs.get(model_version, v2_configs["v2"])
        model = F1WinnerTransformerV2(
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_encoder_layers=cfg["n_encoder_layers"],
            n_cross_attn_layers=cfg["n_cross_attn_layers"],
            d_ff=cfg["d_ff"], dropout=cfg["dropout"],
            context_window=metadata["context_window"],
            num_drivers=metadata["num_drivers"],
            num_constructors=metadata["num_constructors"],
            num_circuits=metadata["num_circuits"],
            d_candidate_raw=metadata["d_candidate_raw"],
            d_context_raw=metadata["d_context_raw"],
        )
    else:
        model = F1WinnerTransformer(
            d_model=192, n_heads=6, n_encoder_layers=3, n_cross_attn_layers=2,
            d_ff=768, dropout=0.15, context_window=metadata["context_window"],
            num_drivers=metadata["num_drivers"],
            num_constructors=metadata["num_constructors"],
            num_circuits=metadata["num_circuits"],
            d_candidate_raw=metadata["d_candidate_raw"],
            d_context_raw=metadata["d_context_raw"],
        )

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded {model_version} model (val_acc={checkpoint.get('best_val_acc', 0):.4f})")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    model = model.to(device)
    model.eval()
    return model, metadata


def predict_2025_season(model, metadata, device, model_version="v1"):
    """Predict winners for all 2025 races."""
    version_suffix_map = {"v1": "", "v2": "_v2", "v2_recent": "_v2_recent", "v3": "_v3", "mlp": "_v2"}
    suffix = version_suffix_map.get(model_version, "")
    data_2025_path = PROCESSED_DATA / f"features_2025{suffix}.pt"
    if not data_2025_path.exists():
        raise FileNotFoundError(f"2025 data not found at {data_2025_path}")

    data_2025 = torch.load(data_2025_path, weights_only=False)
    context = data_2025["context"]
    candidates = data_2025["candidates"]
    time_gaps = data_2025.get("time_gaps", torch.zeros(len(context), metadata["context_window"]))
    driver_enc = metadata["driver_encoder"]

    all_predictions = []
    summary = []
    correct = 0

    with torch.no_grad():
        for i in range(len(context)):
            ctx = context[i:i+1].to(device)
            cand = candidates[i:i+1].to(device)
            gaps = time_gaps[i:i+1].to(device)

            logits = model(ctx, cand, gaps)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

            driver_indices = candidates[i, :, 0].long().numpy()
            driver_names = []
            for idx in driver_indices:
                try:
                    driver_names.append(driver_enc.decode(int(idx)))
                except:
                    driver_names.append(f"UNK_{idx}")

            sorted_idx = np.argsort(probs)[::-1]
            pred = driver_names[sorted_idx[0]]
            round_num = i + 1
            real = REAL_WINNERS_2025.get(round_num, "?")
            match = pred == real
            if match:
                correct += 1

            summary.append({
                "round": round_num, "predicted": pred, "actual": real,
                "correct": match, "confidence": float(probs[sorted_idx[0]]),
                "top3": [driver_names[idx] for idx in sorted_idx[:3]],
            })

            for rank, idx in enumerate(sorted_idx):
                name = driver_names[idx]
                if name.startswith("UNK"):
                    continue
                all_predictions.append({
                    "year": 2025, "round": round_num, "rank": rank + 1,
                    "driver": name, "win_probability": float(probs[idx]),
                })

    pred_df = pd.DataFrame(all_predictions)
    summary_df = pd.DataFrame(summary)
    return pred_df, summary_df, correct


def main(model_version="v1"):
    device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device} | Model: {model_version}")

    model, metadata = load_model_and_metadata(device, model_version)
    pred_df, summary_df, correct = predict_2025_season(model, metadata, device, model_version)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "predictions").mkdir(exist_ok=True)

    pred_df.to_csv(OUTPUTS_DIR / "predictions" / "2025_predictions.csv", index=False)
    summary_df.to_csv(OUTPUTS_DIR / "predictions" / "2025_summary.csv", index=False)

    n_races = len(summary_df)
    print(f"\n{'='*60}")
    print(f"2025 RESULTS: {correct}/{n_races} = {correct/n_races*100:.1f}%")
    print(f"{'='*60}")

    # Show predictions vs real
    print(f"\n{'R':>3s} {'Pred':<6s} {'Real':<6s} {'Conf':>6s} {'Match'}")
    print("-" * 40)
    for _, row in summary_df.iterrows():
        print(f'{int(row["round"]):3d} {row["predicted"]:<6s} {row["actual"]:<6s} '
              f'{row["confidence"]:6.1%} {"YES" if row["correct"] else "NO"}')

    # Win counts comparison
    pred_wins = pred_df[pred_df["rank"] == 1]["driver"].value_counts()
    real_wins = pd.Series(REAL_WINNERS_2025).value_counts()

    print(f"\n{'Driver':<8s} {'Pred':>5s} {'Real':>5s}")
    print("-" * 20)
    all_drivers = sorted(set(list(pred_wins.index) + list(real_wins.index)),
                        key=lambda x: real_wins.get(x, 0), reverse=True)
    for d in all_drivers:
        print(f'{d:<8s} {pred_wins.get(d, 0):>5d} {real_wins.get(d, 0):>5d}')

    # Report
    report = [
        "=" * 60,
        "F1 2025 SEASON PREDICTION REPORT",
        f"Accuracy: {correct}/{n_races} = {correct/n_races*100:.1f}%",
        f"Model: Combined 2014-2024 training",
        "=" * 60,
    ]
    with open(OUTPUTS_DIR / "predictions" / "prediction_report.txt", "w") as f:
        f.write("\n".join(report))

    print(f"\nFiles saved to {OUTPUTS_DIR / 'predictions'}")


if __name__ == "__main__":
    main()
