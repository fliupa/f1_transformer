"""Post-hoc analysis and visualization of F1 predictions."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

if os.environ.get("COLAB"):
    OUTPUTS_DIR = Path("/content/drive/MyDrive/f1_transformer/outputs")


def load_predictions():
    """Load 2025 predictions."""
    path = OUTPUTS_DIR / "predictions" / "2025_predictions.csv"
    if not path.exists():
        print(f"No predictions found at {path}")
        return None
    return pd.read_csv(path)


def plot_win_probability_heatmap(df, save_path):
    """Plot heatmap of win probabilities per driver per race."""
    if df is None or df.empty:
        return

    # Pivot: rows=drivers, cols=races, values=win_probability
    pivot = df.pivot_table(
        index="driver", columns="round", values="win_probability", aggfunc="max"
    )

    plt.figure(figsize=(20, max(8, len(pivot) * 0.3)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "Win Probability"})
    plt.title("F1 2025 Season - Win Probability Heatmap", fontsize=16, fontweight="bold")
    plt.xlabel("Race Round", fontsize=12)
    plt.ylabel("Driver", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {save_path}")
    plt.close()


def plot_championship_projection(df, save_path):
    """Plot projected championship standings based on predictions."""
    if df is None or df.empty:
        return

    # Get all round-1 predictions (predicted winners for each race)
    winners = df[df["rank"] == 1].copy()
    if "year" in winners.columns:
        winners = winners.groupby(["year", "round"]).first().reset_index()

    win_counts = winners["driver"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(win_counts.index, win_counts.values, color=sns.color_palette("husl", len(win_counts)))
    ax.set_xlabel("Driver", fontsize=12)
    ax.set_ylabel("Predicted Wins", fontsize=12)
    ax.set_title("Projected 2025 Championship - Predicted Race Wins", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    # Add values on top of bars
    for bar, val in zip(bars, win_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Championship projection saved to {save_path}")
    plt.close()


def plot_top3_probabilities(df, save_path):
    """Plot top-3 probability distribution for key races."""
    if df is None or df.empty:
        return

    # Get unique races
    races = df.groupby(["round", "circuit"] if "circuit" in df.columns else "round").first().reset_index()
    n_races = min(6, len(races))  # Show first 6 races
    selected_rounds = races["round"].head(n_races).values

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, round_num in enumerate(selected_rounds):
        race_df = df[df["round"] == round_num].nlargest(8, "win_probability")

        colors = sns.color_palette("husl", len(race_df))
        axes[i].barh(race_df["driver"], race_df["win_probability"], color=colors)
        axes[i].set_title(f"Round {int(round_num)}")
        axes[i].set_xlabel("Win Probability")
        axes[i].set_xlim(0, 1)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("F1 2025 - Top Win Probabilities by Race", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Top-3 probabilities saved to {save_path}")
    plt.close()


def generate_report(df):
    """Generate a text report of predictions."""
    if df is None or df.empty:
        return

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("F1 2025 SEASON PREDICTION REPORT")
    report_lines.append("=" * 60)

    # Overall favorite
    winners = df[df["rank"] == 1]
    win_counts = winners["driver"].value_counts()
    report_lines.append(f"\nCHAMPIONSHIP FAVORITE: {win_counts.index[0]} ({win_counts.iloc[0]} wins)")

    # Top 5 drivers
    report_lines.append("\nTOP 5 DRIVERS BY PREDICTED WINS:")
    for driver, wins in win_counts.head(5).items():
        report_lines.append(f"  {driver}: {wins} wins")

    # Most competitive races (lowest top probability -> most uncertainty)
    race_probs = winners.groupby("round").agg({"win_probability": "max"}).sort_values("win_probability")
    report_lines.append("\nMOST COMPETITIVE RACES (lowest confidence):")
    for i, (round_num, row) in enumerate(race_probs.head(3).iterrows()):
        winner = winners[winners["round"] == round_num]["driver"].values[0]
        report_lines.append(f"  Round {int(round_num)}: {winner} ({row['win_probability']:.1%} confidence)")

    # Least competitive races
    report_lines.append("\nLEAST COMPETITIVE RACES (highest confidence):")
    for i, (round_num, row) in enumerate(race_probs.tail(3).iterrows()):
        winner = winners[winners["round"] == round_num]["driver"].values[0]
        report_lines.append(f"  Round {int(round_num)}: {winner} ({row['win_probability']:.1%} confidence)")

    report_lines.append("\n" + "=" * 60)
    report_lines.append("FULL PREDICTED RACE WINNERS:")
    report_lines.append("=" * 60)
    for _, row in winners.iterrows():
        report_lines.append(
            f"  Round {int(row['round']):2d}: {row['driver']:<5s} ({row['win_probability']:.1%})"
        )

    return "\n".join(report_lines)


def main():
    """Run analysis and generate all plots/reports."""
    print("=" * 60)
    print("F1 2025 PREDICTION ANALYSIS")
    print("=" * 60)

    df = load_predictions()
    if df is None:
        return

    plots_dir = OUTPUTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating plots...")
    plot_win_probability_heatmap(df, plots_dir / "win_probability_heatmap.png")
    plot_championship_projection(df, plots_dir / "championship_projection.png")
    plot_top3_probabilities(df, plots_dir / "top3_probabilities.png")

    # Generate report
    report = generate_report(df)
    report_path = OUTPUTS_DIR / "predictions" / "prediction_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
