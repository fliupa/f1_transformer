"""Feature engineering V2: ELO ratings, momentum, track affinity + context stats.

Adds 7 new features and improves context vectors with distribution statistics.
Fixes dnf_rate duplication bug from V1.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
if os.environ.get("COLAB"):
    RAW_DATA = Path("/content/drive/MyDrive/f1_transformer/data/raw")
    PROCESSED_DATA = Path("/content/drive/MyDrive/f1_transformer/data/processed")


def load_race_data() -> pd.DataFrame:
    """Load raw race results."""
    from .feature_engineering import load_race_data as _load
    return _load()


def load_circuit_data() -> pd.DataFrame:
    """Load static circuit data."""
    from .feature_engineering import load_circuit_data as _load
    return _load()


def load_weather_data() -> Optional[pd.DataFrame]:
    """Load weather data."""
    from .feature_engineering import load_weather_data as _load
    return _load()


def compute_elo_ratings(df: pd.DataFrame, k_factor: float = 32.0) -> pd.DataFrame:
    """Compute driver and constructor ELO ratings over time.

    Each driver starts at 1500. After each race, ratings update based on
    finishing position vs expected position (from grid).

    Constructor ELO is the average of its two drivers' ELO ratings.

    Modified to handle races without all drivers present (season boundaries).
    """
    df = df.sort_values(["year", "round"]).reset_index(drop=True)
    driver_elo = {}
    constructor_elo = {}

    df["elo_rating"] = 1500.0
    df["constructor_elo_rating"] = 1500.0

    # Process race by race
    race_groups = df.groupby(["year", "round"])
    for (year, round_num), race_idx in race_groups.indices.items():
        race_df = df.loc[race_idx].sort_values("grid_position")

        # Assign current ELO to each driver before the race
        for idx in race_idx:
            driver = df.loc[idx, "driver_abbreviation"]
            constructor = df.loc[idx, "constructor"]
            df.loc[idx, "elo_rating"] = driver_elo.get(driver, 1500.0)
            df.loc[idx, "constructor_elo_rating"] = constructor_elo.get(constructor, 1500.0)

        # Update ELO after the race
        drivers_in_race = len(race_df)
        for _, row in race_df.iterrows():
            driver = row["driver_abbreviation"]
            constructor = row["constructor"]
            finish_pos = row["finish_position"]
            grid_pos = row["grid_position"]

            # Expected position is grid position (normalized)
            expected_rank = min(grid_pos, drivers_in_race) / max(drivers_in_race, 1)
            actual_rank = finish_pos / max(drivers_in_race, 1)

            # Score: 1.0 for win, 0.0 for last
            score = 1.0 - (finish_pos - 1) / max(drivers_in_race - 1, 1)
            expected_score = 1.0 - (grid_pos - 1) / max(drivers_in_race - 1, 1)

            # ELO update
            old_elo = driver_elo.get(driver, 1500.0)
            delta = k_factor * (score - expected_score) * (1.0 + abs(grid_pos - finish_pos) / 20.0)
            driver_elo[driver] = old_elo + delta

        # Update constructor ELO (average of both drivers)
        for constructor in race_df["constructor"].unique():
            const_drivers = race_df[race_df["constructor"] == constructor]["driver_abbreviation"].unique()
            const_elos = [driver_elo.get(d, 1500.0) for d in const_drivers]
            constructor_elo[constructor] = np.mean(const_elos)

    return df


def compute_form_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Compute form trajectory: difference between short-term and medium-term form.

    form_trajectory = avg_finish_last_3 - avg_finish_last_5
    Positive = improving (recent results better than medium-term)
    Negative = declining
    """
    df["form_trajectory"] = df["avg_finish_last_3"] - df["avg_finish_last_5"]
    return df


def compute_track_affinity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute driver's track-specific performance as percentile rank.

    For each driver-circuit pair, compute percentile of that driver's performance
    at that circuit relative to all other drivers at that circuit.
    Higher = better relative to peers at this track.
    """
    df["track_affinity_score"] = 50.0  # default (median percentile)

    for circuit in df["circuit"].unique():
        circuit_mask = df["circuit"] == circuit
        if circuit_mask.sum() < 5:
            continue

        # For each driver at this circuit, compute avg finish vs global avg
        global_avg = df.loc[circuit_mask, "finish_position"].mean()
        driver_avgs = df.loc[circuit_mask].groupby("driver_abbreviation")["finish_position"].mean()

        # Rank: lower avg finish = better = higher percentile
        ranks = driver_avgs.rank(pct=True) * 100  # 0-100 percentile

        for driver, percentile in ranks.items():
            driver_circuit_mask = (df["circuit"] == circuit) & (df["driver_abbreviation"] == driver)
            df.loc[driver_circuit_mask, "track_affinity_score"] = percentile

    return df


def compute_recent_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Compute exponentially weighted recent momentum.

    Weighted avg of last 3 races with decay: 0.7 * race_n + 0.2 * race_n-1 + 0.1 * race_n-2
    Lower = better recent finishes.
    """
    df = df.sort_values(["year", "round"]).reset_index(drop=True)
    df["recent_momentum"] = 10.0

    for driver in df["driver_abbreviation"].unique():
        driver_mask = df["driver_abbreviation"] == driver
        driver_idx = df[driver_mask].index
        finish_vals = df.loc[driver_mask, "finish_position"].values

        for i, idx in enumerate(driver_idx):
            if i >= 3:
                df.loc[idx, "recent_momentum"] = (
                    0.7 * finish_vals[i-1] + 0.2 * finish_vals[i-2] + 0.1 * finish_vals[i-3]
                )
            elif i >= 2:
                df.loc[idx, "recent_momentum"] = (
                    0.7 * finish_vals[i-1] + 0.3 * finish_vals[i-2]
                )
            elif i >= 1:
                df.loc[idx, "recent_momentum"] = float(finish_vals[i-1])

    return df


def compute_constructor_track_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Compute constructor's historical average finish at each circuit.

    For each constructor-circuit pair, compute average finish position
    across both drivers in all previous races at this circuit.
    Defaults to global constructor avg if no track history.
    """
    df = df.sort_values(["year", "round"]).reset_index(drop=True)
    df["constructor_track_avg"] = 10.0

    # Compute global constructor averages as fallback
    constructor_global_avg = df.groupby("constructor")["finish_position"].mean()

    for constructor in df["constructor"].unique():
        for circuit in df["circuit"].unique():
            mask = (df["constructor"] == constructor) & (df["circuit"] == circuit)
            if mask.sum() > 2:
                avg = df.loc[mask, "finish_position"].mean()
                df.loc[mask, "constructor_track_avg"] = avg
            else:
                df.loc[df["constructor"] == constructor, "constructor_track_avg"] = (
                    df.loc[df["constructor"] == constructor, "constructor_track_avg"]
                    .fillna(constructor_global_avg.get(constructor, 10.0))
                )

    return df


def compute_grid_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical positions gained/lost from grid position.

    Average of (grid_position - finish_position) over a driver's history.
    Positive = gains positions during races. Negative = loses positions.
    """
    df["grid_to_finish_delta"] = 0.0
    for driver in df["driver_abbreviation"].unique():
        driver_mask = df["driver_abbreviation"] == driver
        driver_idx = df[driver_mask].index
        grid_vals = df.loc[driver_mask, "grid_position"].values
        finish_vals = df.loc[driver_mask, "finish_position"].values

        for i, idx in enumerate(driver_idx):
            if i > 0:
                df.loc[idx, "grid_to_finish_delta"] = np.mean(grid_vals[:i] - finish_vals[:i])

    return df


def build_all_features_v2(
    df: Optional[pd.DataFrame] = None,
    circuits_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build all V2 features on top of V1 base features.

    Extends the V1 feature set with:
    - elo_rating, constructor_elo_rating
    - form_trajectory
    - track_affinity_score
    - recent_momentum
    - constructor_track_avg
    - grid_to_finish_delta
    """
    from .feature_engineering import build_all_features as build_v1

    if df is None:
        df = load_race_data()

    # 1. Build V1 base features
    featured_df = build_v1(df, circuits_df, weather_df)

    # 2. Add V2 features
    print("\n--- V2: ELO Ratings ---")
    featured_df = compute_elo_ratings(featured_df)

    print("--- V2: Form Trajectory ---")
    featured_df = compute_form_trajectory(featured_df)

    print("--- V2: Track Affinity ---")
    featured_df = compute_track_affinity(featured_df)

    print("--- V2: Recent Momentum ---")
    featured_df = compute_recent_momentum(featured_df)

    print("--- V2: Constructor Track Avg ---")
    featured_df = compute_constructor_track_avg(featured_df)

    print("--- V2: Grid Delta ---")
    featured_df = compute_grid_delta(featured_df)

    print(f"Feature engineering V2 complete: {len(featured_df)} rows, {len(featured_df.columns)} columns")
    return featured_df


# ─── Updated Feature Column Lists (V2: no dnf_rate duplicate, + new features) ───

NUMERIC_DRIVER_COLS_V2 = [
    "grid_position", "q_position", "avg_finish_last_3", "total_points_last_3",
    "avg_finish_last_5",
    # NOTE: dnf_rate moved to constructor only (was duplicated in V1)
    "season_points_driver", "championship_position", "points_gap_to_leader",
    "track_history_score",
    # ─── V2 NEW ───
    "elo_rating", "form_trajectory", "track_affinity_score",
    "recent_momentum", "grid_to_finish_delta",
]

CONSTRUCTOR_COLS_V2 = [
    "constructor_season_points", "constructor_avg_finish_last_3",
    "constructor_points_last_3", "dnf_rate",
    # ─── V2 NEW ───
    "constructor_elo_rating", "constructor_track_avg",
]

# Circuit, weather, race_context stay the same
CIRCUIT_COLS_V2 = [
    "length_km", "corners", "drs_zones", "altitude_m", "track_type_code",
    "downforce_code", "tyre_degradation_code", "overtaking_code",
]

WEATHER_COLS_V2 = [
    "openmeteo_temp_mean", "openmeteo_humidity", "openmeteo_precip_mm",
    "openmeteo_wind_speed", "openmeteo_rain_flag", "openmeteo_pressure",
]

RACE_CONTEXT_COLS_V2 = ["round"]


def get_feature_vector_size_v2(context_window: int = 10) -> tuple:
    """Calculate V2 feature vector sizes.

    Returns:
        (d_candidate_raw, d_context_raw)
    """
    n_numeric = len(NUMERIC_DRIVER_COLS_V2)
    n_constructor = len(CONSTRUCTOR_COLS_V2)
    n_circuit = len(CIRCUIT_COLS_V2)
    n_weather = len(WEATHER_COLS_V2)
    n_race = len(RACE_CONTEXT_COLS_V2)

    # Per-driver: 3 embedding indices + numeric + constructor + circuit + weather + race
    d_candidate = 3 + n_numeric + n_constructor + n_circuit + n_weather + n_race

    # Per-race context: mean+std+max of driver numerics + constructor + circuit + weather + race + top3 + winner
    # V2 uses 3x driver numeric features (mean, std, max) instead of just mean
    d_context = (
        n_numeric * 3 +      # mean + std + max of driver features
        n_constructor * 3 +  # mean + std + max of constructor features
        n_circuit +          # circuit (same for all drivers in race)
        n_weather +          # weather (same for all drivers in race)
        n_race +             # round
        3 * 16 +             # top-3 finisher driver embedding dims (16 each)
        16                   # winner driver embedding dim
    )

    return d_candidate, d_context
