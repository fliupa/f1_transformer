"""Feature engineering for F1 race prediction.

Computes per-driver, per-race features including:
- Driver form (recent results)
- Constructor form
- Track history
- Season standings
- Race context features (weather, circuit)
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
    """Load raw race results, trying multiple formats."""
    paths = [
        RAW_DATA / "races" / "race_results_all.csv",
        RAW_DATA / "races" / "race_results_all.pkl",
    ]
    for p in paths:
        if p.exists():
            if p.suffix == ".pkl":
                df = pd.read_pickle(p)
            else:
                df = pd.read_csv(p)
            if "event_date" in df.columns:
                df["event_date"] = pd.to_datetime(df["event_date"], format='mixed')
            print(f"Loaded {len(df)} race results from {p}")
            return df
    raise FileNotFoundError("No race results found. Run fetch_fastf1.py first.")


def load_circuit_data() -> pd.DataFrame:
    """Load static circuit data."""
    paths = [
        RAW_DATA / "circuits" / "circuits.csv",
        RAW_DATA / "circuits" / "circuits.pkl",
    ]
    for p in paths:
        if p.exists():
            if p.suffix == ".pkl":
                return pd.read_pickle(p)
            return pd.read_csv(p)
    print("No circuit data found, using embedded defaults.")
    return None


def load_weather_data() -> Optional[pd.DataFrame]:
    """Load weather data."""
    path = RAW_DATA / "weather" / "weather_all.csv"
    if path.exists():
        df = pd.read_csv(path)
        if "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(df["event_date"], format='mixed')
        print(f"Loaded {len(df)} weather records")
        return df
    return None


def compute_driver_form(df: pd.DataFrame) -> pd.DataFrame:
    """Compute driver form features.

    For each race, compute:
    - avg_finish_last_3: Average finishing position in last 3 races
    - total_points_last_3: Total points in last 3 races
    - avg_finish_last_5: Average finishing position in last 5 races
    - dnf_rate: DNF rate in last 10 races
    - season_points_so_far: Cumulative points this season
    - season_position: Championship position before this race
    """
    df = df.sort_values(["year", "round", "grid_position"])
    df = df.reset_index(drop=True)

    drivers = df["driver_abbreviation"].unique()

    for driver in drivers:
        mask = df["driver_abbreviation"] == driver
        driver_idx = df[mask].index

        # Season cumulative points
        df.loc[mask, "season_points_driver"] = (
            df.loc[mask, "points"].cumsum() - df.loc[mask, "points"]
        )

        # Rolling averages
        finish_vals = df.loc[mask, "finish_position"].values
        points_vals = df.loc[mask, "points"].values
        dnf_vals = df.loc[mask, "dnf"].values

        for i, idx in enumerate(driver_idx):
            # Last 3
            if i >= 1:
                start3 = max(0, i - 3)
                df.loc[idx, "avg_finish_last_3"] = np.mean(finish_vals[start3:i])
                df.loc[idx, "total_points_last_3"] = np.sum(points_vals[start3:i])
            else:
                df.loc[idx, "avg_finish_last_3"] = 10.0
                df.loc[idx, "total_points_last_3"] = 0.0

            # Last 5
            if i >= 1:
                start5 = max(0, i - 5)
                df.loc[idx, "avg_finish_last_5"] = np.mean(finish_vals[start5:i])
            else:
                df.loc[idx, "avg_finish_last_5"] = 10.0

            # DNF rate last 10
            if i >= 1:
                start10 = max(0, i - 10)
                df.loc[idx, "dnf_rate"] = np.mean(dnf_vals[start10:i])
            else:
                df.loc[idx, "dnf_rate"] = 0.0

    return df


def compute_constructor_form(df: pd.DataFrame) -> pd.DataFrame:
    """Compute constructor form features.

    For each race compute:
    - constructor_avg_finish_last_3: Average finish of both drivers last 3 races
    - constructor_points_last_3: Total constructor points last 3 races
    - constructor_season_points: Cumulative constructor points this season
    """
    df = df.sort_values(["year", "round", "constructor"])
    df = df.reset_index(drop=True)

    constructors = df["constructor"].unique()
    constructor_season_points = {}

    for const in constructors:
        const_mask = df["constructor"] == const
        const_idx = df[const_mask].index

        # Compute per-race constructor points (sum of both drivers)
        for idx in const_idx:
            race_key = (df.loc[idx, "year"], df.loc[idx, "round"])
            race_mask = (df["year"] == race_key[0]) & (df["round"] == race_key[1]) & (df["constructor"] == const)
            race_points = df.loc[race_mask, "points"].sum()

            if const not in constructor_season_points:
                constructor_season_points[const] = {}
            prev = constructor_season_points[const].get("running_total", 0)
            df.loc[idx, "constructor_season_points"] = prev
            constructor_season_points[const]["running_total"] = prev + race_points

        # Rolling averages
        finish_vals = df.loc[const_mask, "finish_position"].values
        points_vals = df.loc[const_mask, "points"].values

        for i, idx in enumerate(const_idx):
            if i >= 1:
                start3 = max(0, i - 3)
                df.loc[idx, "constructor_avg_finish_last_3"] = np.mean(finish_vals[start3:i])
                df.loc[idx, "constructor_points_last_3"] = np.sum(points_vals[start3:i])
            else:
                df.loc[idx, "constructor_avg_finish_last_3"] = 10.0
                df.loc[idx, "constructor_points_last_3"] = 0.0

    return df


def compute_track_history(df: pd.DataFrame) -> pd.DataFrame:
    """Compute driver historical performance at each circuit.

    track_history_score: average finishing position at this circuit (all years before this race).
    """
    df = df.sort_values(["year", "round"])
    df = df.reset_index(drop=True)
    df["track_history_score"] = np.nan

    # Group by driver and circuit
    for (driver, circuit), group in df.groupby(["driver_abbreviation", "circuit"]):
        group = group.sort_values(["year", "round"])
        finish_vals = group["finish_position"].values
        indices = group.index

        for i, idx in enumerate(indices):
            if i > 0:
                df.loc[idx, "track_history_score"] = np.mean(finish_vals[:i])
            else:
                df.loc[idx, "track_history_score"] = 10.0  # No history

    return df


def compute_championship_position(df: pd.DataFrame) -> pd.DataFrame:
    """Compute driver championship position before each race."""
    df = df.sort_values(["year", "round"])
    df["championship_position"] = 0

    for year in df["year"].unique():
        year_df = df[df["year"] == year].sort_values(["round"])
        driver_points = {}

        for idx, row in year_df.iterrows():
            driver = row["driver_abbreviation"]
            # Assign rank based on current points
            current_points = driver_points.copy()
            current_points[driver] = row["season_points_driver"]
            ranked = sorted(current_points.items(), key=lambda x: x[1], reverse=True)
            rank = next(i + 1 for i, (d, _) in enumerate(ranked) if d == driver)
            df.loc[idx, "championship_position"] = rank

            # Update points
            driver_points[driver] = current_points.get(driver, 0) + row["season_points_driver"]

    df["points_gap_to_leader"] = 0.0
    for year in df["year"].unique():
        year_df = df[df["year"] == year].sort_values(["round"])
        for round_num in year_df["round"].unique():
            round_mask = (df["year"] == year) & (df["round"] == round_num)
            max_points = df.loc[round_mask, "season_points_driver"].max()
            df.loc[round_mask, "points_gap_to_leader"] = max_points - df.loc[round_mask, "season_points_driver"]

    return df


def merge_circuit_features(df: pd.DataFrame, circuits_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Merge static circuit features into race data."""
    if circuits_df is None:
        circuits_df = load_circuit_data()

    if circuits_df is None:
        # Use default values
        df["length_km"] = 5.0
        df["corners"] = 16
        df["drs_zones"] = 2
        df["altitude_m"] = 100
        df["track_type_code"] = 2  # permanent
        df["downforce_code"] = 1   # medium
        df["tyre_degradation_code"] = 1  # medium
        df["overtaking_code"] = 1  # medium
        df["lap_record_s"] = 85.0
        return df

    # Try exact match first, then fuzzy
    circuit_map = circuits_df.set_index("circuit_name").to_dict("index")

    for col in ["length_km", "corners", "drs_zones", "altitude_m", "track_type_code",
                "downforce_code", "tyre_degradation_code", "overtaking_code", "lap_record_s"]:
        df[col] = np.nan

    for idx, row in df.iterrows():
        circuit = row["circuit"]
        if circuit in circuit_map:
            info = circuit_map[circuit]
        else:
            # Fuzzy match
            found = False
            for name, info_data in circuit_map.items():
                if circuit and name.lower() in circuit.lower():
                    info = info_data
                    found = True
                    break
            if not found:
                continue

        for col in info:
            if col in df.columns:
                df.loc[idx, col] = info[col]

    # Fill NaN with medians
    for col in ["length_km", "corners", "drs_zones", "altitude_m",
                "track_type_code", "downforce_code", "tyre_degradation_code", "overtaking_code"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def merge_weather_features(df: pd.DataFrame, weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Merge weather features into race data."""
    if weather_df is None:
        weather_df = load_weather_data()

    weather_cols = ["openmeteo_temp_mean", "openmeteo_humidity", "openmeteo_precip_mm",
                    "openmeteo_wind_speed", "openmeteo_rain_flag", "openmeteo_pressure"]

    for col in weather_cols:
        df[col] = np.nan

    if weather_df is not None:
        for idx, row in df.iterrows():
            match = weather_df[
                (weather_df["year"] == row["year"]) &
                (weather_df["circuit"] == row["circuit"])
            ]
            if not match.empty:
                # Try exact circuit match first, then fuzzy
                for col in weather_cols:
                    if col in match.columns:
                        df.loc[idx, col] = match[col].values[0]

        # Fill NaN with medians
        for col in weather_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    # Default weather if no data
    if df["openmeteo_temp_mean"].isna().all():
        df["openmeteo_temp_mean"] = 25.0
        df["openmeteo_humidity"] = 55.0
        df["openmeteo_precip_mm"] = 0.0
        df["openmeteo_wind_speed"] = 12.0
        df["openmeteo_rain_flag"] = 0
        df["openmeteo_pressure"] = 1013.0

    return df


def build_all_features(
    df: Optional[pd.DataFrame] = None,
    circuits_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build all features for every driver-race combination.

    Returns DataFrame with all engineered features.
    """
    if df is None:
        df = load_race_data()

    print("Computing driver form features...")
    df = compute_driver_form(df)

    print("Computing constructor form features...")
    df = compute_constructor_form(df)

    print("Computing track history features...")
    df = compute_track_history(df)

    print("Computing championship position...")
    df = compute_championship_position(df)

    print("Merging circuit features...")
    df = merge_circuit_features(df, circuits_df)

    print("Merging weather features...")
    df = merge_weather_features(df, weather_df)

    print(f"Feature engineering complete: {len(df)} rows, {len(df.columns)} columns")
    return df


def get_feature_columns() -> dict:
    """Return lists of feature column names by category."""
    numeric_driver_features = [
        "grid_position", "q_position", "avg_finish_last_3", "total_points_last_3",
        "avg_finish_last_5", "dnf_rate", "season_points_driver", "championship_position",
        "points_gap_to_leader", "track_history_score",
    ]
    constructor_features = [
        "constructor_season_points", "constructor_avg_finish_last_3", "constructor_points_last_3",
        "dnf_rate",
    ]
    circuit_features = [
        "length_km", "corners", "drs_zones", "altitude_m", "track_type_code",
        "downforce_code", "tyre_degradation_code", "overtaking_code",
    ]
    weather_features = [
        "openmeteo_temp_mean", "openmeteo_humidity", "openmeteo_precip_mm",
        "openmeteo_wind_speed", "openmeteo_rain_flag", "openmeteo_pressure",
    ]
    race_context_features = [
        "round",
    ]

    return {
        "numeric_driver": numeric_driver_features,
        "constructor": constructor_features,
        "circuit": circuit_features,
        "weather": weather_features,
        "race_context": race_context_features,
        "all_numeric": numeric_driver_features + constructor_features + circuit_features + weather_features + race_context_features,
    }


def main():
    """Run full feature engineering pipeline."""
    df = load_race_data()
    circuits_df = load_circuit_data()
    weather_df = load_weather_data()

    featured_df = build_all_features(df, circuits_df, weather_df)

    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    featured_df.to_pickle(PROCESSED_DATA / "features_flat.pkl")
    print(f"Saved {len(featured_df)} rows to {PROCESSED_DATA / 'features_flat.pkl'}")

    # Print summary
    print("\nFeature Summary:")
    print(f"  Total rows: {len(featured_df)}")
    print(f"  Total columns: {len(featured_df.columns)}")
    print(f"  Seasons: {featured_df['year'].unique().tolist()}")
    print(f"  Drivers: {featured_df['driver_abbreviation'].nunique()}")
    print(f"  Circuits: {featured_df['circuit'].nunique()}")


if __name__ == "__main__":
    main()
