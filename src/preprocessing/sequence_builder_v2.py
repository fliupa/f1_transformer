"""Sequence builder V2: extended context vectors with distribution stats.

V2 improvements over V1:
- Context includes std and max in addition to mean of driver features
- Uses V2 feature column lists (no dnf_rate duplicate, + ELO, momentum, etc.)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
import os

from .encoders import DriverEncoder, ConstructorEncoder, CircuitEncoder
from .sequence_builder import SequenceBuilder, FeatureConfig


@dataclass
class FeatureConfigV2(FeatureConfig):
    """Extended config with V2 column lists."""
    numeric_driver_cols: Optional[List[str]] = None
    constructor_cols: Optional[List[str]] = None
    circuit_cols: Optional[List[str]] = None
    weather_cols: Optional[List[str]] = None
    race_context_cols: Optional[List[str]] = None
    # V2: include std and max in context
    include_std_max: bool = True
    # Override parent defaults to match V2 column counts
    num_numeric_driver: int = 14
    num_constructor_feat: int = 6


class SequenceBuilderV2(SequenceBuilder):
    """Extended sequence builder with distribution stats in context.

    Context now includes:
    - mean (original V1 behavior)
    - std (NEW: captures performance spread across drivers)
    - max (NEW: captures the best driver's stats)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        driver_enc: DriverEncoder,
        constructor_enc: ConstructorEncoder,
        circuit_enc: CircuitEncoder,
        config: Optional[FeatureConfigV2] = None,
        numeric_driver_cols: Optional[List[str]] = None,
        constructor_cols: Optional[List[str]] = None,
        circuit_cols: Optional[List[str]] = None,
        weather_cols: Optional[List[str]] = None,
        race_context_cols: Optional[List[str]] = None,
    ):
        self.config = config or FeatureConfigV2()

        # Use custom column lists if provided, else fall back to SequenceBuilder defaults
        self.numeric_driver_cols = numeric_driver_cols or [
            "grid_position", "q_position", "avg_finish_last_3", "total_points_last_3",
            "avg_finish_last_5", "dnf_rate", "season_points_driver",
            "championship_position", "points_gap_to_leader", "track_history_score",
        ]
        self.constructor_cols = constructor_cols or [
            "constructor_season_points", "constructor_avg_finish_last_3",
            "constructor_points_last_3", "dnf_rate",
        ]
        self.circuit_cols = circuit_cols or [
            "length_km", "corners", "drs_zones", "altitude_m", "track_type_code",
            "downforce_code", "tyre_degradation_code", "overtaking_code",
        ]
        self.weather_cols = weather_cols or [
            "openmeteo_temp_mean", "openmeteo_humidity", "openmeteo_precip_mm",
            "openmeteo_wind_speed", "openmeteo_rain_flag", "openmeteo_pressure",
        ]
        self.race_context_cols = race_context_cols or ["round"]

        # Initialize with parent's init (bypassing SequenceBuilder's __init__ column setup)
        # Call grandparent-like setup manually
        self.df = df.sort_values(["year", "round"]).reset_index(drop=True)
        self.driver_enc = driver_enc
        self.constructor_enc = constructor_enc
        self.circuit_enc = circuit_enc
        self._identify_races()

    def build_driver_features(self, race_df: pd.DataFrame) -> np.ndarray:
        """Override: Uses V2 feature sizing (embedding indices, not embedding dims).

        The parent SequenceBuilder uses get_feature_vector_size() which counts
        embedding dimensions (16+8+8=32) in the feature vector. V2 stores only
        embedding indices (3 ints) and lets the model do embedding lookup.
        """
        from .feature_engineering_v2 import get_feature_vector_size_v2
        d_candidate, _ = get_feature_vector_size_v2(self.config.context_window)

        num_drivers = len(race_df)
        features = np.zeros((num_drivers, d_candidate), dtype=np.float32)

        for i, (_, row) in enumerate(race_df.iterrows()):
            offset = 0

            # Driver embedding index
            features[i, offset] = self.driver_enc.encode(row["driver_abbreviation"])
            offset += 1

            # Constructor embedding index
            features[i, offset] = self.constructor_enc.encode(row["constructor"])
            offset += 1

            # Circuit embedding index
            features[i, offset] = self.circuit_enc.encode(row["circuit"])
            offset += 1

            # Numeric driver features (V2 column list)
            for col in self.numeric_driver_cols:
                val = row.get(col, 0)
                features[i, offset] = val if not pd.isna(val) else 0
                offset += 1

            # Constructor features (V2 column list)
            for col in self.constructor_cols:
                val = row.get(col, 0)
                features[i, offset] = val if not pd.isna(val) else 0
                offset += 1

            # Circuit features
            for col in self.circuit_cols:
                val = row.get(col, 0)
                features[i, offset] = val if not pd.isna(val) else 0
                offset += 1

            # Weather features
            for col in self.weather_cols:
                val = row.get(col, 0)
                features[i, offset] = val if not pd.isna(val) else 0
                offset += 1

            # Race context
            for col in self.race_context_cols:
                val = row.get(col, 0)
                features[i, offset] = val if not pd.isna(val) else 0
                offset += 1

        return features

    def build_context_features(self, race_df: pd.DataFrame) -> np.ndarray:
        """Build context summary vector with mean, std, and max of driver features.

        V2 improvement: captures distribution of driver performance in each race,
        not just the average.
        """
        n_numeric = len(self.numeric_driver_cols)
        n_constructor = len(self.constructor_cols)
        n_circuit = len(self.circuit_cols)
        n_weather = len(self.weather_cols)
        n_race = len(self.race_context_cols)

        # V2: 3x driver numeric (mean, std, max) + 3x constructor + circuit + weather + race + top3 + winner
        d_context = (
            n_numeric * 3 + n_constructor * 3 + n_circuit + n_weather + n_race +
            self.config.num_top_finishers * self.config.driver_embed_dim +
            self.config.driver_embed_dim
        )

        features = np.zeros(d_context, dtype=np.float32)
        offset = 0

        # Mean of driver numeric features
        for col in self.numeric_driver_cols:
            vals = race_df[col].values if col in race_df.columns else np.array([0])
            features[offset] = np.nanmean(vals) if not np.all(np.isnan(vals)) else 0
            offset += 1

        # Std of driver numeric features (NEW)
        for col in self.numeric_driver_cols:
            vals = race_df[col].values if col in race_df.columns else np.array([0])
            features[offset] = np.nanstd(vals) if not np.all(np.isnan(vals)) else 0
            offset += 1

        # Max of driver numeric features (NEW)
        for col in self.numeric_driver_cols:
            vals = race_df[col].values if col in race_df.columns else np.array([0])
            features[offset] = np.nanmax(vals) if not np.all(np.isnan(vals)) else 0
            offset += 1

        # Mean of constructor features
        for col in self.constructor_cols:
            vals = race_df[col].values if col in race_df.columns else np.array([0])
            features[offset] = np.nanmean(vals) if not np.all(np.isnan(vals)) else 0
            offset += 1

        # Std of constructor features (NEW)
        for col in self.constructor_cols:
            vals = race_df[col].values if col in race_df.columns else np.array([0])
            features[offset] = np.nanstd(vals) if not np.all(np.isnan(vals)) else 0
            offset += 1

        # Max of constructor features (NEW)
        for col in self.constructor_cols:
            vals = race_df[col].values if col in race_df.columns else np.array([0])
            features[offset] = np.nanmax(vals) if not np.all(np.isnan(vals)) else 0
            offset += 1

        # Circuit features (same for all drivers)
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

        # Top-3 finisher driver embedding indices
        podium = race_df[race_df["finish_position"].isin([1, 2, 3])]
        podium = podium.sort_values("finish_position")
        for j in range(min(self.config.num_top_finishers, len(podium))):
            row = podium.iloc[j]
            features[offset] = self.driver_enc.encode(row["driver_abbreviation"])
            offset += 1
        for j in range(len(podium), self.config.num_top_finishers):
            features[offset] = 0
            offset += 1

        # Winner driver embedding index
        winner = race_df[race_df["finish_position"] == 1]
        if not winner.empty:
            features[offset] = self.driver_enc.encode(winner.iloc[0]["driver_abbreviation"])
        # offset += 1  # last element

        return features
