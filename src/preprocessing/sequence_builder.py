"""Sequence construction for the transformer model.

Builds training sequences where:
- Input: Last N=10 races as context (race summaries)
- Output: Winner of the target race

Each race in the context sequence is a summary vector encoding:
- Pooled driver features (mean/max of all drivers)
- Circuit embedding index
- Weather features
- Top-3 finisher driver IDs
- Winner driver ID
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
if os.environ.get("COLAB"):
    PROCESSED_DATA = Path("/content/drive/MyDrive/f1_transformer/data/processed")

from .encoders import DriverEncoder, ConstructorEncoder, CircuitEncoder


@dataclass
class FeatureConfig:
    """Configuration for feature dimensions."""
    context_window: int = 10
    driver_embed_dim: int = 16
    constructor_embed_dim: int = 8
    circuit_embed_dim: int = 8
    num_numeric_driver: int = 10
    num_constructor_feat: int = 4
    num_circuit_feat: int = 8
    num_weather_feat: int = 6
    num_race_context: int = 1
    num_top_finishers: int = 3  # Track top 3 per race


def get_feature_vector_size(config: FeatureConfig) -> Tuple[int, int]:
    """Calculate feature vector sizes.

    Returns:
        (d_candidate_raw, d_context_raw)
        d_candidate_raw: Features per driver candidate
        d_context_raw: Features per race in context
    """
    # Per-driver features: embeddings + numeric
    d_candidate = (
        config.driver_embed_dim +
        config.constructor_embed_dim +
        config.circuit_embed_dim +
        config.num_numeric_driver +
        config.num_constructor_feat +
        config.num_circuit_feat +
        config.num_weather_feat +
        config.num_race_context
    )

    # Per-race context: pooled driver features + circuit + weather + top3 + winner
    d_context = (
        config.num_numeric_driver +  # mean-pooled driver features
        config.num_circuit_feat +    # circuit stats
        config.num_weather_feat +    # weather
        config.num_race_context +    # round info
        config.num_top_finishers * config.driver_embed_dim +  # top 3 drivers
        config.driver_embed_dim      # winner
    )

    return d_candidate, d_context


class SequenceBuilder:
    """Builds temporal sequences for transformer training."""

    def __init__(
        self,
        df: pd.DataFrame,
        driver_enc: DriverEncoder,
        constructor_enc: ConstructorEncoder,
        circuit_enc: CircuitEncoder,
        config: Optional[FeatureConfig] = None,
    ):
        self.df = df.sort_values(["year", "round"]).reset_index(drop=True)
        self.driver_enc = driver_enc
        self.constructor_enc = constructor_enc
        self.circuit_enc = circuit_enc
        self.config = config or FeatureConfig()

        # Feature column definitions
        self.numeric_driver_cols = [
            "grid_position", "q_position", "avg_finish_last_3", "total_points_last_3",
            "avg_finish_last_5", "dnf_rate", "season_points_driver",
            "championship_position", "points_gap_to_leader", "track_history_score",
        ]
        self.constructor_cols = [
            "constructor_season_points", "constructor_avg_finish_last_3",
            "constructor_points_last_3", "dnf_rate",
        ]
        self.circuit_cols = [
            "length_km", "corners", "drs_zones", "altitude_m", "track_type_code",
            "downforce_code", "tyre_degradation_code", "overtaking_code",
        ]
        self.weather_cols = [
            "openmeteo_temp_mean", "openmeteo_humidity", "openmeteo_precip_mm",
            "openmeteo_wind_speed", "openmeteo_rain_flag", "openmeteo_pressure",
        ]
        self.race_context_cols = ["round"]

        self._identify_races()

    def _identify_races(self):
        """Create a unique ID for each race (year, round)."""
        self.df["race_id"] = self.df.apply(
            lambda r: f"{r['year']}_{r['round']}_{r['circuit']}", axis=1
        )
        # Get unique races in chronological order
        self.races = self.df[["year", "round", "circuit", "race_id", "event_date"]].drop_duplicates()
        self.races = self.races.sort_values(["year", "round"]).reset_index(drop=True)

    def get_race_drivers(self, year: int, round_num: int) -> pd.DataFrame:
        """Get all drivers for a specific race."""
        mask = (self.df["year"] == year) & (self.df["round"] == round_num)
        return self.df[mask].sort_values("grid_position").reset_index(drop=True)

    def build_driver_features(self, race_df: pd.DataFrame) -> np.ndarray:
        """Build per-driver feature vectors for a race.

        Args:
            race_df: DataFrame with all drivers for one race (sorted by grid position)

        Returns:
            ndarray of shape (num_drivers, d_candidate_raw)
        """
        num_drivers = len(race_df)
        d_candidate, _ = get_feature_vector_size(self.config)

        features = np.zeros((num_drivers, d_candidate), dtype=np.float32)
        col_offset = 0

        # Driver embedding indices (one-hot index, not actual embedding)
        # We store the index, embedding lookup happens in the model
        # Actually, let's store raw features and let the model handle embeddings
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

            # Numeric driver features
            for col in self.numeric_driver_cols:
                val = row.get(col, 0)
                features[i, offset] = val if not pd.isna(val) else 0
                offset += 1

            # Constructor features
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
        """Build context summary vector for a single race.

        Encodes: pooled driver stats + circuit + weather + top3 drivers + winner.

        Args:
            race_df: DataFrame with all drivers for one race

        Returns:
            ndarray of shape (d_context_raw,)
        """
        _, d_context = get_feature_vector_size(self.config)
        features = np.zeros(d_context, dtype=np.float32)
        offset = 0

        # Mean-pooled driver numeric features
        for col in self.numeric_driver_cols:
            vals = race_df[col].values if col in race_df.columns else np.array([0])
            features[offset] = np.nanmean(vals) if not np.all(np.isnan(vals)) else 0
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

        # Top-3 finisher driver embedding indices
        podium = race_df[race_df["finish_position"].isin([1, 2, 3])]
        podium = podium.sort_values("finish_position")
        for j in range(min(self.config.num_top_finishers, len(podium))):
            row = podium.iloc[j]
            features[offset] = self.driver_enc.encode(row["driver_abbreviation"])
            offset += 1
        # Pad missing podium
        for j in range(len(podium), self.config.num_top_finishers):
            features[offset] = 0  # unknown
            offset += 1

        # Winner driver embedding index
        winner = race_df[race_df["finish_position"] == 1]
        if not winner.empty:
            features[offset] = self.driver_enc.encode(winner.iloc[0]["driver_abbreviation"])
        offset += 1

        return features

    def build_sequences(self, start_year: int, end_year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build training sequences for years [start_year, end_year].

        Each sequence:
        - context: (N, d_context_raw) last N races
        - candidates: (20, d_candidate_raw) drivers in target race
        - winner_idx: index (0-19) of winner in candidates
        - time_gaps: (N,) days between each context race and target
        - target_year, target_round: metadata

        Returns:
            context_seqs: (num_seqs, N, d_context_raw)
            candidate_feats: (num_seqs, 20, d_candidate_raw)
            winner_indices: (num_seqs,)
            time_gaps: (num_seqs, N)
            metadata: (num_seqs, 2) [year, round]
        """
        N = self.config.context_window

        # Get unique races in range
        race_list = self.races[
            (self.races["year"] >= start_year) & (self.races["year"] <= end_year)
        ].sort_values(["year", "round"]).reset_index(drop=True)

        # Build index: for each race, find previous N races
        all_races = self.races.sort_values(["year", "round"]).reset_index(drop=True)
        race_index_map = {row["race_id"]: i for i, row in all_races.iterrows()}

        context_seqs = []
        candidate_feats = []
        winner_indices = []
        time_gaps_list = []
        metadata = []

        for _, target_race in race_list.iterrows():
            target_year = target_race["year"]
            target_round = target_race["round"]
            target_id = target_race["race_id"]

            # Find index in chronological list
            try:
                target_chrono_idx = race_index_map[target_id]
            except KeyError:
                continue

            # Get previous N races (can span across seasons)
            prev_indices = list(range(max(0, target_chrono_idx - N), target_chrono_idx))
            if len(prev_indices) < 2:  # Need at least 2 context races
                continue

            prev_races = all_races.iloc[prev_indices]

            # Build context features for each previous race
            context = np.zeros((N, 0), dtype=np.float32)  # Will shape correctly
            context_rows = []
            time_gaps = np.zeros(N, dtype=np.float32)

            target_date = pd.Timestamp(target_race["event_date"])

            for j, (_, prev_race) in enumerate(prev_races.iterrows()):
                prev_df = self.get_race_drivers(prev_race["year"], prev_race["round"])
                if len(prev_df) < 2:
                    continue
                ctx = self.build_context_features(prev_df)
                context_rows.append(ctx)

                # Time gap in days
                prev_date = pd.Timestamp(prev_race["event_date"])
                time_gaps[j] = max(1, (target_date - prev_date).days)

            if len(context_rows) < 2:
                continue

            # Pad context if needed (for early races with fewer than N context races)
            if len(context_rows) < N:
                pad_size = N - len(context_rows)
                pad_vec = np.zeros_like(context_rows[0]) if context_rows else np.zeros(1)
                context_rows = [pad_vec] * pad_size + context_rows
                time_gaps = np.concatenate([np.full(pad_size, 365.0), time_gaps[-len(context_rows):]])

            context = np.array(context_rows[-N:])  # Take last N
            time_gaps = time_gaps[-N:]

            # Build candidate features for target race
            candidates_df = self.get_race_drivers(target_year, target_round)
            candidates = self.build_driver_features(candidates_df)

            # Get winner index
            winner_mask = candidates_df["finish_position"] == 1
            if not winner_mask.any():
                continue
            winner_idx = int(candidates_df[winner_mask].index[0])
            # Clamp to 20
            winner_idx = min(winner_idx, min(19, len(candidates_df) - 1))

            # Pad/truncate candidates to 20
            if len(candidates) < 20:
                pad = np.zeros((20 - len(candidates), candidates.shape[1]), dtype=np.float32)
                candidates = np.concatenate([candidates, pad], axis=0)
            else:
                candidates = candidates[:20]

            context_seqs.append(context)
            candidate_feats.append(candidates)
            winner_indices.append(winner_idx)
            time_gaps_list.append(time_gaps)
            metadata.append([target_year, target_round])

        print(f"Built {len(context_seqs)} sequences for {start_year}-{end_year}")

        return (
            np.array(context_seqs, dtype=np.float32),
            np.array(candidate_feats, dtype=np.float32),
            np.array(winner_indices, dtype=np.int64),
            np.array(time_gaps_list, dtype=np.float32),
            np.array(metadata, dtype=np.int64),
        )
