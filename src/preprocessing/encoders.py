"""Categorical encoders for drivers, constructors, and circuits.

Creates and persists mappings from string IDs to integer indices.
Handles unknown/new entities for robustness.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
if os.environ.get("COLAB"):
    PROCESSED_DATA = Path("/content/drive/MyDrive/f1_transformer/data/processed")


class DriverEncoder:
    """Maps driver abbreviations to integer IDs."""

    def __init__(self):
        self.driver_to_id: Dict[str, int] = {}
        self.id_to_driver: Dict[int, str] = {}
        self.unknown_id = 0

    def fit(self, driver_abbrs: list[str]):
        """Build mapping from list of driver abbreviations."""
        unique = sorted(set(driver_abbrs))
        # Reserve 0 for unknown
        for i, abbr in enumerate(unique):
            self.driver_to_id[abbr] = i + 1
            self.id_to_driver[i + 1] = abbr
        print(f"DriverEncoder: {len(self.driver_to_id)} drivers")
        return self

    def encode(self, abbr: str) -> int:
        """Encode a driver abbreviation to integer ID."""
        return self.driver_to_id.get(abbr, self.unknown_id)

    def decode(self, idx: int) -> str:
        """Decode integer ID back to driver abbreviation."""
        return self.id_to_driver.get(idx, "UNK")

    def __len__(self):
        return len(self.driver_to_id) + 1  # +1 for unknown


class ConstructorEncoder:
    """Maps constructor names to integer IDs."""

    def __init__(self):
        self.constructor_to_id: Dict[str, int] = {}
        self.id_to_constructor: Dict[int, str] = {}
        self.unknown_id = 0

    def fit(self, constructor_names: list[str]):
        unique = sorted(set(constructor_names))
        for i, name in enumerate(unique):
            self.constructor_to_id[name] = i + 1
            self.id_to_constructor[i + 1] = name
        print(f"ConstructorEncoder: {len(self.constructor_to_id)} constructors")
        return self

    def encode(self, name: str) -> int:
        return self.constructor_to_id.get(name, self.unknown_id)

    def decode(self, idx: int) -> str:
        return self.id_to_constructor.get(idx, "UNK")

    def __len__(self):
        return len(self.constructor_to_id) + 1


class CircuitEncoder:
    """Maps circuit names to integer IDs."""

    def __init__(self):
        self.circuit_to_id: Dict[str, int] = {}
        self.id_to_circuit: Dict[int, str] = {}
        self.unknown_id = 0

    def fit(self, circuit_names: list[str]):
        unique = sorted(set(circuit_names))
        for i, name in enumerate(unique):
            self.circuit_to_id[name] = i + 1
            self.id_to_circuit[i + 1] = name
        print(f"CircuitEncoder: {len(self.circuit_to_id)} circuits")
        return self

    def encode(self, name: str) -> int:
        return self.circuit_to_id.get(name, self.unknown_id)

    def decode(self, idx: int) -> str:
        return self.id_to_circuit.get(idx, "UNK")

    def __len__(self):
        return len(self.circuit_to_id) + 1


def build_encoders(race_df: pd.DataFrame) -> tuple[DriverEncoder, ConstructorEncoder, CircuitEncoder]:
    """Build all encoders from a race results DataFrame."""
    driver_enc = DriverEncoder().fit(race_df["driver_abbreviation"].unique().tolist())
    constructor_enc = ConstructorEncoder().fit(race_df["constructor"].unique().tolist())
    circuit_enc = CircuitEncoder().fit(race_df["circuit"].unique().tolist())
    return driver_enc, constructor_enc, circuit_enc


def save_encoders(driver_enc, constructor_enc, circuit_enc, path=PROCESSED_DATA):
    """Persist encoders to disk."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "driver_encoder.pkl", "wb") as f:
        pickle.dump(driver_enc, f)
    with open(path / "constructor_encoder.pkl", "wb") as f:
        pickle.dump(constructor_enc, f)
    with open(path / "circuit_encoder.pkl", "wb") as f:
        pickle.dump(circuit_enc, f)
    print(f"Encoders saved to {path}")


def load_encoders(path=PROCESSED_DATA):
    """Load persisted encoders."""
    with open(path / "driver_encoder.pkl", "rb") as f:
        driver_enc = pickle.load(f)
    with open(path / "constructor_encoder.pkl", "rb") as f:
        constructor_enc = pickle.load(f)
    with open(path / "circuit_encoder.pkl", "rb") as f:
        circuit_enc = pickle.load(f)
    return driver_enc, constructor_enc, circuit_enc
