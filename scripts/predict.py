"""Predict 2025 F1 season winners using trained model.

Loads the combined 2014-2024 model, predicts all 24 races,
and compares against real 2025 results.

Expected: ~41.7% accuracy (10/24 correct)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.predict_2025 import main

if __name__ == "__main__":
    main()
