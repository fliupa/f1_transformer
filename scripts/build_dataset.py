"""Build dataset: features, encoders, sequences, normalization.

Entry point that calls src.preprocessing.build_dataset.
Generates features_train.pt, features_val.pt, features_2025.pt,
features_finetune.pt (ablation), and metadata.pkl.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.build_dataset import main

if __name__ == "__main__":
    main()
