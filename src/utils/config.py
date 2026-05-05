"""Global configuration constants and YAML loaders."""
import yaml
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"
F1_CACHE = DATA_DIR / "f1_cache"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Colab paths (used when running in Colab)
COLAB_ROOT = Path("/content/drive/MyDrive/f1_transformer")
COLAB_DATA = COLAB_ROOT / "data"
COLAB_MODELS = COLAB_ROOT / "models"


def load_model_config(path=None):
    """Load model configuration from YAML."""
    path = path or (CONFIG_DIR / "model_config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_training_config(path=None):
    """Load training configuration from YAML."""
    path = path or (CONFIG_DIR / "training_config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def is_colab():
    """Detect if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_device(force_device=None):
    """Get the appropriate torch device."""
    if force_device:
        return force_device
    import torch
    if is_colab():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
