import os
from pathlib import Path


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = Path(os.getenv("SIM_PROJECT_ROOT", _default_project_root())).resolve()
SIM_APP_PORT = int(os.getenv("SIM_APP_PORT", "5000"))
SIM_MODELS_DIR = os.getenv("SIM_MODELS_DIR", "assets/models")
SIM_DATA_DIR = os.getenv("SIM_DATA_DIR", "assets/data")
SIM_CATALOG_DIR = os.getenv("SIM_CATALOG_DIR", "assets/data/catalog")
SIM_CHROMA_DIR = os.getenv("SIM_CHROMA_DIR", "artifacts/chroma")
