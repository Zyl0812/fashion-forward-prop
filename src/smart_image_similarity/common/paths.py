from smart_image_similarity.common.settings import (
    PROJECT_ROOT,
    SIM_CATALOG_DIR,
    SIM_CHROMA_DIR,
    SIM_DATA_DIR,
    SIM_MODELS_DIR,
)


SRC_DIR = PROJECT_ROOT / "src"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = (PROJECT_ROOT / SIM_MODELS_DIR).resolve()
DATA_DIR = (PROJECT_ROOT / SIM_DATA_DIR).resolve()
CATALOG_DIR = (PROJECT_ROOT / SIM_CATALOG_DIR).resolve()
CHROMA_DIR = (PROJECT_ROOT / SIM_CHROMA_DIR).resolve()
LABELS_PATH = DATA_DIR / "fashion-labels.csv"


def ensure_runtime_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
