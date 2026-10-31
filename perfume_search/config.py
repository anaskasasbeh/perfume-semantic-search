"""Project paths and settings.

Every value can be overridden with an environment variable, so the same code
runs on a laptop, in tests and on Vercel without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE pairs from a local .env file (no extra dependency).

    Non-empty values already present in the environment win, so Vercel's
    dashboard settings are never overwritten.
    """
    if not path.exists():
        return
    # utf-8-sig also strips the invisible BOM that some Windows editors add.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        if not os.environ.get(key):
            os.environ[key] = value.strip().strip('"').strip("'")


load_dotenv()

# --- Paths -----------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_CSV = Path(os.getenv("PERFUME_RAW_CSV", DATA_DIR / "raw" / "perfume_data.csv"))
EVAL_QUERIES = DATA_DIR / "eval" / "queries.json"
ARTIFACTS_DIR = Path(os.getenv("PERFUME_ARTIFACTS_DIR", ROOT / "artifacts"))
CACHE_DIR = Path(os.getenv("PERFUME_CACHE_DIR", ROOT / ".cache"))
REPORTS_DIR = ROOT / "reports"
WEB_DIR = ROOT / "web"

# --- Embeddings ------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
QUERY_TIMEOUT_MS = int(os.getenv("QUERY_TIMEOUT_MS", "8000"))

# --- Dataset schema --------------------------------------------------------
# Maps our internal field names to the column names in the source CSV.
# To plug in another shop's catalog, change the right-hand side only.
COLUMN_MAP: dict[str, str] = {
    "name": "Name",
    "brand": "Brand",
    "description": "Description",
    "notes": "Notes",
    "image_url": "Image URL",
}

# --- Clustering ------------------------------------------------------------
K_RANGE = (5, 12)  # candidate numbers of groups for the home page
MIN_GROUP_FRACTION = 0.015  # reject k values that create very small groups (<1.5%)
RANDOM_SEED = 42
