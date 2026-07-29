from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://app:app@db:5432/football",
)
DB_POOL_SIZE = _get_int_env("DB_POOL_SIZE", 5)
DB_MAX_OVERFLOW = _get_int_env("DB_MAX_OVERFLOW", 10)
DB_POOL_TIMEOUT = _get_int_env("DB_POOL_TIMEOUT", 30)
DB_POOL_RECYCLE = _get_int_env("DB_POOL_RECYCLE", 1800)
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "models")))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
REPORT_DIR = Path(os.getenv("REPORT_DIR", str(MODEL_DIR / "reports")))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
