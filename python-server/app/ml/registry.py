from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from app.ml.best_registry import load_registry, make_registry_key


def _sorted_models(paths: list[Path]) -> list[Path]:
    return sorted(
        [path for path in paths if path.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def list_models(model_dir: Path) -> list[Path]:
    return _sorted_models(list(model_dir.rglob("*.pkl")))


def list_models_by_name(model_dir: Path, model_name: str) -> list[Path]:
    pattern = f"{model_name}_*.pkl"
    return _sorted_models(list(model_dir.rglob(pattern)))


def load_latest_model(model_dir: Path) -> tuple[Any | None, dict[str, Any] | None]:
    models = list_models(model_dir)
    if not models:
        return None, None
    model_path = models[0]
    model = joblib.load(model_path)
    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {
            "model_name": model_path.stem,
            "model_version": model_path.stem,
            "artifact": model_path.name,
        }
    return model, meta


def load_latest_model_by_name(
    model_dir: Path, model_name: str
) -> tuple[Any | None, dict[str, Any] | None]:
    models = list_models_by_name(model_dir, model_name)
    if not models:
        return None, None
    model_path = models[0]
    model = joblib.load(model_path)
    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {
            "model_name": model_name,
            "model_version": model_path.stem,
            "artifact": model_path.name,
        }
    return model, meta


def load_best_model_by_name(
    model_dir: Path,
    model_name: str,
    *,
    feature_version: str = "v1",
    league_code: str | None = None,
    season_label: str | None = None,
) -> tuple[Any | None, dict[str, Any] | None]:
    registry = load_registry(model_dir)
    key = make_registry_key(model_name, feature_version, league_code, season_label)
    entry = registry.get(key)
    if not isinstance(entry, dict):
        return None, None
    artifact = entry.get("artifact")
    if not isinstance(artifact, str):
        return None, None
    model_path = model_dir / artifact
    if not model_path.exists():
        return None, None
    model = joblib.load(model_path)
    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {
            "model_name": model_name,
            "model_version": model_path.stem,
            "artifact": model_path.name,
        }
    return model, meta


def load_model_by_version(
    model_dir: Path,
    model_name: str,
    model_version: str,
) -> tuple[Any | None, dict[str, Any] | None]:
    filename = f"{model_name}_{model_version}.pkl"
    model_path = model_dir / filename
    if not model_path.exists():
        matches = _sorted_models(list(model_dir.rglob(filename)))
        if not matches:
            return None, None
        model_path = matches[0]
    model = joblib.load(model_path)
    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {
            "model_name": model_name,
            "model_version": model_version,
            "artifact": model_path.name,
        }
    return model, meta
