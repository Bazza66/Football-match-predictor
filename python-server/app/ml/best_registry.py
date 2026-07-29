from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


LOWER_IS_BETTER_TOKENS = ("log_loss", "brier", "ece")


def metric_lower_is_better(metric_name: str) -> bool:
    name = metric_name.lower()
    return any(token in name for token in LOWER_IS_BETTER_TOKENS)


def make_registry_key(
    model_name: str,
    feature_version: str,
    league_code: str | None,
    season_label: str | None,
) -> str:
    league_part = league_code or "all"
    season_part = season_label or "all"
    return f"{model_name}|{feature_version}|{league_part}|{season_part}"


def extract_metric(metrics: dict | None, metric_name: str) -> float | None:
    if not metrics:
        return None
    value = metrics.get(metric_name)
    if isinstance(value, (int, float)) and value == value:
        return float(value)
    return None


def _registry_path(model_dir: Path) -> Path:
    return model_dir / "best_models.json"


def load_registry(model_dir: Path) -> dict[str, Any]:
    path = _registry_path(model_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_registry(model_dir: Path, registry: dict[str, Any]) -> None:
    path = _registry_path(model_dir)
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def update_best_registry(
    model_dir: Path,
    key: str,
    candidate: dict[str, Any],
    *,
    metric_name: str,
    metric_value: float,
) -> bool:
    registry = load_registry(model_dir)
    current = registry.get(key)
    lower_is_better = metric_lower_is_better(metric_name)

    def _is_better(new_value: float, old_value: float) -> bool:
        return new_value < old_value if lower_is_better else new_value > old_value

    should_update = True
    if isinstance(current, dict):
        existing_value = current.get("metric_value")
        if isinstance(existing_value, (int, float)) and existing_value == existing_value:
            should_update = _is_better(metric_value, float(existing_value))

    if not should_update:
        return False

    registry[key] = {
        **candidate,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    save_registry(model_dir, registry)
    return True
