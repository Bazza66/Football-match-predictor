from __future__ import annotations

import argparse
from datetime import datetime

from sqlalchemy.orm import joinedload

from app.config import MODEL_DIR
from app.db.models import ModelRun
from app.db.session import SessionLocal
from app.ml.best_registry import (
    extract_metric,
    load_registry,
    metric_lower_is_better,
    make_registry_key,
    save_registry,
)


def backfill_best(
    *,
    metric: str,
    reset: bool,
    model_name: str | None,
    feature_version: str | None,
    league_code: str | None,
    season_label: str | None,
) -> int:
    registry: dict[str, dict] = {} if reset else load_registry(MODEL_DIR)
    lower_is_better = metric_lower_is_better(metric)

    with SessionLocal() as session:
        query = (
            session.query(ModelRun)
            .options(joinedload(ModelRun.artifacts))
            .filter(ModelRun.status == "completed")
        )
        if model_name:
            query = query.filter(ModelRun.model_name == model_name)
        if feature_version:
            query = query.filter(ModelRun.feature_version == feature_version)
        if league_code:
            query = query.filter(ModelRun.league_code == league_code)
        if season_label:
            query = query.filter(ModelRun.season_label == season_label)

        runs = query.all()

    updated = 0
    for run in runs:
        metric_value = extract_metric(run.metrics, metric)
        if metric_value is None:
            continue
        model_artifact = next(
            (artifact for artifact in run.artifacts if artifact.artifact_type == "model"),
            None,
        )
        if model_artifact is None:
            continue
        meta_artifact = next(
            (artifact for artifact in run.artifacts if artifact.artifact_type == "model_meta"),
            None,
        )

        key = make_registry_key(
            run.model_name, run.feature_version, run.league_code, run.season_label
        )
        current = registry.get(key)

        def is_better(new: float, old: float) -> bool:
            return new < old if lower_is_better else new > old

        should_update = True
        if current and isinstance(current.get("metric_value"), (int, float)):
            should_update = is_better(metric_value, float(current["metric_value"]))

        if not should_update:
            continue

        registry[key] = {
            "run_id": run.run_id,
            "model_name": run.model_name,
            "model_version": run.model_version,
            "feature_version": run.feature_version,
            "league_code": run.league_code,
            "season_label": run.season_label,
            "artifact": model_artifact.path,
            "meta": meta_artifact.path if meta_artifact else None,
            "metric_name": metric,
            "metric_value": metric_value,
            "direction": "lower_is_better" if lower_is_better else "higher_is_better",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        updated += 1

    save_registry(MODEL_DIR, registry)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        default="val_log_loss",
        help="Metric to determine the best model (default: val_log_loss).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Rebuild the registry from scratch.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--feature-version", default=None)
    parser.add_argument("--league", default=None)
    parser.add_argument("--season", default=None)
    args = parser.parse_args()

    updated = backfill_best(
        metric=args.metric,
        reset=args.reset,
        model_name=args.model_name,
        feature_version=args.feature_version,
        league_code=args.league,
        season_label=args.season,
    )
    print(f"Updated best registry entries: {updated}")


if __name__ == "__main__":
    main()
