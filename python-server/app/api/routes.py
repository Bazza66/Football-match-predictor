from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import json
from types import SimpleNamespace
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

import pandas as pd

from app.config import MODEL_DIR, REPORT_DIR
from app.db.models import Feature, League, Match, ModelRun, Prediction, Season
from app.ml.best_registry import load_registry
from app.ml.dataset import FEATURE_COLUMNS
from app.ml.predict_to_db import resolve_class_order
from app.ml.registry import load_latest_model_by_name, load_model_by_version
from app.db.session import get_session, SessionLocal
from app.schemas import (
    LeagueOut,
    MatchOut,
    ModelArtifactOut,
    ModelRunBestOut,
    ModelRunOut,
    ModelRunSummaryOut,
    DatasetIntegrityReportOut,
    LabelAuditReportOut,
    TrainingDivergenceReportOut,
    ModelEvalReportOut,
    TuningSweepReportOut,
    SeasonOut,
    PredictionRequest,
    PredictionOut,
)

router = APIRouter()


@router.get("/health")
def health(db: Session = Depends(get_session)):
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {"status": "ok"}


@router.get("/leagues", response_model=list[LeagueOut])
def list_leagues(db: Session = Depends(get_session)):
    leagues = db.query(League).order_by(League.name).all()
    return [LeagueOut(id=l.id, code=l.code, name=l.name) for l in leagues]


@router.get("/seasons", response_model=list[SeasonOut])
def list_seasons(
    league: str | None = Query(default=None, description="League code, e.g. en.1"),
    db: Session = Depends(get_session),
):
    query = db.query(Season).join(League, Season.league_id == League.id)
    if league:
        query = query.filter(League.code == league)
    seasons = query.order_by(Season.label).all()
    return [SeasonOut(id=s.id, league_id=s.league_id, label=s.label) for s in seasons]


@router.get("/matches", response_model=list[MatchOut])
def list_matches(
    league: str | None = Query(default=None, description="League code"),
    season: str | None = Query(default=None, description="Season label"),
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_session),
):
    query = (
        db.query(Match)
        .options(joinedload(Match.home_team), joinedload(Match.away_team))
        .join(Season, Match.season_id == Season.id)
        .join(League, Season.league_id == League.id)
    )
    if league:
        query = query.filter(League.code == league)
    if season:
        query = query.filter(Season.label == season)
    matches = query.order_by(Match.match_date.desc().nullslast()).limit(limit).all()
    return [
        MatchOut(
            id=m.id,
            season_id=m.season_id,
            round=m.round,
            date=m.match_date,
            time=m.match_time,
            home_team=m.home_team.name,
            away_team=m.away_team.name,
            home_score=m.home_goals,
            away_score=m.away_goals,
        )
        for m in matches
    ]


@router.post("/predict", response_model=PredictionOut)
def predict_match(
    payload: PredictionRequest,
    feature_version: str = Query(default="v1"),
    model_name: str = Query(default="main_xgboost"),
    model_version: str | None = Query(default=None),
    db: Session = Depends(get_session),
):
    if payload.match_id is None:
        raise HTTPException(status_code=400, detail="match_id is required.")

    match = db.query(Match).filter(Match.id == payload.match_id).one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found.")

    feature_row = (
        db.query(Feature)
        .filter(Feature.match_id == match.id, Feature.version == feature_version)
        .one_or_none()
    )
    if not feature_row:
        raise HTTPException(
            status_code=400,
            detail="Missing feature row for match/version.",
        )

    if model_version:
        model, meta = load_model_by_version(MODEL_DIR, model_name, model_version)
    else:
        model, meta = load_latest_model_by_name(MODEL_DIR, model_name)
    if model is None or meta is None:
        raise HTTPException(status_code=404, detail="Model artifact not found.")

    class_order = resolve_class_order(meta)
    class_index = {label: idx for idx, label in enumerate(class_order)}

    features = {col: getattr(feature_row, col) for col in FEATURE_COLUMNS}
    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    probs = model.predict_proba(X)

    resolved_model_version = meta.get("model_version") or model_version or "unknown"
    existing = (
        db.query(Prediction)
        .filter(
            Prediction.match_id == match.id,
            Prediction.model_name == model_name,
            Prediction.model_version == resolved_model_version,
        )
        .one_or_none()
    )
    if existing:
        return PredictionOut(
            match_id=existing.match_id,
            model_name=existing.model_name,
            model_version=existing.model_version,
            predicted_home_score=existing.predicted_home_score,
            predicted_away_score=existing.predicted_away_score,
            win_prob_home=existing.win_prob_home,
            win_prob_draw=existing.win_prob_draw,
            win_prob_away=existing.win_prob_away,
        )

    prediction = Prediction(
        match_id=match.id,
        model_name=model_name,
        model_version=resolved_model_version,
        predicted_home_score=None,
        predicted_away_score=None,
        win_prob_home=float(probs[0][class_index["H"]]) if "H" in class_index else None,
        win_prob_draw=float(probs[0][class_index["D"]]) if "D" in class_index else None,
        win_prob_away=float(probs[0][class_index["A"]]) if "A" in class_index else None,
    )
    db.add(prediction)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(Prediction)
            .filter(
                Prediction.match_id == match.id,
                Prediction.model_name == model_name,
                Prediction.model_version == resolved_model_version,
            )
            .one_or_none()
        )
        if existing:
            return PredictionOut(
                match_id=existing.match_id,
                model_name=existing.model_name,
                model_version=existing.model_version,
                predicted_home_score=existing.predicted_home_score,
                predicted_away_score=existing.predicted_away_score,
                win_prob_home=existing.win_prob_home,
                win_prob_draw=existing.win_prob_draw,
                win_prob_away=existing.win_prob_away,
            )
        raise

    return PredictionOut(
        match_id=prediction.match_id,
        model_name=prediction.model_name,
        model_version=prediction.model_version,
        predicted_home_score=prediction.predicted_home_score,
        predicted_away_score=prediction.predicted_away_score,
        win_prob_home=prediction.win_prob_home,
        win_prob_draw=prediction.win_prob_draw,
        win_prob_away=prediction.win_prob_away,
    )


@router.get("/predictions", response_model=list[PredictionOut])
def list_predictions(
    match_id: int | None = Query(default=None),
    model_name: str | None = Query(default=None),
    model_version: str | None = Query(default=None),
    limit: int = Query(default=50, le=500),
    db: Session = Depends(get_session),
):
    query = db.query(Prediction)
    if match_id is not None:
        query = query.filter(Prediction.match_id == match_id)
    if model_name:
        query = query.filter(Prediction.model_name == model_name)
    if model_version:
        query = query.filter(Prediction.model_version == model_version)
    predictions = query.order_by(Prediction.created_at.desc()).limit(limit).all()
    return [
        PredictionOut(
            match_id=pred.match_id,
            model_name=pred.model_name,
            model_version=pred.model_version,
            predicted_home_score=pred.predicted_home_score,
            predicted_away_score=pred.predicted_away_score,
            win_prob_home=pred.win_prob_home,
            win_prob_draw=pred.win_prob_draw,
            win_prob_away=pred.win_prob_away,
        )
        for pred in predictions
    ]


@router.get("/model-runs", response_model=list[ModelRunOut])
def list_model_runs(
    league: str | None = Query(default=None, description="League code"),
    season: str | None = Query(default=None, description="Season label"),
    feature_version: str | None = Query(default=None),
    model_name: str | None = Query(default=None),
    limit: int = Query(default=100, le=500),
    db: Session = Depends(get_session),
):
    query = db.query(ModelRun).options(joinedload(ModelRun.artifacts))
    if league:
        query = query.filter(ModelRun.league_code == league)
    if season:
        query = query.filter(ModelRun.season_label == season)
    if feature_version:
        query = query.filter(ModelRun.feature_version == feature_version)
    if model_name:
        query = query.filter(ModelRun.model_name == model_name)
    runs = query.order_by(ModelRun.created_at.desc()).limit(limit).all()
    response: list[ModelRunOut] = []
    for run in runs:
        artifacts = [
            ModelArtifactOut(
                artifact_type=artifact.artifact_type,
                path=artifact.path,
                checksum=artifact.checksum,
                size_bytes=artifact.size_bytes,
            )
            for artifact in run.artifacts
        ]
        response.append(
            ModelRunOut(
                run_id=run.run_id,
                model_name=run.model_name,
                model_version=run.model_version,
                feature_version=run.feature_version,
                commit_hash=run.commit_hash,
                league_code=run.league_code,
                season_label=run.season_label,
                status=run.status,
                metrics=run.metrics,
                params=run.params,
                data_summary=run.data_summary,
                created_at=run.created_at,
                artifacts=artifacts,
            )
        )
    return response


def _run_dir_from_artifacts(artifacts: list[ModelArtifactOut] | list[Any]) -> Path | None:
    if not artifacts:
        return None
    for artifact in artifacts:
        path = getattr(artifact, "path", None) if not isinstance(artifact, dict) else artifact.get("path")
        if not isinstance(path, str):
            continue
        candidate = MODEL_DIR / path
        if candidate.exists():
            return candidate.parent
    return None


@router.post("/model-runs/{run_id}/clear")
def clear_model_run(
    run_id: str,
    db: Session = Depends(get_session),
):
    run = (
        db.query(ModelRun)
        .options(joinedload(ModelRun.artifacts))
        .filter(ModelRun.run_id == run_id)
        .one_or_none()
    )
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")

    run.status = "cleared"
    db.add(run)
    db.commit()

    flag_path = None
    run_dir = _run_dir_from_artifacts(run.artifacts)
    if run_dir and run_dir.exists():
        flag_path = run_dir / "CLEARED.flag"
        timestamp = datetime.now(timezone.utc).isoformat()
        flag_path.write_text(f"cleared_at: {timestamp}\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "status": run.status,
        "flag_path": flag_path.as_posix() if flag_path else None,
    }


@router.post("/model-runs/{run_id}/promote")
def promote_model_run(
    run_id: str,
    db: Session = Depends(get_session),
):
    run = (
        db.query(ModelRun)
        .options(joinedload(ModelRun.artifacts))
        .filter(ModelRun.run_id == run_id)
        .one_or_none()
    )
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")

    promoted: list[dict[str, Any]] = []
    for artifact in run.artifacts:
        source_path = MODEL_DIR / artifact.path
        if not source_path.exists():
            promoted.append(
                {
                    "source": artifact.path,
                    "destination": None,
                    "status": "missing",
                }
            )
            continue

        destination = MODEL_DIR / source_path.name
        if destination.exists():
            promoted.append(
                {
                    "source": artifact.path,
                    "destination": destination.name,
                    "status": "skipped",
                }
            )
            continue

        if source_path.suffix == ".json":
            try:
                payload = json.loads(source_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and "artifact" in payload:
                    payload["artifact"] = destination.name
                destination.write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
            except Exception:
                shutil.copy2(source_path, destination)
        else:
            shutil.copy2(source_path, destination)

        promoted.append(
            {
                "source": artifact.path,
                "destination": destination.name,
                "status": "copied",
            }
        )

    return {
        "run_id": run_id,
        "model_name": run.model_name,
        "model_version": run.model_version,
        "promoted": promoted,
    }


def _metric_direction(metric_name: str) -> bool:
    metric_name = metric_name.lower()
    if "log_loss" in metric_name or "brier" in metric_name or "ece" in metric_name:
        return True
    return False


def _run_train_per_league(args: SimpleNamespace) -> None:
    try:
        from app.ml import train_per_league as train_per_league_module

        train_per_league_module.run(args)
    except Exception as exc:
        print(f"Train-per-league failed: {exc}")


@router.post("/train-per-league/run")
def run_train_per_league(
    background_tasks: BackgroundTasks,
    trainer: Literal["main", "baseline"] = Query(default="main"),
    model: Literal["xgboost", "catboost"] = Query(default="xgboost"),
    version: str = Query(default="v1"),
    season: str | None = Query(default=None),
    train_ratio: float = Query(default=0.7),
    val_ratio: float = Query(default=0.15),
    tune: bool = Query(default=False),
    n_iter: int = Query(default=10, ge=1),
    calibrate: bool = Query(default=False),
    calibration_method: Literal["sigmoid", "isotonic"] = Query(default="sigmoid"),
    class_weight: Literal["none", "balanced"] = Query(default="balanced"),
    walk_forward: bool = Query(default=False),
    cv_splits: int = Query(default=0, ge=0),
    seed: int = Query(default=42),
    allow_leakage: bool = Query(default=False),
    verbose_eval: int = Query(default=1, ge=0),
    early_stopping_rounds: int = Query(default=0, ge=0),
    n_estimators: int | None = Query(default=None, ge=1),
    heartbeat_seconds: int = Query(default=30, ge=0),
    skip_existing: bool = Query(default=True),
    best_metric: str = Query(default="val_log_loss"),
    remember_best: bool = Query(default=True),
    continue_on_error: bool = Query(default=True),
):
    if train_ratio + val_ratio >= 0.95:
        raise HTTPException(
            status_code=400, detail="train_ratio + val_ratio must be < 0.95"
        )

    args = SimpleNamespace(
        trainer=trainer,
        model=model,
        version=version,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        season=season,
        tune=tune,
        n_iter=n_iter,
        calibrate=calibrate,
        calibration_method=calibration_method,
        class_weight=class_weight,
        walk_forward=walk_forward,
        cv_splits=cv_splits,
        seed=seed,
        allow_leakage=allow_leakage,
        verbose_eval=verbose_eval,
        early_stopping_rounds=early_stopping_rounds,
        n_estimators=n_estimators,
        heartbeat_seconds=heartbeat_seconds,
        skip_existing=skip_existing,
        best_metric=best_metric,
        remember_best=remember_best,
        continue_on_error=continue_on_error,
    )

    background_tasks.add_task(_run_train_per_league, args)
    return {
        "status": "running",
        "trainer": trainer,
        "model": model,
        "version": version,
        "season": season,
        "skip_existing": skip_existing,
    }


@router.get("/model-runs/top", response_model=list[ModelRunSummaryOut])
def top_model_runs(
    league: str | None = Query(default=None, description="League code"),
    season: str | None = Query(default=None, description="Season label"),
    feature_version: str | None = Query(default=None),
    model_name: str | None = Query(default=None),
    metric: str = Query(default="test_log_loss"),
    limit: int = Query(default=5, le=100),
    db: Session = Depends(get_session),
):
    query = db.query(ModelRun)
    if league:
        query = query.filter(ModelRun.league_code == league)
    if season:
        query = query.filter(ModelRun.season_label == season)
    if feature_version:
        query = query.filter(ModelRun.feature_version == feature_version)
    if model_name:
        query = query.filter(ModelRun.model_name == model_name)
    runs = query.all()
    candidates: list[tuple[ModelRun, float]] = []
    for run in runs:
        value = None
        if run.metrics and metric in run.metrics:
            raw_value = run.metrics.get(metric)
            if isinstance(raw_value, (int, float)):
                value = float(raw_value)
        if value is not None:
            candidates.append((run, value))

    ascending = _metric_direction(metric)
    candidates.sort(key=lambda item: item[1], reverse=not ascending)
    results: list[ModelRunSummaryOut] = []
    for run, value in candidates[:limit]:
        results.append(
            ModelRunSummaryOut(
                run_id=run.run_id,
                model_name=run.model_name,
                model_version=run.model_version,
                feature_version=run.feature_version,
                league_code=run.league_code,
                season_label=run.season_label,
                metric=value,
                metric_name=metric,
                created_at=run.created_at,
            )
        )
    return results


@router.get("/model-runs/best", response_model=list[ModelRunBestOut])
def best_model_runs(
    league: str | None = Query(default=None, description="League code"),
    season: str | None = Query(default=None, description="Season label"),
    feature_version: str | None = Query(default=None),
    model_name: str | None = Query(default=None),
):
    registry = load_registry(MODEL_DIR)
    results: list[ModelRunBestOut] = []
    for entry in registry.values():
        if not isinstance(entry, dict):
            continue
        entry_league = entry.get("league_code")
        entry_season = entry.get("season_label")
        entry_feature = entry.get("feature_version")
        entry_model = entry.get("model_name")

        if league is not None and entry_league != league:
            continue
        if season is not None and entry_season != season:
            continue
        if feature_version is not None and entry_feature != feature_version:
            continue
        if model_name is not None and entry_model != model_name:
            continue

        results.append(
            ModelRunBestOut(
                run_id=entry.get("run_id"),
                model_name=entry_model or "unknown",
                model_version=entry.get("model_version") or "unknown",
                feature_version=entry_feature or "unknown",
                league_code=entry_league,
                season_label=entry_season,
                metric_name=entry.get("metric_name"),
                metric_value=entry.get("metric_value"),
                direction=entry.get("direction"),
                updated_at=entry.get("updated_at"),
            )
        )
    return results


def _report_file_url(report_id: str, filename: str) -> str:
    return f"/dataset-integrity/{report_id}/files/{filename}"


def _get_report_dir(report_id: str) -> Path:
    safe_id = Path(report_id).name
    report_dir = REPORT_DIR / safe_id
    if not report_dir.exists() or not report_dir.is_dir():
        raise HTTPException(status_code=404, detail="Report not found.")
    return report_dir


@router.get("/dataset-integrity", response_model=list[DatasetIntegrityReportOut])
def list_dataset_integrity_reports():
    if not REPORT_DIR.exists():
        return []
    reports: list[DatasetIntegrityReportOut] = []
    for report_dir in sorted(REPORT_DIR.iterdir(), reverse=True):
        if not report_dir.is_dir():
            continue
        summary_path = report_dir / "dataset_integrity.md"
        if not summary_path.exists():
            continue
        created_at = datetime.fromtimestamp(
            summary_path.stat().st_mtime, tz=timezone.utc
        )
        report_id = report_dir.name
        missingness = report_dir / "missingness_train_val.png"
        golden = report_dir / "golden_sample.csv"
        reports.append(
            DatasetIntegrityReportOut(
                report_id=report_id,
                created_at=created_at,
                summary_url=_report_file_url(report_id, "dataset_integrity.md"),
                missingness_url=_report_file_url(report_id, "missingness_train_val.png")
                if missingness.exists()
                else None,
                golden_sample_url=_report_file_url(report_id, "golden_sample.csv")
                if golden.exists()
                else None,
            )
        )
    return reports


@router.get("/dataset-integrity/{report_id}/files/{filename}")
def get_dataset_integrity_file(report_id: str, filename: str):
    report_dir = _get_report_dir(report_id)
    safe_name = Path(filename).name
    file_path = report_dir / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found.")
    resolved = file_path.resolve()
    if report_dir.resolve() not in resolved.parents:
        raise HTTPException(status_code=404, detail="Report file not found.")
    return FileResponse(resolved)


def _label_audit_file_url(report_id: str, filename: str) -> str:
    return f"/label-audits/{report_id}/files/{filename}"


def _get_label_audit_dir(report_id: str) -> Path:
    safe_id = Path(report_id).name
    if not safe_id.startswith("label_audit_"):
        raise HTTPException(status_code=404, detail="Report not found.")
    report_dir = REPORT_DIR / safe_id
    if not report_dir.exists() or not report_dir.is_dir():
        raise HTTPException(status_code=404, detail="Report not found.")
    return report_dir


@router.get("/label-audits", response_model=list[LabelAuditReportOut])
def list_label_audits(
    league: str | None = Query(default=None, description="League code"),
    season: str | None = Query(default=None, description="Season label"),
):
    if not REPORT_DIR.exists():
        return []
    index_path = REPORT_DIR / "label_audits_index.json"
    reports: list[LabelAuditReportOut] = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            entries = []
        for entry in entries:
            report_id = entry.get("report_id")
            run_id = entry.get("run_id")
            created_at_raw = entry.get("created_at")
            if not report_id or not run_id or not created_at_raw:
                continue
            created_at = datetime.fromisoformat(created_at_raw)
            entry_league = entry.get("league_code")
            entry_season = entry.get("season_label")
            if league is not None and entry_league != league:
                continue
            if season is not None and entry_season != season:
                continue
            report_file = entry.get("report_file", "label_audit.md")
            chart_file = entry.get("chart_file", "predicted_vs_true.png")
            confusion_file = entry.get("confusion_file", "confusion_matrix.png")
            reports.append(
                LabelAuditReportOut(
                    report_id=report_id,
                    run_id=run_id,
                    created_at=created_at,
                    report_url=_label_audit_file_url(report_id, report_file),
                    chart_url=_label_audit_file_url(report_id, chart_file)
                    if chart_file
                    else None,
                    confusion_url=_label_audit_file_url(report_id, confusion_file)
                    if confusion_file
                    else None,
                    league_code=entry_league,
                    season_label=entry_season,
                )
            )
        return reports
    for report_dir in sorted(REPORT_DIR.glob("label_audit_*"), reverse=True):
        if not report_dir.is_dir():
            continue
        report_path = report_dir / "label_audit.md"
        if not report_path.exists():
            continue
        created_at = datetime.fromtimestamp(
            report_path.stat().st_mtime, tz=timezone.utc
        )
        report_id = report_dir.name
        run_id = report_id.replace("label_audit_", "", 1)
        chart_path = report_dir / "predicted_vs_true.png"
        reports.append(
            LabelAuditReportOut(
                report_id=report_id,
                run_id=run_id,
                created_at=created_at,
                report_url=_label_audit_file_url(report_id, "label_audit.md"),
                chart_url=_label_audit_file_url(report_id, "predicted_vs_true.png")
                if chart_path.exists()
                else None,
                confusion_url=_label_audit_file_url(report_id, "confusion_matrix.png")
                if (report_dir / "confusion_matrix.png").exists()
                else None,
                league_code=None,
                season_label=None,
            )
        )
    return reports


@router.get("/label-audits/{report_id}/files/{filename}")
def get_label_audit_file(report_id: str, filename: str):
    report_dir = _get_label_audit_dir(report_id)
    safe_name = Path(filename).name
    file_path = report_dir / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found.")
    resolved = file_path.resolve()
    if report_dir.resolve() not in resolved.parents:
        raise HTTPException(status_code=404, detail="Report file not found.")
    return FileResponse(resolved)


def _run_label_audit(run_id: str) -> None:
    try:
        from app.reporting import label_audit as label_audit_module

        with SessionLocal() as session:
            run = session.query(ModelRun).filter(ModelRun.run_id == run_id).one_or_none()
            if not run:
                return
            metrics = run.metrics or {}
            label_counts = None
            if isinstance(run.data_summary, dict):
                label_counts = run.data_summary.get("label_counts")

        label_audit_module.build_label_audit_from_run(
            run_id=run_id,
            metrics=metrics,
            label_counts=label_counts,
            league_code=run.league_code,
            season_label=run.season_label,
        )
    except Exception as exc:
        print(f"Label audit failed for run {run_id}: {exc}")


@router.post("/label-audits/run")
def run_label_audit(
    background_tasks: BackgroundTasks,
    run_id: str = Query(..., description="Model run id"),
    db: Session = Depends(get_session),
):
    run = db.query(ModelRun).filter(ModelRun.run_id == run_id).one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")
    metrics = run.metrics or {}
    if "val_confusion_matrix" not in metrics:
        raise HTTPException(
            status_code=400,
            detail="Run metrics missing val_confusion_matrix; re-train with validation metrics enabled.",
        )
    background_tasks.add_task(_run_label_audit, run_id)
    return {"status": "running", "run_id": run_id}


def _divergence_file_url(report_id: str, filename: str) -> str:
    return f"/training-divergence/{report_id}/files/{filename}"


def _get_divergence_dir(report_id: str) -> Path:
    safe_id = Path(report_id).name
    if not safe_id.startswith("training_divergence_"):
        raise HTTPException(status_code=404, detail="Report not found.")
    report_dir = REPORT_DIR / safe_id
    if not report_dir.exists() or not report_dir.is_dir():
        raise HTTPException(status_code=404, detail="Report not found.")
    return report_dir


@router.get("/training-divergence", response_model=list[TrainingDivergenceReportOut])
def list_training_divergence_reports(
    league: str | None = Query(default=None, description="League code"),
    season: str | None = Query(default=None, description="Season label"),
):
    if not REPORT_DIR.exists():
        return []
    index_path = REPORT_DIR / "training_divergence_index.json"
    reports: list[TrainingDivergenceReportOut] = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            entries = []
        for entry in entries:
            report_id = entry.get("report_id")
            run_id = entry.get("run_id")
            created_at_raw = entry.get("created_at")
            if not report_id or not run_id or not created_at_raw:
                continue
            entry_league = entry.get("league_code")
            entry_season = entry.get("season_label")
            if league is not None and entry_league != league:
                continue
            if season is not None and entry_season != season:
                continue
            created_at = datetime.fromisoformat(created_at_raw)
            report_file = entry.get("report_file", "training_divergence.md")
            loss_chart_file = entry.get("loss_chart_file", "loss_lr.png")
            grad_chart_file = entry.get("grad_chart_file")
            reports.append(
                TrainingDivergenceReportOut(
                    report_id=report_id,
                    run_id=run_id,
                    created_at=created_at,
                    report_url=_divergence_file_url(report_id, report_file),
                    loss_chart_url=_divergence_file_url(report_id, loss_chart_file)
                    if loss_chart_file
                    else None,
                    grad_chart_url=_divergence_file_url(report_id, grad_chart_file)
                    if grad_chart_file
                    else None,
                    league_code=entry_league,
                    season_label=entry_season,
                )
            )
        return reports
    for report_dir in sorted(REPORT_DIR.glob("training_divergence_*"), reverse=True):
        if not report_dir.is_dir():
            continue
        report_path = report_dir / "training_divergence.md"
        if not report_path.exists():
            continue
        created_at = datetime.fromtimestamp(
            report_path.stat().st_mtime, tz=timezone.utc
        )
        report_id = report_dir.name
        run_id = report_id.replace("training_divergence_", "", 1)
        loss_chart = report_dir / "loss_lr.png"
        grad_chart = report_dir / "grad_norm.png"
        reports.append(
            TrainingDivergenceReportOut(
                report_id=report_id,
                run_id=run_id,
                created_at=created_at,
                report_url=_divergence_file_url(report_id, "training_divergence.md"),
                loss_chart_url=_divergence_file_url(report_id, "loss_lr.png")
                if loss_chart.exists()
                else None,
                grad_chart_url=_divergence_file_url(report_id, "grad_norm.png")
                if grad_chart.exists()
                else None,
                league_code=None,
                season_label=None,
            )
        )
    return reports


@router.get("/training-divergence/{report_id}/files/{filename}")
def get_training_divergence_file(report_id: str, filename: str):
    report_dir = _get_divergence_dir(report_id)
    safe_name = Path(filename).name
    file_path = report_dir / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found.")
    resolved = file_path.resolve()
    if report_dir.resolve() not in resolved.parents:
        raise HTTPException(status_code=404, detail="Report file not found.")
    return FileResponse(resolved)


def _run_training_divergence(run_id: str) -> None:
    try:
        from app.reporting import training_divergence as divergence_module

        with SessionLocal() as session:
            run = session.query(ModelRun).filter(ModelRun.run_id == run_id).one_or_none()
            if not run:
                return

        divergence_module.build_report_from_run(run)
    except Exception as exc:
        print(f"Training divergence failed for run {run_id}: {exc}")


@router.post("/training-divergence/run")
def run_training_divergence(
    background_tasks: BackgroundTasks,
    run_id: str = Query(..., description="Model run id"),
    db: Session = Depends(get_session),
):
    run = db.query(ModelRun).filter(ModelRun.run_id == run_id).one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")
    background_tasks.add_task(_run_training_divergence, run_id)
    return {"status": "running", "run_id": run_id}


MODEL_EVAL_INDEX = REPORT_DIR / "model_eval_index.json"
MODEL_EVAL_DIR = REPORT_DIR / "model_eval"


def _model_eval_file_url(report_id: str, filename: str) -> str:
    return f"/model-eval/{report_id}/files/{filename}"


def _get_model_eval_dir(report_id: str) -> Path:
    safe_id = Path(report_id).name
    report_dir = MODEL_EVAL_DIR / safe_id
    if not report_dir.exists() or not report_dir.is_dir():
        raise HTTPException(status_code=404, detail="Report not found.")
    return report_dir


def _load_model_eval_index() -> list[dict]:
    if not MODEL_EVAL_INDEX.exists():
        return []
    try:
        payload = json.loads(MODEL_EVAL_INDEX.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return payload
    return []


def _write_model_eval_index(entries: list[dict]) -> None:
    MODEL_EVAL_INDEX.write_text(
        json.dumps(entries, indent=2), encoding="utf-8"
    )


def _upsert_model_eval_entry(entry: dict) -> None:
    entries = _load_model_eval_index()
    report_id = entry.get("report_id")
    if not report_id:
        return
    updated = []
    replaced = False
    for existing in entries:
        if existing.get("report_id") == report_id:
            updated.append(entry)
            replaced = True
        else:
            updated.append(existing)
    if not replaced:
        updated.insert(0, entry)
    _write_model_eval_index(updated)


def _run_model_eval_report(
    *,
    report_id: str,
    run_id: str,
    model_name: str,
    model_version: str,
    feature_version: str,
    league_code: str | None,
    season_label: str | None,
    baseline_name: str,
) -> None:
    entry = {
        "report_id": report_id,
        "run_id": run_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "status": "running",
        "league_code": league_code,
        "season_label": season_label,
        "model_name": model_name,
        "model_version": model_version,
        "baseline_name": baseline_name,
        "feature_version": feature_version,
    }
    _upsert_model_eval_entry(entry)

    output_dir = MODEL_EVAL_DIR / report_id
    try:
        from app.reporting import model_eval as model_eval_module

        model_eval_module.build_report(
            version=feature_version,
            model_name=model_name,
            baseline_name=baseline_name,
            league_code=league_code,
            season_label=season_label,
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42,
            lookup="best",
            model_version=model_version,
            baseline_version=None,
            output_dir=output_dir,
            overconfidence_threshold=0.99,
            missingness_rate=0.05,
            missingness_mode="both",
            bins=20,
        )

        payload: dict[str, Any] = {}
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        main_payload = payload.get("model_main", {}) if isinstance(payload, dict) else {}
        baseline_payload = (
            payload.get("model_baseline", {}) if isinstance(payload, dict) else {}
        )

        completed = {
            **entry,
            "status": "completed",
            "model_name": main_payload.get("name") or model_name,
            "model_version": main_payload.get("version") or model_version,
            "baseline_name": baseline_payload.get("name") or baseline_name,
            "baseline_version": baseline_payload.get("version"),
            "report_file": "report.md" if (output_dir / "report.md").exists() else None,
            "metrics_json": "metrics.json" if (output_dir / "metrics.json").exists() else None,
            "metrics_csv": "metrics.csv" if (output_dir / "metrics.csv").exists() else None,
            "confusion_matrix": "confusion_matrix.png"
            if (output_dir / "confusion_matrix.png").exists()
            else None,
            "confusion_matrix_baseline": "confusion_matrix_baseline.png"
            if (output_dir / "confusion_matrix_baseline.png").exists()
            else None,
            "calibration": "calibration.png"
            if (output_dir / "calibration.png").exists()
            else None,
            "entropy_hist": "entropy_hist.png"
            if (output_dir / "entropy_hist.png").exists()
            else None,
            "max_prob_hist": "max_prob_hist.png"
            if (output_dir / "max_prob_hist.png").exists()
            else None,
            "subsets": "subsets.csv"
            if (output_dir / "subsets.csv").exists()
            else None,
        }
        _upsert_model_eval_entry(completed)
    except Exception as exc:
        failed = {**entry, "status": "failed", "error": str(exc)}
        _upsert_model_eval_entry(failed)


@router.get("/model-eval", response_model=list[ModelEvalReportOut])
def list_model_eval_reports(
    league: str | None = Query(default=None, description="League code"),
    season: str | None = Query(default=None, description="Season label"),
):
    entries = _load_model_eval_index()
    if not entries and MODEL_EVAL_DIR.exists():
        entries = []
        for report_dir in sorted(MODEL_EVAL_DIR.iterdir(), reverse=True):
            if not report_dir.is_dir():
                continue
            metrics_path = report_dir / "metrics.json"
            created_at = datetime.fromtimestamp(
                metrics_path.stat().st_mtime, tz=timezone.utc
            )
            model_name = None
            model_version = None
            baseline_name = None
            baseline_version = None
            if metrics_path.exists():
                try:
                    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                    main_payload = payload.get("model_main", {})
                    baseline_payload = payload.get("model_baseline", {})
                    model_name = main_payload.get("name")
                    model_version = main_payload.get("version")
                    baseline_name = baseline_payload.get("name")
                    baseline_version = baseline_payload.get("version")
                    created_at_raw = payload.get("created_at")
                    if created_at_raw:
                        created_at = datetime.fromisoformat(created_at_raw)
                except json.JSONDecodeError:
                    pass
            entries.append(
                {
                    "report_id": report_dir.name,
                    "created_at": created_at.isoformat(),
                    "status": "completed",
                    "model_name": model_name,
                    "model_version": model_version,
                    "baseline_name": baseline_name,
                    "baseline_version": baseline_version,
                    "report_file": "report.md" if (report_dir / "report.md").exists() else None,
                    "metrics_json": "metrics.json" if metrics_path.exists() else None,
                    "metrics_csv": "metrics.csv" if (report_dir / "metrics.csv").exists() else None,
                    "confusion_matrix": "confusion_matrix.png"
                    if (report_dir / "confusion_matrix.png").exists()
                    else None,
                    "confusion_matrix_baseline": "confusion_matrix_baseline.png"
                    if (report_dir / "confusion_matrix_baseline.png").exists()
                    else None,
                    "calibration": "calibration.png"
                    if (report_dir / "calibration.png").exists()
                    else None,
                    "entropy_hist": "entropy_hist.png"
                    if (report_dir / "entropy_hist.png").exists()
                    else None,
                    "max_prob_hist": "max_prob_hist.png"
                    if (report_dir / "max_prob_hist.png").exists()
                    else None,
                    "subsets": "subsets.csv"
                    if (report_dir / "subsets.csv").exists()
                    else None,
                }
            )

    reports: list[ModelEvalReportOut] = []
    for entry in entries:
        entry_league = entry.get("league_code")
        entry_season = entry.get("season_label")
        if league is not None and entry_league != league:
            continue
        if season is not None and entry_season != season:
            continue
        created_at_raw = entry.get("created_at")
        if created_at_raw:
            created_at = datetime.fromisoformat(created_at_raw)
        else:
            created_at = datetime.now(tz=timezone.utc)
        report_id = entry.get("report_id") or "unknown"
        report_file = entry.get("report_file")
        reports.append(
            ModelEvalReportOut(
                report_id=report_id,
                run_id=entry.get("run_id"),
                created_at=created_at,
                status=entry.get("status", "completed"),
                report_url=_model_eval_file_url(report_id, report_file)
                if report_file
                else None,
                metrics_json_url=_model_eval_file_url(report_id, entry["metrics_json"])
                if entry.get("metrics_json")
                else None,
                metrics_csv_url=_model_eval_file_url(report_id, entry["metrics_csv"])
                if entry.get("metrics_csv")
                else None,
                confusion_url=_model_eval_file_url(report_id, entry["confusion_matrix"])
                if entry.get("confusion_matrix")
                else None,
                confusion_baseline_url=_model_eval_file_url(
                    report_id, entry["confusion_matrix_baseline"]
                )
                if entry.get("confusion_matrix_baseline")
                else None,
                calibration_url=_model_eval_file_url(report_id, entry["calibration"])
                if entry.get("calibration")
                else None,
                entropy_hist_url=_model_eval_file_url(report_id, entry["entropy_hist"])
                if entry.get("entropy_hist")
                else None,
                max_prob_hist_url=_model_eval_file_url(report_id, entry["max_prob_hist"])
                if entry.get("max_prob_hist")
                else None,
                subsets_url=_model_eval_file_url(report_id, entry["subsets"])
                if entry.get("subsets")
                else None,
                league_code=entry.get("league_code"),
                season_label=entry.get("season_label"),
                model_name=entry.get("model_name"),
                model_version=entry.get("model_version"),
                baseline_name=entry.get("baseline_name"),
                baseline_version=entry.get("baseline_version"),
                feature_version=entry.get("feature_version"),
                error=entry.get("error"),
            )
        )

    reports.sort(key=lambda item: item.created_at, reverse=True)
    return reports


@router.post("/model-eval/run")
def run_model_eval_report(
    background_tasks: BackgroundTasks,
    run_id: str = Query(..., description="Model run id"),
    db: Session = Depends(get_session),
):
    run = db.query(ModelRun).filter(ModelRun.run_id == run_id).one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_id = Path(f"{timestamp}_{run_id}").name

    baseline_name = "baseline_classifier"
    background_tasks.add_task(
        _run_model_eval_report,
        report_id=report_id,
        run_id=run.run_id,
        model_name=run.model_name,
        model_version=run.model_version,
        feature_version=run.feature_version,
        league_code=run.league_code,
        season_label=run.season_label,
        baseline_name=baseline_name,
    )

    return {"status": "running", "report_id": report_id}


@router.get("/model-eval/{report_id}/files/{filename}")
def get_model_eval_file(report_id: str, filename: str):
    report_dir = _get_model_eval_dir(report_id)
    safe_name = Path(filename).name
    file_path = report_dir / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found.")
    resolved = file_path.resolve()
    if report_dir.resolve() not in resolved.parents:
        raise HTTPException(status_code=404, detail="Report file not found.")
    return FileResponse(resolved)


TUNING_SWEEP_INDEX = REPORT_DIR / "tuning_sweep_index.json"
TUNING_SWEEP_DIR = REPORT_DIR / "tuning_sweep"


def _tuning_sweep_file_url(report_id: str, filename: str) -> str:
    return f"/tuning-sweep/{report_id}/files/{filename}"


def _get_tuning_sweep_dir(report_id: str) -> Path:
    safe_id = Path(report_id).name
    report_dir = TUNING_SWEEP_DIR / safe_id
    if not report_dir.exists() or not report_dir.is_dir():
        raise HTTPException(status_code=404, detail="Report not found.")
    return report_dir


def _load_tuning_sweep_index() -> list[dict]:
    if not TUNING_SWEEP_INDEX.exists():
        return []
    try:
        payload = json.loads(TUNING_SWEEP_INDEX.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return payload
    return []


def _write_tuning_sweep_index(entries: list[dict]) -> None:
    TUNING_SWEEP_INDEX.write_text(
        json.dumps(entries, indent=2), encoding="utf-8"
    )


def _upsert_tuning_sweep_entry(entry: dict) -> None:
    entries = _load_tuning_sweep_index()
    report_id = entry.get("report_id")
    if not report_id:
        return
    updated = []
    replaced = False
    for existing in entries:
        if existing.get("report_id") == report_id:
            updated.append(entry)
            replaced = True
        else:
            updated.append(existing)
    if not replaced:
        updated.insert(0, entry)
    _write_tuning_sweep_index(updated)


def _run_tuning_sweep_report(
    *,
    report_id: str,
    trainer: str,
    model: str,
    feature_version: str,
    season_label: str | None,
    league_codes: list[str],
    n_iter: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    class_weight: str,
    calibrate: bool,
    calibration_method: str,
    n_estimators: int | None,
    iterations: int | None,
    fast_mode: bool,
    continue_on_error: bool,
) -> None:
    entry = {
        "report_id": report_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "status": "running",
        "trainer": trainer,
        "model": model,
        "feature_version": feature_version,
        "season_label": season_label,
        "league_codes": league_codes,
        "n_iter": n_iter,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "class_weight": class_weight,
        "calibration": calibrate,
        "calibration_method": calibration_method if calibrate else None,
        "n_estimators": n_estimators,
        "iterations": iterations,
        "fast_mode": fast_mode,
    }
    _upsert_tuning_sweep_entry(entry)

    output_dir = TUNING_SWEEP_DIR / report_id
    try:
        from app.ml import tuning_sweep as tuning_sweep_module

        config = tuning_sweep_module.SweepConfig(
            trainer=trainer,
            model=model,
            version=feature_version,
            league_codes=league_codes,
            season_label=season_label,
            n_iter=n_iter,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            class_weight_mode=class_weight,
            calibrate=calibrate,
            calibration_method=calibration_method,
            n_estimators=n_estimators,
            iterations=iterations,
            fast_mode=fast_mode,
            output_dir=output_dir,
            continue_on_error=continue_on_error,
        )
        tuning_sweep_module.run_sweep(config)

        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists():
            try:
                meta_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(meta_payload, dict):
                    for key in [
                        "trainer",
                        "model",
                        "feature_version",
                        "season_label",
                        "league_codes",
                        "n_iter",
                        "seed",
                        "train_ratio",
                        "val_ratio",
                        "class_weight",
                        "calibration",
                        "calibration_method",
                        "n_estimators",
                        "iterations",
                        "fast_mode",
                    ]:
                        if key in meta_payload:
                            entry[key] = meta_payload.get(key)
                    if meta_payload.get("created_at"):
                        entry["created_at"] = meta_payload.get("created_at")
            except json.JSONDecodeError:
                pass

        completed = {
            **entry,
            "status": "completed",
            "results_csv": "sweep_results.csv"
            if (output_dir / "sweep_results.csv").exists()
            else None,
            "results_tsv": "sweep_results.tsv"
            if (output_dir / "sweep_results.tsv").exists()
            else None,
            "results_md": "sweep_results.md"
            if (output_dir / "sweep_results.md").exists()
            else None,
            "summary_by_league": "summary_by_league.csv"
            if (output_dir / "summary_by_league.csv").exists()
            else None,
            "best_by_league": "best_by_league.csv"
            if (output_dir / "best_by_league.csv").exists()
            else None,
            "metadata": "metadata.json"
            if (output_dir / "metadata.json").exists()
            else None,
            "errors": "errors.csv"
            if (output_dir / "errors.csv").exists()
            else None,
        }
        _upsert_tuning_sweep_entry(completed)
    except Exception as exc:
        progress_path = output_dir / "progress.json"
        if progress_path.exists():
            try:
                payload = json.loads(progress_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            payload["status"] = "failed"
            payload["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
            progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        failed = {**entry, "status": "failed", "error": str(exc)}
        _upsert_tuning_sweep_entry(failed)


@router.get("/tuning-sweep", response_model=list[TuningSweepReportOut])
def list_tuning_sweep_reports(
    league: str | None = Query(default=None, description="League code"),
    season: str | None = Query(default=None, description="Season label"),
):
    entries = _load_tuning_sweep_index()
    if not entries and TUNING_SWEEP_DIR.exists():
        entries = []
        for report_dir in sorted(TUNING_SWEEP_DIR.iterdir(), reverse=True):
            if not report_dir.is_dir():
                continue
            metadata_path = report_dir / "metadata.json"
            created_at = datetime.fromtimestamp(
                report_dir.stat().st_mtime, tz=timezone.utc
            )
            entry: dict[str, Any] = {
                "report_id": report_dir.name,
                "created_at": created_at.isoformat(),
                "status": "completed",
            }
            if metadata_path.exists():
                try:
                    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        entry.update(payload)
                        created_at_raw = payload.get("created_at")
                        if created_at_raw:
                            entry["created_at"] = created_at_raw
                except json.JSONDecodeError:
                    pass
            entry.update(
                {
                    "results_csv": "sweep_results.csv"
                    if (report_dir / "sweep_results.csv").exists()
                    else None,
                    "results_tsv": "sweep_results.tsv"
                    if (report_dir / "sweep_results.tsv").exists()
                    else None,
                    "results_md": "sweep_results.md"
                    if (report_dir / "sweep_results.md").exists()
                    else None,
                    "summary_by_league": "summary_by_league.csv"
                    if (report_dir / "summary_by_league.csv").exists()
                    else None,
                    "best_by_league": "best_by_league.csv"
                    if (report_dir / "best_by_league.csv").exists()
                    else None,
                    "metadata": "metadata.json"
                    if metadata_path.exists()
                    else None,
                    "errors": "errors.csv"
                    if (report_dir / "errors.csv").exists()
                    else None,
                }
            )
            entries.append(entry)

    reports: list[TuningSweepReportOut] = []
    for entry in entries:
        entry_season = entry.get("season_label")
        entry_leagues = entry.get("league_codes")
        if season is not None and entry_season != season:
            continue
        if league is not None:
            if not isinstance(entry_leagues, list) or league not in entry_leagues:
                continue
        created_at_raw = entry.get("created_at")
        if created_at_raw:
            created_at = datetime.fromisoformat(created_at_raw)
        else:
            created_at = datetime.now(tz=timezone.utc)
        report_id = entry.get("report_id") or "unknown"
        report_dir = TUNING_SWEEP_DIR / Path(report_id).name
        progress_payload: dict[str, Any] = {}
        progress_path = report_dir / "progress.json"
        if progress_path.exists():
            try:
                progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                progress_payload = {}
        status = progress_payload.get("status") or entry.get("status", "completed")
        started_at_raw = progress_payload.get("started_at")
        updated_at_raw = progress_payload.get("updated_at")
        started_at = (
            datetime.fromisoformat(started_at_raw)
            if started_at_raw
            else None
        )
        updated_at = (
            datetime.fromisoformat(updated_at_raw)
            if updated_at_raw
            else None
        )
        results_csv = entry.get("results_csv")
        results_tsv = entry.get("results_tsv")
        results_md = entry.get("results_md")
        summary_by_league = entry.get("summary_by_league")
        best_by_league = entry.get("best_by_league")
        metadata_file = entry.get("metadata")
        errors_file = entry.get("errors")
        if report_dir.exists():
            if (report_dir / "sweep_results.csv").exists():
                results_csv = "sweep_results.csv"
            if (report_dir / "sweep_results.tsv").exists():
                results_tsv = "sweep_results.tsv"
            if (report_dir / "sweep_results.md").exists():
                results_md = "sweep_results.md"
            if (report_dir / "summary_by_league.csv").exists():
                summary_by_league = "summary_by_league.csv"
            if (report_dir / "best_by_league.csv").exists():
                best_by_league = "best_by_league.csv"
            if (report_dir / "metadata.json").exists():
                metadata_file = "metadata.json"
            if (report_dir / "errors.csv").exists():
                errors_file = "errors.csv"
        reports.append(
            TuningSweepReportOut(
                report_id=report_id,
                created_at=created_at,
                status=status,
                started_at=started_at,
                updated_at=updated_at,
                percent=progress_payload.get("percent"),
                total_iterations=progress_payload.get("total_iterations"),
                completed_iterations=progress_payload.get("completed_iterations"),
                current_league=progress_payload.get("current_league"),
                current_iteration=progress_payload.get("current_iteration"),
                elapsed_seconds=progress_payload.get("elapsed_seconds"),
                eta_seconds=progress_payload.get("eta_seconds"),
                errors_count=progress_payload.get("errors_count"),
                trainer=entry.get("trainer"),
                model=entry.get("model"),
                feature_version=entry.get("feature_version"),
                season_label=entry_season,
                league_codes=entry_leagues if isinstance(entry_leagues, list) else None,
                n_iter=entry.get("n_iter"),
                seed=entry.get("seed"),
                train_ratio=entry.get("train_ratio"),
                val_ratio=entry.get("val_ratio"),
                class_weight=entry.get("class_weight"),
                calibration=entry.get("calibration"),
                calibration_method=entry.get("calibration_method"),
                n_estimators=entry.get("n_estimators"),
                iterations=entry.get("iterations"),
                fast_mode=entry.get("fast_mode"),
                results_csv_url=_tuning_sweep_file_url(
                    report_id, results_csv
                )
                if results_csv
                else None,
                results_tsv_url=_tuning_sweep_file_url(
                    report_id, results_tsv
                )
                if results_tsv
                else None,
                results_md_url=_tuning_sweep_file_url(
                    report_id, results_md
                )
                if results_md
                else None,
                summary_by_league_url=_tuning_sweep_file_url(
                    report_id, summary_by_league
                )
                if summary_by_league
                else None,
                best_by_league_url=_tuning_sweep_file_url(
                    report_id, best_by_league
                )
                if best_by_league
                else None,
                metadata_url=_tuning_sweep_file_url(
                    report_id, metadata_file
                )
                if metadata_file
                else None,
                errors_url=_tuning_sweep_file_url(report_id, errors_file)
                if errors_file
                else None,
                error=entry.get("error"),
            )
        )

    reports.sort(key=lambda item: item.created_at, reverse=True)
    return reports


@router.post("/tuning-sweep/run")
def run_tuning_sweep_report(
    background_tasks: BackgroundTasks,
    trainer: Literal["main", "baseline"] = Query(default="main"),
    model: Literal["xgboost", "catboost"] = Query(default="xgboost"),
    version: str = Query(default="v1"),
    season: str | None = Query(default=None),
    league: str | None = Query(default=None),
    n_iter: int = Query(default=300, ge=1),
    seed: int = Query(default=42),
    train_ratio: float = Query(default=0.7),
    val_ratio: float = Query(default=0.15),
    class_weight: Literal["none", "balanced"] = Query(default="balanced"),
    calibrate: bool = Query(default=False),
    calibration_method: Literal["sigmoid", "isotonic"] = Query(default="sigmoid"),
    n_estimators: int | None = Query(default=None, ge=1),
    iterations: int | None = Query(default=None, ge=1),
    fast: bool = Query(default=True),
    continue_on_error: bool = Query(default=True),
    db: Session = Depends(get_session),
):
    if train_ratio + val_ratio >= 0.95:
        raise HTTPException(
            status_code=400, detail="train_ratio + val_ratio must be < 0.95"
        )
    if league:
        league_codes = [code.strip() for code in league.split(",") if code.strip()]
    else:
        league_codes = [row.code for row in db.query(League.code).order_by(League.code.asc()).all()]
    if not league_codes:
        raise HTTPException(status_code=400, detail="No leagues available for sweep.")

    n_estimators_final = n_estimators
    iterations_final = iterations
    if fast and trainer == "main":
        if model == "xgboost" and n_estimators_final is None:
            n_estimators_final = 50
        if model == "catboost" and iterations_final is None:
            iterations_final = 200

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_id = Path(f"{timestamp}_tuning_sweep").name

    entry = {
        "report_id": report_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "status": "running",
        "trainer": trainer,
        "model": model,
        "feature_version": version,
        "season_label": season,
        "league_codes": league_codes,
        "n_iter": n_iter,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "class_weight": class_weight,
        "calibration": calibrate,
        "calibration_method": calibration_method if calibrate else None,
        "n_estimators": n_estimators_final,
        "iterations": iterations_final,
        "fast_mode": fast,
    }
    _upsert_tuning_sweep_entry(entry)

    background_tasks.add_task(
        _run_tuning_sweep_report,
        report_id=report_id,
        trainer=trainer,
        model=model,
        feature_version=version,
        season_label=season,
        league_codes=league_codes,
        n_iter=n_iter,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        class_weight=class_weight,
        calibrate=calibrate,
        calibration_method=calibration_method,
        n_estimators=n_estimators_final,
        iterations=iterations_final,
        fast_mode=fast,
        continue_on_error=continue_on_error,
    )

    return {"status": "running", "report_id": report_id}


@router.get("/tuning-sweep/{report_id}/files/{filename}")
def get_tuning_sweep_file(report_id: str, filename: str):
    report_dir = _get_tuning_sweep_dir(report_id)
    safe_name = Path(filename).name
    file_path = report_dir / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found.")
    resolved = file_path.resolve()
    if report_dir.resolve() not in resolved.parents:
        raise HTTPException(status_code=404, detail="Report file not found.")
    return FileResponse(resolved)
