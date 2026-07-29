from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import REPORT_DIR
from app.db.models import League
from app.db.session import SessionLocal
from app.ml.dataset import load_feature_dataset, temporal_split
from app.ml.labels import LABELS, LABEL_IDS, encode_labels, predict_class_ids
from app.ml.metrics import classification_metrics, class_priors
from app.ml.train_baseline import (
    build_pipeline as build_baseline_pipeline,
    compute_sample_weights as baseline_sample_weights,
    sample_param_candidates as sample_baseline_candidates,
)
from app.ml.train_main import (
    build_pipeline as build_main_pipeline,
    compute_sample_weights as main_sample_weights,
    sample_param_candidates as sample_main_candidates,
)


@dataclass
class SweepConfig:
    trainer: str
    model: str
    version: str
    league_codes: list[str]
    season_label: str | None
    n_iter: int
    seed: int
    train_ratio: float
    val_ratio: float
    class_weight_mode: str
    calibrate: bool
    calibration_method: str
    n_estimators: int | None
    iterations: int | None
    fast_mode: bool
    output_dir: Path
    continue_on_error: bool


def list_league_codes() -> list[str]:
    with SessionLocal() as session:
        rows = session.query(League.code).order_by(League.code.asc()).all()
    return [code for (code,) in rows]


def _validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    if train_ratio + val_ratio >= 0.95:
        raise ValueError("train_ratio + val_ratio must be < 0.95")


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("No rows.", encoding="utf-8")
        return
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---" for _ in headers]) + " |")
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_eta(seconds: float | None) -> float | None:
    if seconds is None or seconds < 0:
        return None
    return float(seconds)


def _candidate_list(config: SweepConfig) -> list[dict[str, Any]]:
    if config.trainer == "baseline":
        return sample_baseline_candidates(config.n_iter, config.seed)
    return sample_main_candidates(config.model, config.n_iter, config.seed)


def _apply_param_overrides(
    config: SweepConfig, params: dict[str, Any]
) -> dict[str, Any]:
    merged = params.copy()
    if config.trainer == "main" and config.model == "xgboost":
        if config.n_estimators is not None:
            merged["n_estimators"] = int(config.n_estimators)
    if config.trainer == "main" and config.model == "catboost":
        if config.iterations is not None:
            merged["iterations"] = int(config.iterations)
    if config.trainer == "baseline":
        merged["class_weight"] = None if config.class_weight_mode == "none" else "balanced"
    return merged


def _fit_and_eval(
    config: SweepConfig,
    *,
    split,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    label_ids = LABEL_IDS
    labels = LABELS

    if config.trainer == "baseline":
        pipeline = build_baseline_pipeline(params)
        if config.class_weight_mode == "balanced":
            weights = baseline_sample_weights(list(y_train), label_ids)
            pipeline.fit(split.X_train, y_train, model__sample_weight=weights)
        else:
            pipeline.fit(split.X_train, y_train)
    else:
        pipeline = build_main_pipeline(config.model, params)
        if config.class_weight_mode == "balanced":
            weights = main_sample_weights(list(y_train), label_ids)
            pipeline.fit(split.X_train, y_train, model__sample_weight=weights)
        else:
            pipeline.fit(split.X_train, y_train)

    if config.calibrate:
        from sklearn.calibration import CalibratedClassifierCV

        weight_fn = (
            main_sample_weights
            if config.trainer == "main"
            else baseline_sample_weights
        )
        sample_weight_val = None
        if config.class_weight_mode == "balanced":
            sample_weight_val = weight_fn(list(y_val), label_ids)
        try:
            calibrator = CalibratedClassifierCV(
                pipeline, method=config.calibration_method, cv="prefit"
            )
            calibrator.fit(split.X_val, y_val, sample_weight=sample_weight_val)
            model = calibrator
        except Exception:
            calibrator = CalibratedClassifierCV(
                pipeline, method=config.calibration_method, cv=3
            )
            sample_weight_train = None
            if config.class_weight_mode == "balanced":
                sample_weight_train = weight_fn(list(y_train), label_ids)
            calibrator.fit(split.X_train, y_train, sample_weight=sample_weight_train)
            model = calibrator
    else:
        model = pipeline

    val_prob = model.predict_proba(split.X_val)
    test_prob = model.predict_proba(split.X_test)
    val_pred = predict_class_ids(val_prob)
    test_pred = predict_class_ids(test_prob)

    metrics: dict[str, Any] = {}
    metrics.update(
        classification_metrics("val", y_val, val_pred, val_prob, label_ids, labels)
    )
    metrics.update(
        classification_metrics(
            "test", y_test, test_pred, test_prob, label_ids, labels
        )
    )
    return metrics


def run_sweep(config: SweepConfig) -> None:
    _validate_split_ratios(config.train_ratio, config.val_ratio)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = config.output_dir / "sweep_results.csv"
    results_tsv_path = config.output_dir / "sweep_results.tsv"
    progress_path = config.output_dir / "progress.json"
    started_at = datetime.now(tz=timezone.utc)
    total_iters = len(config.league_codes) * config.n_iter
    completed_iters = 0

    errors: list[dict[str, str]] = []

    candidates = _candidate_list(config)
    fieldnames = [
        "iteration",
        "league_code",
        "season_label",
        "trainer",
        "model",
        "params",
        "val_accuracy",
        "val_log_loss",
        "val_f1_macro",
        "test_accuracy",
        "test_log_loss",
        "test_f1_macro",
        "test_brier_score",
        "train_size",
        "val_size",
        "test_size",
        "majority_class",
    ]

    last_progress_ts = 0.0
    current_league = None
    current_iter = 0

    with results_path.open("w", newline="", encoding="utf-8") as csv_file, results_tsv_path.open(
        "w", newline="", encoding="utf-8"
    ) as tsv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        tsv_writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter="\t")
        csv_writer.writeheader()
        tsv_writer.writeheader()
        csv_file.flush()
        tsv_file.flush()

        _write_progress(
            progress_path,
            {
                "status": "running",
                "started_at": started_at.isoformat(),
                "updated_at": started_at.isoformat(),
                "total_iterations": total_iters,
                "completed_iterations": completed_iters,
                "percent": 0.0,
                "current_league": None,
                "current_iteration": 0,
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
                "errors_count": 0,
            },
        )

        log_every = max(1, min(50, config.n_iter // 10)) if config.n_iter else 1

        for league_code in config.league_codes:
            current_league = league_code
            current_iter = 0
            print(f"[tuning_sweep] Starting league {league_code} ({config.n_iter} iters)")
            try:
                df = load_feature_dataset(
                    config.version,
                    league_code=league_code,
                    season_label=config.season_label,
                    include_match_id=True,
                )
                if df.empty:
                    raise RuntimeError("No feature rows available for league.")
                split = temporal_split(
                    df, train_ratio=config.train_ratio, val_ratio=config.val_ratio
                )

                y_train = np.asarray(encode_labels(split.y_train))
                y_val = np.asarray(encode_labels(split.y_val))
                y_test = np.asarray(encode_labels(split.y_test))

                if len(set(y_train)) < len(LABELS):
                    raise RuntimeError(
                        "Train split is missing one or more result classes."
                    )

                class_priors_train = class_priors(list(y_train), LABEL_IDS)
                majority_class = int(
                    max(range(len(class_priors_train)), key=class_priors_train.__getitem__)
                )

                for idx, params in enumerate(candidates, start=1):
                    current_iter = idx
                    merged_params = _apply_param_overrides(config, params)
                    metrics = _fit_and_eval(
                        config,
                        split=split,
                        y_train=y_train,
                        y_val=y_val,
                        y_test=y_test,
                        params=merged_params,
                    )
                    row = {
                        "iteration": idx,
                        "league_code": league_code,
                        "season_label": config.season_label or "all",
                        "trainer": config.trainer,
                        "model": config.model if config.trainer == "main" else "baseline",
                        "params": json.dumps(merged_params, sort_keys=True),
                        "val_accuracy": metrics.get("val_accuracy"),
                        "val_log_loss": metrics.get("val_log_loss"),
                        "val_f1_macro": metrics.get("val_f1_macro"),
                        "test_accuracy": metrics.get("test_accuracy"),
                        "test_log_loss": metrics.get("test_log_loss"),
                        "test_f1_macro": metrics.get("test_f1_macro"),
                        "test_brier_score": metrics.get("test_brier_score"),
                        "train_size": int(len(split.X_train)),
                        "val_size": int(len(split.X_val)),
                        "test_size": int(len(split.X_test)),
                        "majority_class": majority_class,
                    }
                    csv_writer.writerow(row)
                    tsv_writer.writerow(row)
                    csv_file.flush()
                    tsv_file.flush()

                    completed_iters += 1
                    now = time.time()
                    if now - last_progress_ts >= 1.0 or completed_iters == total_iters:
                        elapsed = now - started_at.timestamp()
                        rate = completed_iters / elapsed if elapsed > 0 else 0.0
                        remaining = total_iters - completed_iters
                        eta = remaining / rate if rate > 0 else None
                        _write_progress(
                            progress_path,
                            {
                                "status": "running",
                                "started_at": started_at.isoformat(),
                                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
                                "total_iterations": total_iters,
                                "completed_iterations": completed_iters,
                                "percent": float(completed_iters / total_iters * 100)
                                if total_iters
                                else 0.0,
                                "current_league": current_league,
                                "current_iteration": current_iter,
                                "elapsed_seconds": float(elapsed),
                                "eta_seconds": _format_eta(eta),
                                "errors_count": len(errors),
                            },
                        )
                        last_progress_ts = now

                    if idx % log_every == 0 or idx == config.n_iter:
                        print(
                            f"[tuning_sweep] {league_code} iter {idx}/{config.n_iter} "
                            f"({completed_iters}/{total_iters} total)"
                        )
            except Exception as exc:
                message = str(exc) or exc.__class__.__name__
                if not config.continue_on_error:
                    raise
                errors.append({"league_code": league_code, "error": message})
                remaining = max(0, config.n_iter - current_iter)
                completed_iters += remaining
                now = time.time()
                elapsed = now - started_at.timestamp()
                rate = completed_iters / elapsed if elapsed > 0 else 0.0
                remaining = total_iters - completed_iters
                eta = remaining / rate if rate > 0 else None
                _write_progress(
                    progress_path,
                    {
                        "status": "running",
                        "started_at": started_at.isoformat(),
                        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
                        "total_iterations": total_iters,
                        "completed_iterations": completed_iters,
                        "percent": float(completed_iters / total_iters * 100)
                        if total_iters
                        else 0.0,
                        "current_league": current_league,
                        "current_iteration": current_iter,
                        "elapsed_seconds": float(elapsed),
                        "eta_seconds": _format_eta(eta),
                        "errors_count": len(errors),
                    },
                )
                last_progress_ts = now
                print(f"[tuning_sweep] Skipped league {league_code}: {message}")
                continue
            print(f"[tuning_sweep] Completed league {league_code}")

    results_df = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    md_path = config.output_dir / "sweep_results.md"
    _write_markdown_table(results_df, md_path)

    if not results_df.empty:
        summary = (
            results_df.groupby("league_code", dropna=False)
            .agg(
                count=("iteration", "count"),
                test_accuracy_mean=("test_accuracy", "mean"),
                test_accuracy_std=("test_accuracy", "std"),
                test_accuracy_min=("test_accuracy", "min"),
                test_accuracy_max=("test_accuracy", "max"),
                val_accuracy_mean=("val_accuracy", "mean"),
                val_accuracy_std=("val_accuracy", "std"),
                test_log_loss_mean=("test_log_loss", "mean"),
                test_log_loss_std=("test_log_loss", "std"),
            )
            .reset_index()
        )
        summary_path = config.output_dir / "summary_by_league.csv"
        summary.to_csv(summary_path, index=False)

        best_by_league = (
            results_df.sort_values(
                ["league_code", "test_accuracy"],
                ascending=[True, False],
            )
            .groupby("league_code", dropna=False)
            .head(1)
            .reset_index(drop=True)
        )
        best_path = config.output_dir / "best_by_league.csv"
        best_by_league.to_csv(best_path, index=False)

    if errors:
        error_path = config.output_dir / "errors.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False)

    meta = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "trainer": config.trainer,
        "model": config.model if config.trainer == "main" else "baseline",
        "feature_version": config.version,
        "season_label": config.season_label,
        "league_codes": config.league_codes,
        "n_iter": config.n_iter,
        "seed": config.seed,
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "class_weight": config.class_weight_mode,
        "calibration": config.calibrate,
        "calibration_method": config.calibration_method if config.calibrate else None,
        "n_estimators": config.n_estimators,
        "iterations": config.iterations,
        "fast_mode": config.fast_mode,
        "artifacts": {
            "results_csv": str(results_path),
            "results_tsv": str(results_tsv_path),
            "results_md": str(config.output_dir / "sweep_results.md"),
            "summary_by_league": str(config.output_dir / "summary_by_league.csv"),
            "best_by_league": str(config.output_dir / "best_by_league.csv"),
            "errors": str(config.output_dir / "errors.csv"),
        },
    }
    (config.output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    finished_at = datetime.now(tz=timezone.utc)
    elapsed = finished_at.timestamp() - started_at.timestamp()
    _write_progress(
        progress_path,
        {
            "status": "completed",
            "started_at": started_at.isoformat(),
            "updated_at": finished_at.isoformat(),
            "total_iterations": total_iters,
            "completed_iterations": completed_iters,
            "percent": 100.0 if total_iters else 0.0,
            "current_league": current_league,
            "current_iteration": current_iter,
            "elapsed_seconds": float(elapsed),
            "eta_seconds": 0.0,
            "errors_count": len(errors),
        },
    )

    print(f"Sweep results saved to {results_path}")
    if errors:
        print(f"Some leagues failed. See {config.output_dir / 'errors.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer",
        default="main",
        choices=["main", "baseline"],
        help="Sweep trainer type.",
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["xgboost", "catboost"],
        help="Model to sweep when trainer=main.",
    )
    parser.add_argument("--version", default="v1")
    parser.add_argument("--league", default=None)
    parser.add_argument("--season", default=None)
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--class-weight",
        default="balanced",
        choices=["none", "balanced"],
        help="Class weighting strategy for training (balanced uses inverse frequency).",
    )
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable probability calibration on the validation split.",
    )
    parser.add_argument(
        "--calibration-method",
        default="sigmoid",
        choices=["sigmoid", "isotonic"],
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Override the number of boosting rounds for XGBoost.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the number of boosting rounds for CatBoost.",
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fast sweep defaults (lower boosting rounds unless overridden).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: MODEL_DIR/reports/tuning_sweep/<timestamp>)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip leagues that fail and continue.",
    )
    args = parser.parse_args()

    _validate_split_ratios(args.train_ratio, args.val_ratio)

    if args.league:
        league_codes = [code.strip() for code in args.league.split(",") if code.strip()]
    else:
        league_codes = list_league_codes()

    if not league_codes:
        raise RuntimeError("No leagues found in the database.")

    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = REPORT_DIR / "tuning_sweep" / timestamp

    n_estimators = args.n_estimators
    iterations = args.iterations
    if args.fast and args.trainer == "main":
        if args.model == "xgboost" and n_estimators is None:
            n_estimators = 50
        if args.model == "catboost" and iterations is None:
            iterations = 200

    config = SweepConfig(
        trainer=args.trainer,
        model=args.model,
        version=args.version,
        league_codes=league_codes,
        season_label=args.season,
        n_iter=args.n_iter,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        class_weight_mode=args.class_weight,
        calibrate=args.calibrate,
        calibration_method=args.calibration_method,
        n_estimators=n_estimators,
        iterations=iterations,
        fast_mode=bool(args.fast),
        output_dir=output_dir,
        continue_on_error=bool(args.continue_on_error),
    )

    run_sweep(config)


if __name__ == "__main__":
    main()
