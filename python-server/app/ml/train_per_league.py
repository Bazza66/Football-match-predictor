from __future__ import annotations

import argparse
import threading
import time

from app.db.models import League, ModelRun
from app.db.session import SessionLocal
from app.ml import train_baseline, train_main


def list_league_codes() -> list[str]:
    with SessionLocal() as session:
        rows = session.query(League.code).order_by(League.code.asc()).all()
    return [code for (code,) in rows]


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.trainer == "baseline":
        return "baseline_classifier"
    if args.model == "catboost":
        return "main_catboost"
    return "main_xgboost"


def build_expected_params(args: argparse.Namespace) -> dict:
    expected: dict = {
        "tuning_enabled": bool(args.tune),
        "tuning_candidates": int(args.n_iter) if args.tune else 0,
        "walk_forward": bool(args.walk_forward) if args.tune else False,
        "calibration_enabled": bool(args.calibrate),
        "calibration_method": args.calibration_method if args.calibrate else None,
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "n_estimators": args.n_estimators if args.model == "xgboost" else None,
        "class_weight": args.class_weight,
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "cv_splits": int(args.cv_splits),
    }
    if args.trainer == "main":
        expected["model_type"] = args.model
        if not args.tune:
            expected["hyperparameters"] = train_main.DEFAULT_PARAMS[args.model]
    else:
        if not args.tune:
            expected["hyperparameters"] = train_baseline.DEFAULT_PARAMS
    return expected


def _values_match(expected: object, actual: object) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) < 1e-9
    return expected == actual


def params_match(run_params: dict | None, expected: dict) -> bool:
    if not run_params:
        return False
    for key, expected_value in expected.items():
        if key not in run_params:
            return False
        actual_value = run_params[key]
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            if expected_value != actual_value:
                return False
            continue
        if not _values_match(expected_value, actual_value):
            return False
    return True


def list_completed_leagues(
    model_name: str, feature_version: str, season_label: str | None, expected: dict
) -> set[str]:
    with SessionLocal() as session:
        query = session.query(ModelRun.league_code, ModelRun.params).filter(
            ModelRun.model_name == model_name,
            ModelRun.feature_version == feature_version,
            ModelRun.status == "completed",
        )
        if season_label is None:
            query = query.filter(ModelRun.season_label.is_(None))
        else:
            query = query.filter(ModelRun.season_label == season_label)
        rows = query.all()
    completed: set[str] = set()
    for code, params in rows:
        if code and params_match(params, expected):
            completed.add(code)
    return completed


def train_league_main(args: argparse.Namespace, league_code: str) -> None:
    train_main.train(
        args.version,
        model_name=args.model,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        league_code=league_code,
        season_label=args.season,
        tune=args.tune,
        n_iter=args.n_iter,
        calibrate=args.calibrate,
        calibration_method=args.calibration_method,
        walk_forward=args.walk_forward,
        seed=args.seed,
        allow_leakage=args.allow_leakage,
        verbose_eval=args.verbose_eval,
        early_stopping_rounds=args.early_stopping_rounds,
        n_estimators=args.n_estimators if args.model == "xgboost" else None,
        class_weight_mode=args.class_weight,
        cv_splits=args.cv_splits,
        best_metric=args.best_metric,
        remember_best=args.remember_best,
    )


def train_league_baseline(args: argparse.Namespace, league_code: str) -> None:
    train_baseline.train(
        args.version,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        league_code=league_code,
        season_label=args.season,
        tune=args.tune,
        n_iter=args.n_iter,
        calibrate=args.calibrate,
        calibration_method=args.calibration_method,
        walk_forward=args.walk_forward,
        seed=args.seed,
        allow_leakage=args.allow_leakage,
        class_weight_mode=args.class_weight,
        cv_splits=args.cv_splits,
        best_metric=args.best_metric,
        remember_best=args.remember_best,
    )


def _start_heartbeat(label: str, interval_seconds: int) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def _run() -> None:
        start = time.time()
        while not stop_event.wait(interval_seconds):
            elapsed = int(time.time() - start)
            print(f"... {label} running ({elapsed}s elapsed)", flush=True)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return stop_event, thread


def run(args: argparse.Namespace) -> None:
    if args.train_ratio + args.val_ratio >= 0.95:
        raise ValueError("train_ratio + val_ratio must be < 0.95")

    league_codes = list_league_codes()
    if not league_codes:
        raise RuntimeError("No leagues found in the database.")

    trainer = train_league_main if args.trainer == "main" else train_league_baseline
    model_name = resolve_model_name(args)
    expected_params = build_expected_params(args)
    completed_leagues: set[str] = set()
    if args.skip_existing:
        completed_leagues = list_completed_leagues(
            model_name, args.version, args.season, expected_params
        )

    errors: list[tuple[str, str]] = []
    for league_code in league_codes:
        if args.skip_existing and league_code in completed_leagues:
            print(
                f"=== Skipping league {league_code} (already trained: {model_name}, params match) ===",
                flush=True,
            )
            continue
        print(f"=== Training league {league_code} ===", flush=True)
        stop_event = None
        thread = None
        if args.heartbeat_seconds and args.heartbeat_seconds > 0:
            stop_event, thread = _start_heartbeat(league_code, args.heartbeat_seconds)
        try:
            trainer(args, league_code)
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            if not args.continue_on_error:
                if stop_event is not None:
                    stop_event.set()
                if thread is not None:
                    thread.join(timeout=1)
                raise
            print(f"Skipped league {league_code}: {message}", flush=True)
            errors.append((league_code, message))
        finally:
            if stop_event is not None:
                stop_event.set()
            if thread is not None:
                thread.join(timeout=1)

    if errors:
        print("Skipped leagues:")
        for league_code, message in errors:
            print(f"  {league_code}: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer",
        default="main",
        choices=["main", "baseline"],
        help="Which trainer to run for each league.",
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["xgboost", "catboost"],
        help="Model to train when --trainer=main.",
    )
    parser.add_argument("--version", default="v1")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--season", default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable probability calibration on the validation split.",
    )
    parser.add_argument(
        "--calibration-method",
        default="sigmoid",
        choices=["sigmoid", "isotonic"],
    )
    parser.add_argument(
        "--class-weight",
        default="balanced",
        choices=["none", "balanced"],
        help="Class weighting strategy for training (balanced uses inverse frequency).",
    )
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=0,
        help="Number of time-series CV splits to evaluate (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-leakage", action="store_true")
    parser.add_argument(
        "--verbose-eval",
        type=int,
        default=0,
        help="Print evaluation metrics every N iterations (0 disables).",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=0,
        help="Stop boosting if validation metric does not improve after N rounds (0 disables).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Override the number of boosting rounds for XGBoost (default uses config).",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="Print a heartbeat message every N seconds while training a league. Use 0 to disable.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip leagues that already have a completed run for the same trainer/model/version/season.",
    )
    parser.add_argument(
        "--best-metric",
        default="val_log_loss",
        help="Metric used to track the best model (default: val_log_loss).",
    )
    parser.add_argument(
        "--remember-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track the best model in models/best_models.json.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip leagues that fail and continue.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
