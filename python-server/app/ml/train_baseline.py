from __future__ import annotations

import argparse
import uuid
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import MODEL_DIR
from app.db.session import SessionLocal
from app.ml.artifacts import git_sha, save_model
from app.ml.best_registry import extract_metric, make_registry_key, update_best_registry
from app.ml.dataset import (
    FEATURE_COLUMNS,
    load_feature_dataset,
    temporal_split,
    temporal_split_date_ranges,
)
from app.ml.leakage_checks import run_leakage_checks
from app.ml.labels import LABELS, LABEL_IDS, encode_labels, label_counts, predict_class_ids
from app.ml.metrics import classification_metrics, class_priors
from app.ml.tracking import build_artifact_info, record_run
from app.reporting.label_audit import write_label_audit
from app.reporting.training_divergence import (
    compute_feature_scaling_stats,
    step_metrics_from_summary,
    write_training_divergence_report,
)


DEFAULT_PARAMS: dict[str, Any] = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "C": 1.0,
}

TUNING_GRID: dict[str, list[Any]] = {
    "C": [0.1, 0.5, 1.0, 2.0, 5.0],
    "solver": ["lbfgs", "saga"],
}


def build_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    model_params = DEFAULT_PARAMS.copy()
    if params:
        model_params.update(params)
    model = LogisticRegression(**model_params)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def compute_sample_weights(y: list[int], label_ids: list[int]) -> list[float]:
    counts = {label_id: 0 for label_id in label_ids}
    for value in y:
        counts[int(value)] = counts.get(int(value), 0) + 1
    total = sum(counts.values()) or 1
    weights = {label_id: total / (len(label_ids) * max(1, count)) for label_id, count in counts.items()}
    return [weights[int(value)] for value in y]


def aggregate_cv_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["accuracy", "f1_macro", "log_loss", "brier_score"]
    summary: dict[str, Any] = {}
    for key in keys:
        values = []
        for metrics in metrics_list:
            metric_key = metrics.get(key)
            if isinstance(metric_key, float):
                values.append(metric_key)
        if not values:
            continue
        avg = float(sum(values) / len(values))
        variance = sum((val - avg) ** 2 for val in values) / len(values)
        summary[f"cv_{key}_mean"] = avg
        summary[f"cv_{key}_std"] = float(variance**0.5)
    return summary


def compute_cv_metrics(
    X,
    y,
    *,
    params: dict[str, Any] | None,
    label_ids: list[int],
    labels: list[str],
    n_splits: int,
    class_weight_mode: str,
) -> dict[str, Any]:
    if n_splits <= 1:
        return {}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, Any]] = []
    for split_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]
        base_params = params.copy() if params else {}
        base_params["class_weight"] = None if class_weight_mode == "none" else "balanced"
        pipeline = build_pipeline(base_params or None)
        if class_weight_mode == "balanced":
            weights = compute_sample_weights(list(y_train), label_ids)
            pipeline.fit(X_train, y_train, model__sample_weight=weights)
        else:
            pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)
        y_pred = predict_class_ids(y_prob)
        metrics = classification_metrics(
            f"cv_{split_idx}", y_test, y_pred, y_prob, label_ids, labels
        )
        fold_metrics.append(
            {
                "accuracy": metrics.get(f"cv_{split_idx}_accuracy"),
                "f1_macro": metrics.get(f"cv_{split_idx}_f1_macro"),
                "log_loss": metrics.get(f"cv_{split_idx}_log_loss"),
                "brier_score": metrics.get(f"cv_{split_idx}_brier_score"),
            }
        )
    summary = aggregate_cv_metrics(fold_metrics)
    summary["cv_folds"] = fold_metrics
    return summary


def sample_param_candidates(n_iter: int, seed: int) -> list[dict[str, Any]]:
    sampler = ParameterSampler(TUNING_GRID, n_iter=n_iter, random_state=seed)
    return list(sampler)


def tune_params(
    X_train,
    y_train,
    X_val,
    y_val,
    labels: list[str],
    *,
    n_iter: int,
    seed: int,
    walk_forward: bool,
    class_weight_mode: str,
    n_splits: int = 3,
) -> tuple[dict[str, Any], float]:
    label_ids = LABEL_IDS
    candidates = sample_param_candidates(n_iter, seed)
    best_params: dict[str, Any] = {}
    best_score = float("inf")

    if walk_forward and len(X_train) <= n_splits:
        walk_forward = False
    if walk_forward:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for params in candidates:
            fold_scores: list[float] = []
            for train_idx, val_idx in tscv.split(X_train):
                tuned_params = params.copy()
                tuned_params["class_weight"] = (
                    None if class_weight_mode == "none" else "balanced"
                )
                pipeline = build_pipeline(tuned_params)
                pipeline.fit(X_train.iloc[train_idx], y_train[train_idx])
                y_prob = pipeline.predict_proba(X_train.iloc[val_idx])
                y_pred = predict_class_ids(y_prob)
                metrics = classification_metrics(
                    "val",
                    y_train[val_idx],
                    y_pred,
                    y_prob,
                    label_ids,
                    labels,
                )
                score = metrics.get("val_log_loss")
                if score is None or score != score:
                    score = float("inf")
                fold_scores.append(float(score))
            avg_score = sum(fold_scores) / len(fold_scores) if fold_scores else float("inf")
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
    else:
        for params in candidates:
            tuned_params = params.copy()
            tuned_params["class_weight"] = (
                None if class_weight_mode == "none" else "balanced"
            )
            pipeline = build_pipeline(tuned_params)
            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_val)
            y_pred = predict_class_ids(y_prob)
            metrics = classification_metrics(
                "val", y_val, y_pred, y_prob, label_ids, labels
            )
            score = metrics.get("val_log_loss")
            if score is None or score != score:
                score = float("inf")
            if float(score) < best_score:
                best_score = float(score)
                best_params = params

    return best_params, best_score


def artifact_type_for_path(path) -> str:
    name = path.name
    if name.endswith("_schema.json"):
        return "feature_schema"
    if name.endswith("_label_encoder.joblib"):
        return "label_encoder"
    if name.endswith(".json"):
        return "model_meta"
    if name.endswith(".pkl"):
        return "model"
    return "artifact"


def train(
    version: str,
    train_ratio: float,
    val_ratio: float,
    *,
    league_code: str | None,
    season_label: str | None,
    tune: bool,
    n_iter: int,
    calibrate: bool,
    calibration_method: str,
    walk_forward: bool,
    seed: int,
    allow_leakage: bool,
    class_weight_mode: str,
    cv_splits: int,
    best_metric: str | None = "val_log_loss",
    remember_best: bool = True,
) -> None:
    df = load_feature_dataset(
        version,
        league_code=league_code,
        season_label=season_label,
        include_match_id=True,
    )
    if df.empty:
        raise RuntimeError("No feature rows available. Run the ETL first.")

    split = temporal_split(df, train_ratio=train_ratio, val_ratio=val_ratio)
    split_ranges = temporal_split_date_ranges(df, train_ratio, val_ratio)

    y_train = np.asarray(encode_labels(split.y_train))
    y_val = np.asarray(encode_labels(split.y_val))
    y_test = np.asarray(encode_labels(split.y_test))

    if len(set(y_train)) < len(LABELS):
        raise RuntimeError(
            "Train split is missing one or more result classes. "
            "Increase the dataset size or adjust split ratios."
        )

    leakage_summary = None
    with SessionLocal() as session:
        leakage_summary = run_leakage_checks(
            session,
            version=version,
            league_code=league_code,
            season_label=season_label,
        )
    if (
        leakage_summary.mismatches
        or leakage_summary.date_order_violations
        or leakage_summary.negative_rest_days
    ):
        if not allow_leakage:
            raise RuntimeError(
                "Leakage checks failed. "
                "Rebuild features or re-run ETL with correct ordering."
            )

    labels = LABELS
    label_ids = LABEL_IDS

    class_priors_train = class_priors(list(y_train), label_ids)
    majority_class = int(max(range(len(class_priors_train)), key=class_priors_train.__getitem__))

    best_params: dict[str, Any] = {}
    best_score = float("nan")
    if tune:
        best_params, best_score = tune_params(
            split.X_train,
            y_train,
            split.X_val,
            y_val,
            labels,
            n_iter=n_iter,
            seed=seed,
            walk_forward=walk_forward,
            class_weight_mode=class_weight_mode,
        )

    cv_summary: dict[str, Any] = {}
    if cv_splits and cv_splits > 1:
        cv_summary = compute_cv_metrics(
            split.X_train,
            y_train,
            params=best_params if tune else None,
            label_ids=label_ids,
            labels=labels,
            n_splits=cv_splits,
            class_weight_mode=class_weight_mode,
        )

    model_params = (best_params.copy() if tune else {})
    model_params["class_weight"] = None if class_weight_mode == "none" else "balanced"
    pipeline = build_pipeline(model_params or None)
    if class_weight_mode == "balanced":
        sample_weight = compute_sample_weights(list(y_train), label_ids)
        pipeline.fit(split.X_train, y_train, model__sample_weight=sample_weight)
    else:
        pipeline.fit(split.X_train, y_train)

    if calibrate:
        sample_weight_val = None
        if class_weight_mode == "balanced":
            sample_weight_val = compute_sample_weights(list(y_val), label_ids)
        try:
            calibrator = CalibratedClassifierCV(
                pipeline, method=calibration_method, cv="prefit"
            )
            calibrator.fit(split.X_val, y_val, sample_weight=sample_weight_val)
            model = calibrator
        except Exception:
            print(
                "Warning: prefit calibration is not supported in this scikit-learn "
                "version; calibrating with 3-fold CV on the training set."
            )
            calibrator = CalibratedClassifierCV(
                pipeline, method=calibration_method, cv=3
            )
            sample_weight_train = None
            if class_weight_mode == "balanced":
                sample_weight_train = compute_sample_weights(list(y_train), label_ids)
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
        classification_metrics(
            "val", y_val, val_pred, val_prob, label_ids, labels
        )
    )
    metrics.update(
        classification_metrics(
            "test", y_test, test_pred, test_prob, label_ids, labels
        )
    )
    if class_priors_train:
        val_prior_probs = [class_priors_train for _ in range(len(y_val))]
        test_prior_probs = [class_priors_train for _ in range(len(y_test))]
        val_prior_pred = [majority_class for _ in range(len(y_val))]
        test_prior_pred = [majority_class for _ in range(len(y_test))]
        metrics.update(
            classification_metrics(
                "val_baseline_prior",
                y_val,
                val_prior_pred,
                val_prior_probs,
                label_ids,
                labels,
            )
        )
        metrics.update(
            classification_metrics(
                "test_baseline_prior",
                y_test,
                test_prior_pred,
                test_prior_probs,
                label_ids,
                labels,
            )
        )
        metrics.update(
            classification_metrics(
                "val_baseline_majority",
                y_val,
                val_prior_pred,
                None,
                label_ids,
                labels,
            )
        )
        metrics.update(
            classification_metrics(
                "test_baseline_majority",
                y_test,
                test_prior_pred,
                None,
                label_ids,
                labels,
            )
        )

    run_id = uuid.uuid4().hex
    feature_stats = compute_feature_scaling_stats(split.X_train)
    step_metrics = step_metrics_from_summary(
        metrics.get("val_log_loss"),
        None,
    )
    divergence_paths = write_training_divergence_report(
        run_id=run_id,
        model_name="baseline_classifier",
        league_code=league_code,
        season_label=season_label,
        step_metrics=step_metrics,
        feature_stats=feature_stats,
        val_true=y_val,
        val_pred=val_pred,
        learning_rate=None,
    )
    label_audit_paths = write_label_audit(
        run_id=run_id,
        label_counts_train=label_counts(y_train),
        label_counts_val=label_counts(y_val),
        confusion_matrix=metrics["val_confusion_matrix"],
        metrics=metrics,
        league_code=league_code,
        season_label=season_label,
    )
    feature_schema = {
        "version": version,
        "columns": FEATURE_COLUMNS,
    }
    model_path, meta, artifact_paths = save_model(
        model,
        model_name="baseline_classifier",
        model_dir=MODEL_DIR,
        metrics=metrics,
        extra_meta={
            "feature_version": version,
            "classes": labels,
            "label_mapping": {label: idx for idx, label in enumerate(labels)},
            "features": FEATURE_COLUMNS,
            "calibration_method": calibration_method if calibrate else None,
        },
        feature_schema=feature_schema,
        run_id=run_id,
    )

    artifacts = [
        build_artifact_info(path, artifact_type_for_path(path), MODEL_DIR)
        for path in artifact_paths
    ]
    params = {
        "tuning_enabled": tune,
        "tuning_best_score": best_score,
        "tuning_candidates": n_iter if tune else 0,
        "walk_forward": walk_forward if tune else False,
        "hyperparameters": best_params if tune else DEFAULT_PARAMS,
        "calibration_enabled": calibrate,
        "calibration_method": calibration_method if calibrate else None,
        "class_weight": class_weight_mode,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "cv_splits": cv_splits,
    }
    train_min_date, train_max_date = split_ranges["train"]
    val_min_date, val_max_date = split_ranges["val"]
    test_min_date, test_max_date = split_ranges["test"]
    data_summary = {
        "train_size": int(len(split.X_train)),
        "val_size": int(len(split.X_val)),
        "test_size": int(len(split.X_test)),
        "min_date": str(df["match_date"].min()),
        "max_date": str(df["match_date"].max()),
        "train_min_date": train_min_date,
        "train_max_date": train_max_date,
        "val_min_date": val_min_date,
        "val_max_date": val_max_date,
        "test_min_date": test_min_date,
        "test_max_date": test_max_date,
        "league_code": league_code,
        "season_label": season_label,
        "label_mapping": {label: idx for idx, label in enumerate(labels)},
        "label_counts": {
            "train": label_counts(y_train),
            "val": label_counts(y_val),
            "test": label_counts(y_test),
        },
        "label_audit": label_audit_paths,
        "divergence_report": divergence_paths,
        "leakage_checks": leakage_summary.__dict__ if leakage_summary else None,
        "cv_summary": cv_summary if cv_summary else None,
    }

    record_run(
        run_id=run_id,
        model_name="baseline_classifier",
        model_version=meta["model_version"],
        feature_version=version,
        commit_hash=git_sha(),
        league_code=league_code,
        season_label=season_label,
        metrics=metrics,
        params=params,
        data_summary=data_summary,
        artifacts=artifacts,
    )

    if remember_best and best_metric:
        metric_value = extract_metric(metrics, best_metric)
        if metric_value is not None:
            key = make_registry_key(
                "baseline_classifier", version, league_code, season_label
            )
            try:
                artifact_path = model_path.relative_to(MODEL_DIR).as_posix()
            except ValueError:
                artifact_path = model_path.name
            try:
                meta_path = model_path.with_suffix(".json").relative_to(MODEL_DIR).as_posix()
            except ValueError:
                meta_path = model_path.with_suffix(".json").name
            update_best_registry(
                MODEL_DIR,
                key,
                {
                    "run_id": run_id,
                    "model_name": "baseline_classifier",
                    "model_version": meta["model_version"],
                    "feature_version": version,
                    "league_code": league_code,
                    "season_label": season_label,
                    "artifact": artifact_path,
                    "meta": meta_path,
                },
                metric_name=best_metric,
                metric_value=metric_value,
            )

    print(f"Saved model to {model_path}")
    print(f"Run id: {run_id}")
    print(f"Classes: {labels}")
    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print(f"Meta: {meta}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--league", default=None)
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
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 0.95:
        raise ValueError("train_ratio + val_ratio must be < 0.95")

    train(
        args.version,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        league_code=args.league,
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


if __name__ == "__main__":
    main()
