from __future__ import annotations

import argparse
import inspect
import uuid
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline

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
from app.reporting.training_divergence import (
    compute_feature_scaling_stats,
    step_metrics_from_catboost,
    step_metrics_from_summary,
    step_metrics_from_xgboost,
    write_training_divergence_report,
)
from app.reporting.label_audit import write_label_audit


DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "n_estimators": 5,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "multi:softprob",
        "eval_metric": ["mlogloss", "merror"],
        "num_class": 3,
        "random_state": 42,
    },
    "catboost": {
        "iterations": 400,
        "depth": 6,
        "learning_rate": 0.05,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "custom_metric": ["Accuracy"],
        "verbose": False,
        "random_seed": 42,
    },
}

TUNING_GRID: dict[str, dict[str, list[Any]]] = {
    "xgboost": {
        "n_estimators": [200, 400, 600],
        "max_depth": [4, 6, 8],
        "min_child_weight": [1, 3, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "gamma": [0.0, 0.1, 0.3],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [1.0, 2.0, 5.0],
    },
    "catboost": {
        "iterations": [200, 400, 600],
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0],
    },
}


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
    model_name: str,
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
        pipeline = build_pipeline(model_name, params)
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


def build_pipeline(model_name: str, params: dict[str, Any] | None = None) -> Pipeline:
    model = build_model(model_name, params)

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def build_model(model_name: str, params: dict[str, Any] | None = None):
    model_params = DEFAULT_PARAMS[model_name].copy()
    if params:
        model_params.update(params)

    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("CatBoost is not installed.") from exc
        return CatBoostClassifier(**model_params)

    from xgboost import XGBClassifier

    return XGBClassifier(**model_params)


def sample_param_candidates(
    model_name: str,
    n_iter: int,
    seed: int,
) -> list[dict[str, Any]]:
    grid = TUNING_GRID[model_name]
    sampler = ParameterSampler(grid, n_iter=n_iter, random_state=seed)
    return list(sampler)


def tune_params(
    model_name: str,
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
    param_overrides: dict[str, Any] | None = None,
    n_splits: int = 3,
) -> tuple[dict[str, Any], float]:
    label_ids = LABEL_IDS
    candidates = sample_param_candidates(model_name, n_iter, seed)
    best_params: dict[str, Any] = {}
    best_score = float("inf")

    param_overrides = param_overrides or {}
    if walk_forward and len(X_train) <= n_splits:
        walk_forward = False
    if walk_forward:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for params in candidates:
            params = params.copy()
            params.update(param_overrides)
            fold_scores: list[float] = []
            for train_idx, val_idx in tscv.split(X_train):
                pipeline = build_pipeline(model_name, params)
                if class_weight_mode == "balanced":
                    weights = compute_sample_weights(
                        list(y_train[train_idx]), label_ids
                    )
                    pipeline.fit(
                        X_train.iloc[train_idx],
                        y_train[train_idx],
                        model__sample_weight=weights,
                    )
                else:
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
            params = params.copy()
            params.update(param_overrides)
            pipeline = build_pipeline(model_name, params)
            if class_weight_mode == "balanced":
                weights = compute_sample_weights(list(y_train), label_ids)
                pipeline.fit(X_train, y_train, model__sample_weight=weights)
            else:
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
    model_name: str,
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
    verbose_eval: int,
    early_stopping_rounds: int,
    n_estimators: int | None,
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
    param_overrides: dict[str, Any] = {}
    if model_name == "xgboost" and n_estimators is not None:
        param_overrides["n_estimators"] = int(n_estimators)

    class_priors_train = class_priors(list(y_train), label_ids)
    majority_class = int(max(range(len(class_priors_train)), key=class_priors_train.__getitem__))

    best_params: dict[str, Any] = {}
    best_score = float("nan")
    if tune:
        best_params, best_score = tune_params(
            model_name,
            split.X_train,
            y_train,
            split.X_val,
            y_val,
            labels,
            n_iter=n_iter,
            seed=seed,
            walk_forward=walk_forward,
            class_weight_mode=class_weight_mode,
            param_overrides=param_overrides,
        )

    cv_summary: dict[str, Any] = {}
    if cv_splits and cv_splits > 1:
        cv_summary = compute_cv_metrics(
            split.X_train,
            y_train,
            model_name=model_name,
            params=best_params if tune else (param_overrides or None),
            label_ids=label_ids,
            labels=labels,
            n_splits=cv_splits,
            class_weight_mode=class_weight_mode,
        )

    pipeline: Pipeline
    evals_result = None
    verbose_eval = int(verbose_eval)
    early_stopping_rounds = int(early_stopping_rounds)
    use_eval_set = (
        model_name == "xgboost"
        or verbose_eval > 0
        or early_stopping_rounds > 0
    )
    if use_eval_set:
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(split.X_train)
        X_val = imputer.transform(split.X_val)
        fit_params = best_params if tune else None
        if param_overrides:
            fit_params = (fit_params or {}).copy()
            fit_params.update(param_overrides)
        model = build_model(model_name, fit_params)
        sample_weight = None
        sample_weight_val = None
        if class_weight_mode == "balanced":
            sample_weight = compute_sample_weights(list(y_train), label_ids)
            sample_weight_val = compute_sample_weights(list(y_val), label_ids)
        if model_name == "catboost":
            fit_kwargs: dict[str, Any] = {
                "eval_set": (X_val, y_val),
                "verbose": verbose_eval,
                "sample_weight": sample_weight,
            }
            if early_stopping_rounds > 0:
                fit_kwargs["use_best_model"] = True
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            model.fit(X_train, y_train, **fit_kwargs)
            evals_result = model.get_evals_result()
        else:
            fit_kwargs: dict[str, Any] = {
                "eval_set": [(X_val, y_val)],
                "verbose": verbose_eval,
                "sample_weight": sample_weight,
                "sample_weight_eval_set": [sample_weight_val] if sample_weight_val else None,
            }
            if model_name == "xgboost":
                fit_kwargs["eval_set"] = [(X_train, y_train), (X_val, y_val)]
                if sample_weight is not None or sample_weight_val is not None:
                    fit_kwargs["sample_weight_eval_set"] = [
                        sample_weight,
                        sample_weight_val,
                    ]
            if early_stopping_rounds > 0:
                fit_params = inspect.signature(model.fit).parameters
                if "early_stopping_rounds" in fit_params:
                    fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
                elif "callbacks" in fit_params:
                    try:
                        from xgboost import callback as xgb_callback

                        fit_kwargs["callbacks"] = [
                            xgb_callback.EarlyStopping(
                                rounds=early_stopping_rounds, save_best=True
                            )
                        ]
                    except Exception:
                        print(
                            "Warning: XGBoost does not support early stopping in this environment; "
                            "continuing without it."
                        )
                else:
                    print(
                        "Warning: XGBoost does not support early stopping in this environment; "
                        "continuing without it."
                    )
            model.fit(X_train, y_train, **fit_kwargs)
            evals_result = model.evals_result()
        pipeline = Pipeline([("imputer", imputer), ("model", model)])
    else:
        fit_params = best_params if tune else None
        if param_overrides:
            fit_params = (fit_params or {}).copy()
            fit_params.update(param_overrides)
        pipeline = build_pipeline(model_name, fit_params)
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
        classification_metrics("val", y_val, val_pred, val_prob, label_ids, labels)
    )
    metrics.update(
        classification_metrics("test", y_test, test_pred, test_prob, label_ids, labels)
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
    model_key = "main_catboost" if model_name == "catboost" else "main_xgboost"
    feature_stats = compute_feature_scaling_stats(split.X_train)
    learning_rate = None
    if model_name in DEFAULT_PARAMS:
        learning_rate = DEFAULT_PARAMS[model_name].get("learning_rate")
    if tune and best_params.get("learning_rate") is not None:
        learning_rate = best_params.get("learning_rate")
    if param_overrides and param_overrides.get("learning_rate") is not None:
        learning_rate = param_overrides.get("learning_rate")
    step_metrics = []
    if model_name == "xgboost":
        step_metrics = step_metrics_from_xgboost(evals_result, learning_rate)
    elif model_name == "catboost":
        step_metrics = step_metrics_from_catboost(evals_result, learning_rate)
    if not step_metrics:
        step_metrics = step_metrics_from_summary(
            metrics.get("val_log_loss"),
            learning_rate,
        )
    divergence_paths = write_training_divergence_report(
        run_id=run_id,
        model_name=model_key,
        league_code=league_code,
        season_label=season_label,
        step_metrics=step_metrics,
        feature_stats=feature_stats,
        val_true=y_val,
        val_pred=val_pred,
        learning_rate=learning_rate,
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
        model_name=model_key,
        model_dir=MODEL_DIR,
        metrics=metrics,
        extra_meta={
            "feature_version": version,
            "classes": labels,
            "label_mapping": {label: idx for idx, label in enumerate(labels)},
            "features": FEATURE_COLUMNS,
            "model_type": model_name,
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
        "model_type": model_name,
        "tuning_enabled": tune,
        "tuning_best_score": best_score,
        "tuning_candidates": n_iter if tune else 0,
        "walk_forward": walk_forward if tune else False,
        "hyperparameters": best_params if tune else DEFAULT_PARAMS[model_name],
        "calibration_enabled": calibrate,
        "calibration_method": calibration_method if calibrate else None,
        "early_stopping_rounds": early_stopping_rounds,
        "n_estimators": n_estimators if model_name == "xgboost" else None,
        "class_weight": class_weight_mode,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "cv_splits": cv_splits,
    }
    if model_name == "xgboost" and n_estimators is not None:
        params["hyperparameters"] = (params["hyperparameters"] or {}).copy()
        params["hyperparameters"]["n_estimators"] = int(n_estimators)
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
        model_name=model_key,
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
            key = make_registry_key(model_key, version, league_code, season_label)
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
                    "model_name": model_key,
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
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "catboost"])
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
        model_name=args.model,
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
        verbose_eval=args.verbose_eval,
        early_stopping_rounds=args.early_stopping_rounds,
        n_estimators=args.n_estimators,
        class_weight_mode=args.class_weight,
        cv_splits=args.cv_splits,
        best_metric=args.best_metric,
        remember_best=args.remember_best,
    )


if __name__ == "__main__":
    main()

