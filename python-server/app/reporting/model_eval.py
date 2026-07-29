from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from app.config import MODEL_DIR, REPORT_DIR
from app.db.models import Feature, League, Match, Season
from app.db.session import SessionLocal
from app.ml.dataset import (
    CANONICAL_KEY,
    DATE_COLUMN,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    assert_no_reorder_or_mismatch,
)
from app.ml.labels import LABELS, LABEL_IDS, encode_labels, predict_class_ids
from app.ml.metrics import classification_metrics
from app.ml.registry import (
    load_best_model_by_name,
    load_latest_model_by_name,
    load_model_by_version,
)


@dataclass
class EvalSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    test_meta: pd.DataFrame


def load_eval_dataset(
    version: str,
    *,
    league_code: str | None = None,
    season_label: str | None = None,
) -> pd.DataFrame:
    columns = [getattr(Feature, col) for col in FEATURE_COLUMNS]
    with SessionLocal() as session:
        query = (
            session.query(
                Match.id,
                Match.match_date,
                Match.result,
                League.code,
                Season.label,
                *columns,
            )
            .join(Feature, Feature.match_id == Match.id)
            .join(Season, Match.season_id == Season.id)
            .join(League, Season.league_id == League.id)
            .filter(Feature.version == version)
            .filter(Match.match_date.isnot(None))
            .filter(Match.result.isnot(None))
            .filter(Match.status == "played")
        )
        if league_code:
            query = query.filter(League.code == league_code)
        if season_label:
            query = query.filter(Season.label == season_label)
        rows = query.order_by(Match.match_date.asc(), Match.id.asc()).all()

    records: list[dict[str, Any]] = []
    for row in rows:
        offset = 0
        record: dict[str, Any] = {
            CANONICAL_KEY: row[offset],
            DATE_COLUMN: row[offset + 1],
            LABEL_COLUMN: row[offset + 2],
            "league_code": row[offset + 3],
            "season_label": row[offset + 4],
        }
        offset += 5
        for idx, col in enumerate(FEATURE_COLUMNS):
            record[col] = row[offset + idx]
        records.append(record)
    df = pd.DataFrame(records)
    if not df.empty:
        assert_no_reorder_or_mismatch(step="load_eval_dataset", df=df)
    return df


def temporal_split_with_meta(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> EvalSplit:
    if CANONICAL_KEY not in df.columns:
        raise RuntimeError(
            f"temporal_split: missing {CANONICAL_KEY} column (canonical key required)."
        )
    expected_keys = df[CANONICAL_KEY].reset_index(drop=True)
    df = df.sort_values([DATE_COLUMN, CANONICAL_KEY]).reset_index(drop=True)
    assert_no_reorder_or_mismatch(
        step="temporal_split/sort",
        df=df,
        expected_keys=expected_keys,
        allow_reorder=True,
    )
    n = len(df)
    if n < 10:
        raise RuntimeError("Not enough samples to split dataset.")

    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    val_end = min(val_end, n - 1)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    X_train_with_key = train[[CANONICAL_KEY, *FEATURE_COLUMNS]]
    y_train_with_key = train[[CANONICAL_KEY, LABEL_COLUMN]]
    assert_no_reorder_or_mismatch(
        step="temporal_split/train", X=X_train_with_key, y=y_train_with_key
    )
    X_train = X_train_with_key[FEATURE_COLUMNS]
    y_train = y_train_with_key[LABEL_COLUMN]

    X_val_with_key = val[[CANONICAL_KEY, *FEATURE_COLUMNS]]
    y_val_with_key = val[[CANONICAL_KEY, LABEL_COLUMN]]
    assert_no_reorder_or_mismatch(
        step="temporal_split/val", X=X_val_with_key, y=y_val_with_key
    )
    X_val = X_val_with_key[FEATURE_COLUMNS]
    y_val = y_val_with_key[LABEL_COLUMN]

    X_test_with_key = test[[CANONICAL_KEY, *FEATURE_COLUMNS]]
    y_test_with_key = test[[CANONICAL_KEY, LABEL_COLUMN]]
    assert_no_reorder_or_mismatch(
        step="temporal_split/test", X=X_test_with_key, y=y_test_with_key
    )
    X_test = X_test_with_key[FEATURE_COLUMNS]
    y_test = y_test_with_key[LABEL_COLUMN]
    test_meta = test[[CANONICAL_KEY, "league_code", "season_label", LABEL_COLUMN]].copy()

    return EvalSplit(X_train, y_train, X_val, y_val, X_test, y_test, test_meta)


def load_model(
    model_name: str,
    *,
    lookup: str,
    feature_version: str,
    league_code: str | None,
    season_label: str | None,
    model_version: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    model = None
    meta = None
    if model_version:
        model, meta = load_model_by_version(MODEL_DIR, model_name, model_version)
        if model is None or meta is None:
            raise RuntimeError(
                f"Model {model_name} with version {model_version} not found in {MODEL_DIR}."
            )
        return model, meta
    if lookup == "best":
        model, meta = load_best_model_by_name(
            MODEL_DIR,
            model_name,
            feature_version=feature_version,
            league_code=league_code,
            season_label=season_label,
        )
    if model is None:
        model, meta = load_latest_model_by_name(MODEL_DIR, model_name)
    if model is None or meta is None:
        raise RuntimeError(f"Model {model_name} not found in {MODEL_DIR}.")
    return model, meta


def align_probabilities(model: Any, probs: np.ndarray) -> np.ndarray:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model")
        classes = getattr(estimator, "classes_", None)
    if classes is None:
        return probs
    class_order = [int(value) for value in classes]
    if class_order == LABEL_IDS:
        return probs
    index = [class_order.index(label_id) for label_id in LABEL_IDS]
    return probs[:, index]


def compute_overconfidence(
    y_true: np.ndarray, y_pred: list[int], y_prob: np.ndarray, threshold: float
) -> dict[str, Any]:
    max_prob = y_prob.max(axis=1)
    over_mask = max_prob >= threshold
    over_count = int(over_mask.sum())
    if over_count:
        incorrect = int((np.asarray(y_pred) != y_true)[over_mask].sum())
        incorrect_rate = float(incorrect / over_count)
    else:
        incorrect_rate = 0.0
    return {
        "threshold": float(threshold),
        "count": over_count,
        "rate": float(over_count / len(y_true)) if len(y_true) else 0.0,
        "incorrect_rate": incorrect_rate,
        "max_prob_mean": float(max_prob.mean()) if len(max_prob) else float("nan"),
        "max_prob_p95": float(np.percentile(max_prob, 95)) if len(max_prob) else float("nan"),
    }


def compute_entropy(y_prob: np.ndarray) -> np.ndarray:
    eps = 1e-12
    entropy = -np.sum(y_prob * np.log(y_prob + eps), axis=1)
    return entropy / math.log(y_prob.shape[1])


def plot_confusion_matrix(
    matrix: list[list[int]],
    output: Path,
    *,
    title: str | None = None,
) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def plot_histogram(
    values: dict[str, np.ndarray],
    *,
    title: str,
    xlabel: str,
    output: Path,
    bins: int,
) -> None:
    plt.figure(figsize=(7, 4))
    for label, series in values.items():
        plt.hist(series, bins=bins, alpha=0.6, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def plot_calibration(
    calibration: dict[str, dict[str, list[dict[str, float]]]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, len(LABELS), figsize=(12, 4), sharey=True)
    if len(LABELS) == 1:
        axes = [axes]
    for idx, label in enumerate(LABELS):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        for model_name, calib in calibration.items():
            bins = calib.get("per_class", {}).get(label, [])
            if not bins:
                continue
            xs = [float(item["avg_pred"]) for item in bins]
            ys = [float(item["avg_true"]) for item in bins]
            ax.plot(xs, ys, marker="o", label=model_name)
        ax.set_title(f"Calibration {label}")
        ax.set_xlabel("Predicted")
        if idx == 0:
            ax.set_ylabel("Observed")
        ax.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def inject_missingness(
    X: pd.DataFrame, *, rate: float, mode: str, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = X.to_numpy(copy=True)
    mask = rng.random(values.shape) < rate
    if mode == "zero":
        values[mask] = 0.0
    else:
        values[mask] = np.nan
    return pd.DataFrame(values, columns=X.columns, index=X.index)


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, list[int]]:
    y_prob = model.predict_proba(X_test)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = align_probabilities(model, y_prob)
    y_pred = predict_class_ids(y_prob)
    metrics = classification_metrics(
        "test", y_test, y_pred, y_prob, LABEL_IDS, LABELS
    )
    return metrics, y_prob, y_pred


def summarize_per_class(per_class: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for entry in per_class:
        if entry.get("label") == label:
            return {
                "label": label,
                "precision": entry.get("precision"),
                "recall": entry.get("recall"),
                "f1": entry.get("f1"),
                "support": entry.get("support"),
            }
    return {"label": label, "precision": None, "recall": None, "f1": None, "support": None}


def subset_metrics(
    *,
    model_name: str,
    y_true: np.ndarray,
    y_pred: list[int],
    y_prob: np.ndarray,
    meta: pd.DataFrame,
    subset_type: str,
    subset_values: pd.Series,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, idx in subset_values.groupby(subset_values).groups.items():
        idx_list = list(idx)
        if not idx_list:
            continue
        y_true_sub = y_true[idx_list]
        y_pred_sub = [y_pred[i] for i in idx_list]
        y_prob_sub = y_prob[idx_list]
        metrics = classification_metrics(
            "test", y_true_sub, y_pred_sub, y_prob_sub, LABEL_IDS, LABELS
        )
        rows.append(
            {
                "model": model_name,
                "subset_type": subset_type,
                "subset_value": str(value),
                "samples": int(len(idx_list)),
                "accuracy": metrics.get("test_accuracy"),
                "f1_macro": metrics.get("test_f1_macro"),
                "log_loss": metrics.get("test_log_loss"),
            }
        )
    return rows


def build_report(
    *,
    version: str,
    model_name: str,
    baseline_name: str,
    league_code: str | None,
    season_label: str | None,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    lookup: str,
    model_version: str | None,
    baseline_version: str | None,
    output_dir: Path,
    overconfidence_threshold: float,
    missingness_rate: float,
    missingness_mode: str,
    bins: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    df = load_eval_dataset(version, league_code=league_code, season_label=season_label)
    if df.empty:
        raise RuntimeError("No feature rows available. Run the ETL first.")

    split = temporal_split_with_meta(df, train_ratio=train_ratio, val_ratio=val_ratio)
    y_test = np.asarray(encode_labels(split.y_test))

    main_model, main_meta = load_model(
        model_name,
        lookup=lookup,
        feature_version=version,
        league_code=league_code,
        season_label=season_label,
        model_version=model_version,
    )
    baseline_model, baseline_meta = load_model(
        baseline_name,
        lookup=lookup,
        feature_version=version,
        league_code=league_code,
        season_label=season_label,
        model_version=baseline_version,
    )

    main_metrics, main_prob, main_pred = evaluate_model(
        main_model, split.X_test, y_test
    )
    baseline_metrics, baseline_prob, baseline_pred = evaluate_model(
        baseline_model, split.X_test, y_test
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    main_confusion = main_metrics.get("test_confusion_matrix")
    baseline_confusion = baseline_metrics.get("test_confusion_matrix")
    if main_confusion:
        plot_confusion_matrix(
            main_confusion,
            output_dir / "confusion_matrix.png",
            title=f"{model_name} (main)",
        )
    if baseline_confusion:
        plot_confusion_matrix(
            baseline_confusion,
            output_dir / "confusion_matrix_baseline.png",
            title=f"{baseline_name} (baseline)",
        )

    main_entropy = compute_entropy(main_prob)
    baseline_entropy = compute_entropy(baseline_prob)
    plot_histogram(
        {"main": main_entropy, "baseline": baseline_entropy},
        title="Prediction Entropy",
        xlabel="Normalized entropy",
        output=output_dir / "entropy_hist.png",
        bins=bins,
    )

    main_max_prob = main_prob.max(axis=1)
    baseline_max_prob = baseline_prob.max(axis=1)
    plot_histogram(
        {"main": main_max_prob, "baseline": baseline_max_prob},
        title="Max Probability",
        xlabel="Max probability",
        output=output_dir / "max_prob_hist.png",
        bins=bins,
    )

    calibration_payload = {}
    if main_metrics.get("test_calibration") and baseline_metrics.get("test_calibration"):
        calibration_payload = {
            "main": main_metrics.get("test_calibration", {}),
            "baseline": baseline_metrics.get("test_calibration", {}),
        }
        plot_calibration(calibration_payload, output_dir / "calibration.png")

    draw_label = "D"
    main_draw = summarize_per_class(main_metrics.get("test_f1_per_class", []), draw_label)
    baseline_draw = summarize_per_class(
        baseline_metrics.get("test_f1_per_class", []), draw_label
    )

    main_overconfidence = compute_overconfidence(
        y_test, main_pred, main_prob, overconfidence_threshold
    )
    baseline_overconfidence = compute_overconfidence(
        y_test, baseline_pred, baseline_prob, overconfidence_threshold
    )

    missingness_results: dict[str, Any] = {}
    modes = ["nan", "zero"] if missingness_mode == "both" else [missingness_mode]
    for mode in modes:
        try:
            injected = inject_missingness(
                split.X_test, rate=missingness_rate, mode=mode, seed=seed
            )
            injected_metrics, _, _ = evaluate_model(main_model, injected, y_test)
            missingness_results[f"main_{mode}"] = {
                "accuracy": injected_metrics.get("test_accuracy"),
                "f1_macro": injected_metrics.get("test_f1_macro"),
                "log_loss": injected_metrics.get("test_log_loss"),
            }
        except Exception as exc:
            missingness_results[f"main_{mode}"] = {"error": str(exc)}
        try:
            injected = inject_missingness(
                split.X_test, rate=missingness_rate, mode=mode, seed=seed
            )
            injected_metrics, _, _ = evaluate_model(baseline_model, injected, y_test)
            missingness_results[f"baseline_{mode}"] = {
                "accuracy": injected_metrics.get("test_accuracy"),
                "f1_macro": injected_metrics.get("test_f1_macro"),
                "log_loss": injected_metrics.get("test_log_loss"),
            }
        except Exception as exc:
            missingness_results[f"baseline_{mode}"] = {"error": str(exc)}

    test_meta = split.test_meta.reset_index(drop=True)
    test_meta["home_away"] = test_meta[LABEL_COLUMN].map(
        {"H": "home", "D": "draw", "A": "away"}
    )

    subsets = []
    subsets += subset_metrics(
        model_name="main",
        y_true=y_test,
        y_pred=main_pred,
        y_prob=main_prob,
        meta=test_meta,
        subset_type="league",
        subset_values=test_meta["league_code"].fillna("unknown"),
    )
    subsets += subset_metrics(
        model_name="main",
        y_true=y_test,
        y_pred=main_pred,
        y_prob=main_prob,
        meta=test_meta,
        subset_type="season",
        subset_values=test_meta["season_label"].fillna("unknown"),
    )
    subsets += subset_metrics(
        model_name="main",
        y_true=y_test,
        y_pred=main_pred,
        y_prob=main_prob,
        meta=test_meta,
        subset_type="home_away",
        subset_values=test_meta["home_away"].fillna("unknown"),
    )
    subsets += subset_metrics(
        model_name="baseline",
        y_true=y_test,
        y_pred=baseline_pred,
        y_prob=baseline_prob,
        meta=test_meta,
        subset_type="league",
        subset_values=test_meta["league_code"].fillna("unknown"),
    )
    subsets += subset_metrics(
        model_name="baseline",
        y_true=y_test,
        y_pred=baseline_pred,
        y_prob=baseline_prob,
        meta=test_meta,
        subset_type="season",
        subset_values=test_meta["season_label"].fillna("unknown"),
    )
    subsets += subset_metrics(
        model_name="baseline",
        y_true=y_test,
        y_pred=baseline_pred,
        y_prob=baseline_prob,
        meta=test_meta,
        subset_type="home_away",
        subset_values=test_meta["home_away"].fillna("unknown"),
    )
    subsets_df = pd.DataFrame(subsets)
    subsets_df.to_csv(output_dir / "subsets.csv", index=False)

    summary_rows = []
    for label, metrics, meta in [
        ("main", main_metrics, main_meta),
        ("baseline", baseline_metrics, baseline_meta),
    ]:
        summary_rows.append(
            {
                "model": label,
                "model_name": meta.get("model_name"),
                "model_version": meta.get("model_version"),
                "samples": int(len(y_test)),
                "accuracy": metrics.get("test_accuracy"),
                "f1_macro": metrics.get("test_f1_macro"),
                "log_loss": metrics.get("test_log_loss"),
                "brier_score": metrics.get("test_brier_score"),
                "ece": (metrics.get("test_calibration") or {}).get("ece"),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "metrics.csv", index=False)

    def _delta(metric_name: str) -> float | None:
        main_val = main_metrics.get(metric_name)
        baseline_val = baseline_metrics.get(metric_name)
        if main_val is None or baseline_val is None:
            return None
        return float(main_val) - float(baseline_val)

    payload = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "feature_version": version,
        "model_main": {
            "name": model_name,
            "version": main_meta.get("model_version"),
            "metrics": main_metrics,
            "draw_analysis": main_draw,
            "overconfidence": main_overconfidence,
            "entropy": {
                "mean": float(main_entropy.mean()) if len(main_entropy) else float("nan"),
                "p95": float(np.percentile(main_entropy, 95)) if len(main_entropy) else float("nan"),
            },
        },
        "model_baseline": {
            "name": baseline_name,
            "version": baseline_meta.get("model_version"),
            "metrics": baseline_metrics,
            "draw_analysis": baseline_draw,
            "overconfidence": baseline_overconfidence,
            "entropy": {
                "mean": float(baseline_entropy.mean()) if len(baseline_entropy) else float("nan"),
                "p95": float(np.percentile(baseline_entropy, 95)) if len(baseline_entropy) else float("nan"),
            },
        },
        "comparison": {
            "accuracy_delta": _delta("test_accuracy"),
            "f1_macro_delta": _delta("test_f1_macro"),
            "log_loss_delta": _delta("test_log_loss"),
            "brier_score_delta": _delta("test_brier_score"),
            "ece_delta": (main_metrics.get("test_calibration") or {}).get("ece")
            - (baseline_metrics.get("test_calibration") or {}).get("ece")
            if main_metrics.get("test_calibration") and baseline_metrics.get("test_calibration")
            else None,
        },
        "missingness": {
            "rate": missingness_rate,
            "mode": missingness_mode,
            "results": missingness_results,
        },
        "artifacts": {
            "metrics_csv": str(output_dir / "metrics.csv"),
            "metrics_json": str(output_dir / "metrics.json"),
            "confusion_matrix": str(output_dir / "confusion_matrix.png"),
            "confusion_matrix_baseline": str(output_dir / "confusion_matrix_baseline.png"),
            "calibration": str(output_dir / "calibration.png") if calibration_payload else None,
            "entropy_hist": str(output_dir / "entropy_hist.png"),
            "max_prob_hist": str(output_dir / "max_prob_hist.png"),
            "subsets": str(output_dir / "subsets.csv"),
        },
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    def _fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, (int, float)):
            return f"{value:.{digits}f}"
        return str(value)

    report_lines = [
        "# Model evaluation report",
        "",
        f"Created at: {payload['created_at']}",
        f"Feature version: {version}",
        f"League: {league_code or 'all'}",
        f"Season: {season_label or 'all'}",
        "",
        "## Main model",
        f"- Name: {payload['model_main']['name']}",
        f"- Version: {payload['model_main']['version']}",
        f"- Accuracy: {_fmt(main_metrics.get('test_accuracy'))}",
        f"- F1 macro: {_fmt(main_metrics.get('test_f1_macro'))}",
        f"- Log loss: {_fmt(main_metrics.get('test_log_loss'))}",
        f"- Brier score: {_fmt(main_metrics.get('test_brier_score'))}",
        "",
        "## Baseline model",
        f"- Name: {payload['model_baseline']['name']}",
        f"- Version: {payload['model_baseline']['version']}",
        f"- Accuracy: {_fmt(baseline_metrics.get('test_accuracy'))}",
        f"- F1 macro: {_fmt(baseline_metrics.get('test_f1_macro'))}",
        f"- Log loss: {_fmt(baseline_metrics.get('test_log_loss'))}",
        f"- Brier score: {_fmt(baseline_metrics.get('test_brier_score'))}",
        "",
        "## Comparison (main - baseline)",
        f"- Accuracy delta: {_fmt(payload['comparison']['accuracy_delta'])}",
        f"- F1 macro delta: {_fmt(payload['comparison']['f1_macro_delta'])}",
        f"- Log loss delta: {_fmt(payload['comparison']['log_loss_delta'])}",
        f"- Brier score delta: {_fmt(payload['comparison']['brier_score_delta'])}",
        "",
        "## Draw class diagnostics",
        f"- Main draw F1: {_fmt(main_draw.get('f1'))}",
        f"- Baseline draw F1: {_fmt(baseline_draw.get('f1'))}",
        "",
        "## Overconfidence",
        f"- Threshold: {_fmt(overconfidence_threshold, digits=2)}",
        f"- Main incorrect rate: {_fmt(main_overconfidence.get('incorrect_rate'))}",
        f"- Baseline incorrect rate: {_fmt(baseline_overconfidence.get('incorrect_rate'))}",
        "",
        "Artifacts: metrics.json, metrics.csv, confusion matrices, entropy/max-prob histograms, subsets.csv",
    ]

    (output_dir / "report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(f"Model eval report saved to {output_dir}")
    print(summary_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--model", default="main_xgboost")
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--baseline", default="baseline_classifier")
    parser.add_argument("--baseline-version", default=None)
    parser.add_argument("--league", default=None)
    parser.add_argument("--season", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lookup",
        choices=["best", "latest"],
        default="best",
        help="How to resolve model artifacts (best registry or latest artifact).",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--overconfidence-threshold", type=float, default=0.99)
    parser.add_argument("--missingness-rate", type=float, default=0.05)
    parser.add_argument(
        "--missingness-mode",
        choices=["nan", "zero", "both"],
        default="both",
    )
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 0.95:
        raise ValueError("train_ratio + val_ratio must be < 0.95")

    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = REPORT_DIR / "model_eval" / timestamp

    build_report(
        version=args.version,
        model_name=args.model,
        baseline_name=args.baseline,
        league_code=args.league,
        season_label=args.season,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        lookup=args.lookup,
        model_version=args.model_version,
        baseline_version=args.baseline_version,
        output_dir=output_dir,
        overconfidence_threshold=args.overconfidence_threshold,
        missingness_rate=args.missingness_rate,
        missingness_mode=args.missingness_mode,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
