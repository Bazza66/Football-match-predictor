from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.config import REPORT_DIR
from app.db.models import ModelRun
from app.db.session import SessionLocal
from app.ml.labels import LABELS, label_counts


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = []
    lines.append(",".join(headers))
    for row in rows:
        lines.append(",".join(row))
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_loss_lr(step_metrics: list[dict], output: Path, best_step: int | None) -> None:
    steps = [row["step"] for row in step_metrics]
    train_losses = [
        row.get("train_logloss") if row.get("train_logloss") is not None else np.nan
        for row in step_metrics
    ]
    val_losses = [
        row.get("val_logloss") if row.get("val_logloss") is not None else np.nan
        for row in step_metrics
    ]
    lrs = [
        row.get("learning_rate") if row.get("learning_rate") is not None else np.nan
        for row in step_metrics
    ]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps, train_losses, color="#6b7280", label="Train logloss")
    ax1.plot(steps, val_losses, color="#1f77b4", label="Val logloss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Log loss")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    if best_step is not None and best_step in steps:
        idx = steps.index(best_step)
        best_val = val_losses[idx]
        if np.isfinite(best_val):
            ax1.scatter(
                [best_step],
                [best_val],
                color="#f97316",
                zorder=5,
                label="Best val",
            )
            ax1.axvline(best_step, color="#f97316", linestyle="--", alpha=0.6)

    if np.isfinite(lrs).any():
        ax2 = ax1.twinx()
        ax2.plot(steps, lrs, color="#ff7f0e", linestyle="--", label="Learning rate")
        ax2.set_ylabel("Learning rate")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_grad_norm(step_metrics: list[dict], output: Path) -> None:
    steps = [row["step"] for row in step_metrics if row.get("grad_norm") is not None]
    norms = [row.get("grad_norm") for row in step_metrics if row.get("grad_norm") is not None]
    if not steps:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(steps, norms, color="#2ca02c")
    plt.xlabel("Step")
    plt.ylabel("Gradient norm")
    plt.title("Gradient norm vs step")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _detect_divergence(
    step_metrics: list[dict],
    *,
    k_consecutive: int,
    val_pred: Iterable[int],
) -> dict[str, object]:
    val_losses = [row.get("val_logloss") for row in step_metrics]
    loss_increase_step = None
    streak = 0
    for idx in range(1, len(val_losses)):
        current = val_losses[idx]
        prev = val_losses[idx - 1]
        if current is None or prev is None:
            streak = 0
            continue
        if current > prev:
            streak += 1
        else:
            streak = 0
        if streak >= k_consecutive:
            loss_increase_step = step_metrics[idx]["step"]
            break

    nan_inf_steps = []
    for row in step_metrics:
        train_loss = row.get("train_logloss")
        val_loss = row.get("val_logloss")
        if train_loss is not None and not np.isfinite(train_loss):
            nan_inf_steps.append(row["step"])
        if val_loss is not None and not np.isfinite(val_loss):
            nan_inf_steps.append(row["step"])

    pred_list = list(val_pred)
    if pred_list:
        pred_counts = label_counts(pred_list)
    else:
        pred_counts = {label: 0 for label in LABELS}
    total_preds = sum(pred_counts.values())
    saturated = any(count == total_preds and total_preds > 0 for count in pred_counts.values())

    best_step = None
    best_val = None
    for row in step_metrics:
        val_loss = row.get("val_logloss")
        if val_loss is None:
            continue
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_step = row["step"]

    first_overfit_step = None
    if best_step is not None:
        steps = [row["step"] for row in step_metrics]
        start_idx = steps.index(best_step) + 1
        for idx in range(start_idx, len(step_metrics) - k_consecutive + 1):
            window = step_metrics[idx : idx + k_consecutive]
            if any(item.get("val_logloss") is None for item in window):
                continue
            if best_val is None:
                continue
            if all(item["val_logloss"] > best_val for item in window):
                first_overfit_step = step_metrics[idx]["step"]
                break

    final_minus_best = None
    if step_metrics:
        final_val = step_metrics[-1].get("val_logloss")
        if final_val is not None and best_val is not None:
            final_minus_best = float(final_val - best_val)

    return {
        "loss_increase_step": loss_increase_step,
        "nan_inf_steps": sorted(set(nan_inf_steps)),
        "output_saturated": saturated,
        "pred_counts": pred_counts,
        "best_step": best_step,
        "best_val_logloss": best_val,
        "first_overfit_step": first_overfit_step,
        "final_minus_best": final_minus_best,
    }


def _scaling_summary(stats_df: pd.DataFrame) -> dict[str, float]:
    spans = stats_df["p99"] - stats_df["p01"]
    spans = spans.replace(0, np.nan)
    max_span = float(spans.max()) if spans.notna().any() else float("nan")
    min_span = float(spans.min()) if spans.notna().any() else float("nan")
    span_ratio = float(max_span / min_span) if min_span and max_span else float("nan")
    outlier_count = int(stats_df["outlier_flag"].sum()) if "outlier_flag" in stats_df else 0
    return {
        "span_ratio": span_ratio,
        "max_span": max_span,
        "min_span": min_span,
        "outlier_count": outlier_count,
    }


def build_step_metrics_from_eval(
    train_losses: list[float] | None,
    val_losses: list[float] | None,
    *,
    learning_rate: float | None,
) -> list[dict]:
    if not val_losses:
        return []
    steps = []
    for idx, val_loss in enumerate(val_losses, start=1):
        train_loss = None
        if train_losses is not None and idx - 1 < len(train_losses):
            train_loss = train_losses[idx - 1]
        steps.append(
            {
                "step": idx,
                "train_logloss": float(train_loss) if train_loss is not None else None,
                "val_logloss": float(val_loss) if val_loss is not None else None,
                "learning_rate": float(learning_rate) if learning_rate is not None else None,
                "grad_norm": None,
                "weight_norm": None,
            }
        )
    return steps


def step_metrics_from_xgboost(
    evals_result: Mapping[str, Mapping[str, list[float]]] | None,
    learning_rate: float | None,
) -> list[dict]:
    if not evals_result:
        return []
    if "validation_0" in evals_result and "validation_1" in evals_result:
        train = evals_result.get("validation_0") or {}
        val = evals_result.get("validation_1") or {}
    elif "train" in evals_result and "validation" in evals_result:
        train = evals_result.get("train") or {}
        val = evals_result.get("validation") or {}
    elif "train" in evals_result and "eval" in evals_result:
        train = evals_result.get("train") or {}
        val = evals_result.get("eval") or {}
    else:
        train = None
        val = evals_result.get("validation_0") or evals_result.get("eval") or {}

    train_losses = train.get("mlogloss") if train else None
    val_losses = val.get("mlogloss") if val else None

    return build_step_metrics_from_eval(
        train_losses,
        val_losses,
        learning_rate=learning_rate,
    )


def step_metrics_from_catboost(
    evals_result: Mapping[str, Mapping[str, list[float]]] | None,
    learning_rate: float | None,
) -> list[dict]:
    if not evals_result:
        return []
    train = evals_result.get("learn") or evals_result.get("train")
    val = evals_result.get("validation") or evals_result.get("valid") or {}
    loss_key = "MultiClass" if "MultiClass" in val else next(iter(val.keys()), None)
    if loss_key is None:
        return []
    train_losses = train.get(loss_key) if train else None
    val_losses = val.get(loss_key)
    return build_step_metrics_from_eval(
        train_losses,
        val_losses,
        learning_rate=learning_rate,
    )


def step_metrics_from_summary(
    val_loss: float | None,
    learning_rate: float | None,
) -> list[dict]:
    return [
        {
            "step": 1,
            "train_logloss": None,
            "val_logloss": float(val_loss) if val_loss is not None else None,
            "learning_rate": float(learning_rate) if learning_rate is not None else None,
            "grad_norm": None,
            "weight_norm": None,
        }
    ]


def write_training_divergence_report(
    *,
    run_id: str,
    model_name: str,
    league_code: str | None,
    season_label: str | None,
    step_metrics: list[dict],
    feature_stats: pd.DataFrame,
    val_true: Iterable[int],
    val_pred: Iterable[int],
    learning_rate: float | None,
    k_consecutive: int = 5,
) -> dict[str, str]:
    report_dir = REPORT_DIR / f"training_divergence_{run_id}"
    report_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(tz=timezone.utc)

    if not step_metrics:
        raise RuntimeError("No step metrics available for divergence report.")

    if feature_stats.empty:
        feature_stats = pd.DataFrame(
            columns=[
                "feature",
                "median",
                "p01",
                "p99",
                "q25",
                "q75",
                "iqr",
                "min",
                "max",
                "missing_pct",
                "outlier_flag",
            ]
        )

    scaling_csv = report_dir / "feature_scaling.csv"
    feature_stats.to_csv(scaling_csv, index=False)

    step_metrics_path = report_dir / "step_metrics.csv"
    _write_csv(
        step_metrics_path,
        ["step", "train_logloss", "val_logloss", "learning_rate"],
        [
            [
                str(row.get("step")),
                str(row.get("train_logloss")),
                str(row.get("val_logloss")),
                str(row.get("learning_rate")),
            ]
            for row in step_metrics
        ],
    )

    divergence = _detect_divergence(
        step_metrics,
        k_consecutive=k_consecutive,
        val_pred=val_pred,
    )

    divergence_summary_path = report_dir / "divergence_summary.csv"
    summary_rows = []
    if divergence["loss_increase_step"]:
        step = divergence["loss_increase_step"]
        record = next(row for row in step_metrics if row["step"] == step)
        summary_rows.append(
            [
                "loss_increase",
                str(step),
                "N/A",
                str(record.get("val_logloss")),
                "N/A",
                str(record.get("learning_rate")),
            ]
        )
    if divergence["nan_inf_steps"]:
        summary_rows.append(
            [
                "nan_or_inf",
                str(divergence["nan_inf_steps"][0]),
                "N/A",
                "nan/inf",
                "nan/inf",
                "nan/inf",
            ]
        )
    if divergence["output_saturated"]:
        summary_rows.append(
            [
                "output_saturated",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
            ]
        )
    if divergence.get("best_step") is not None:
        best_step = divergence["best_step"]
        best_row = next(row for row in step_metrics if row["step"] == best_step)
        summary_rows.append(
            [
                "best_step",
                str(best_step),
                "N/A",
                str(best_row.get("val_logloss")),
                "N/A",
                str(best_row.get("learning_rate")),
            ]
        )
    if divergence.get("first_overfit_step") is not None:
        overfit_step = divergence["first_overfit_step"]
        overfit_row = next(row for row in step_metrics if row["step"] == overfit_step)
        summary_rows.append(
            [
                "first_overfit_step",
                str(overfit_step),
                "N/A",
                str(overfit_row.get("val_logloss")),
                "N/A",
                str(overfit_row.get("learning_rate")),
            ]
        )
    if divergence.get("final_minus_best") is not None:
        summary_rows.append(
            [
                "final_minus_best",
                "final",
                "N/A",
                str(divergence["final_minus_best"]),
                "N/A",
                "N/A",
            ]
        )
    if not summary_rows:
        summary_rows.append(["none", "N/A", "N/A", "N/A", "N/A", "N/A"])

    _write_csv(
        divergence_summary_path,
        ["reason", "step", "batch", "loss", "accuracy", "learning_rate"],
        summary_rows,
    )

    loss_lr_path = report_dir / "loss_lr.png"
    _plot_loss_lr(step_metrics, loss_lr_path, divergence.get("best_step"))

    grad_norm_path = report_dir / "grad_norm.png"
    _plot_grad_norm(step_metrics, grad_norm_path)
    if not grad_norm_path.exists():
        grad_norm_path = None

    scaling_summary = _scaling_summary(feature_stats) if not feature_stats.empty else {}
    diagnosis = []
    if divergence["loss_increase_step"] and learning_rate and learning_rate > 0.1:
        diagnosis.append("Learning rate may be too high (loss increases).")
    if scaling_summary and scaling_summary.get("span_ratio", 0) > 100:
        diagnosis.append(
            "Feature ranges vary widely (p01-p99 span ratio > 100). Trees don't require scaling, but huge ranges often mean unit/outlier bugs."
        )
    if scaling_summary and scaling_summary.get("outlier_count", 0) > 0:
        diagnosis.append(
            f"Detected {int(scaling_summary.get('outlier_count', 0))} features with extreme outliers."
        )
    if divergence["nan_inf_steps"]:
        diagnosis.append("NaNs/inf detected; check loss setup or numeric stability.")
    if divergence["output_saturated"]:
        diagnosis.append("Outputs saturated to one class; check class balance or loss.")
    if not diagnosis:
        diagnosis.append("No obvious divergence detected.")

    report_path = report_dir / "training_divergence.md"
    report_path.write_text(
        "\n".join(
            [
                "# Training Divergence Report",
                "",
                f"- Run id: {run_id}",
                f"- Model: {model_name}",
                f"- Created: {created_at.isoformat()}",
                f"- League: {league_code or 'all'}",
                f"- Season: {season_label or 'all'}",
                "",
                "## First Divergence Point",
                f"- loss_increase_step: {divergence['loss_increase_step']}",
                f"- nan_inf_steps: {divergence['nan_inf_steps']}",
                f"- output_saturated: {divergence['output_saturated']}",
                "",
                "## Overfit Diagnostics",
                f"- best_step: {divergence['best_step']}",
                f"- first_overfit_step: {divergence['first_overfit_step']}",
                f"- final_minus_best: {divergence['final_minus_best']}",
                "",
                "## Diagnosis",
                *[f"- {item}" for item in diagnosis],
                "",
                "## Files",
                f"- step_metrics.csv: {step_metrics_path}",
                f"- feature_scaling.csv: {scaling_csv}",
                f"- divergence_summary.csv: {divergence_summary_path}",
                f"- loss_lr.png: {loss_lr_path}",
                f"- grad_norm.png: {grad_norm_path or 'N/A'}",
            ]
        ),
        encoding="utf-8",
    )

    index_path = REPORT_DIR / "training_divergence_index.json"
    entries = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            entries = []
    entry = {
        "report_id": report_dir.name,
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "league_code": league_code,
        "season_label": season_label,
        "report_file": report_path.name,
        "loss_chart_file": loss_lr_path.name,
        "grad_chart_file": grad_norm_path.name if grad_norm_path else None,
    }
    entries = [item for item in entries if item.get("report_id") != entry["report_id"]]
    entries.append(entry)
    entries.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    return {
        "report": str(report_path),
        "loss_chart": str(loss_lr_path),
        "grad_chart": str(grad_norm_path) if grad_norm_path else "",
        "step_metrics": str(step_metrics_path),
        "feature_scaling": str(scaling_csv),
        "divergence_summary": str(divergence_summary_path),
    }


def compute_feature_scaling_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "median",
                "p01",
                "p99",
                "q25",
                "q75",
                "iqr",
                "min",
                "max",
                "missing_pct",
                "outlier_flag",
            ]
        )

    q01 = df.quantile(0.01, numeric_only=True)
    q25 = df.quantile(0.25, numeric_only=True)
    q50 = df.quantile(0.50, numeric_only=True)
    q75 = df.quantile(0.75, numeric_only=True)
    q99 = df.quantile(0.99, numeric_only=True)
    iqr = q75 - q25
    min_vals = df.min(skipna=True)
    max_vals = df.max(skipna=True)
    missing_pct = df.isna().mean()

    stats = pd.DataFrame(
        {
            "feature": df.columns,
            "median": q50.reindex(df.columns).values,
            "p01": q01.reindex(df.columns).values,
            "p99": q99.reindex(df.columns).values,
            "q25": q25.reindex(df.columns).values,
            "q75": q75.reindex(df.columns).values,
            "iqr": iqr.reindex(df.columns).values,
            "min": min_vals.reindex(df.columns).values,
            "max": max_vals.reindex(df.columns).values,
            "missing_pct": missing_pct.reindex(df.columns).values,
        }
    )

    iqr_series = stats["iqr"].replace(0, np.nan)
    high_outlier = stats["max"] > (stats["p99"] + 10 * iqr_series)
    low_outlier = stats["min"] < (stats["p01"] - 10 * iqr_series)
    stats["outlier_flag"] = (high_outlier | low_outlier).fillna(False)
    return stats


def build_report_from_run(run: ModelRun) -> None:
    metrics = run.metrics or {}
    val_pred = []
    step_metrics = step_metrics_from_summary(
        metrics.get("val_log_loss"),
        None,
    )
    write_training_divergence_report(
        run_id=run.run_id,
        model_name=run.model_name,
        league_code=run.league_code,
        season_label=run.season_label,
        step_metrics=step_metrics,
        feature_stats=pd.DataFrame(),
        val_true=[],
        val_pred=val_pred,
        learning_rate=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id
    with SessionLocal() as session:
        if not run_id:
            if not args.latest:
                raise RuntimeError("Provide --run-id or use --latest.")
            run = session.query(ModelRun).order_by(ModelRun.created_at.desc()).first()
            if not run:
                raise RuntimeError("No model runs found.")
        else:
            run = session.query(ModelRun).filter(ModelRun.run_id == run_id).one_or_none()
            if not run:
                raise RuntimeError(f"Run id {run_id} not found.")

    build_report_from_run(run)
    print(f"Divergence report written for run {run.run_id}.")


if __name__ == "__main__":
    main()
