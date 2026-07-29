from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)

from app.config import MODEL_DIR, REPORT_DIR
from app.ml.labels import LABELS
from app.db.models import Match, Prediction
from app.db.session import SessionLocal



def load_predictions() -> pd.DataFrame:
    with SessionLocal() as session:
        rows = (
            session.query(
                Prediction.model_name,
                Prediction.model_version,
                Match.result,
                Prediction.win_prob_home,
                Prediction.win_prob_draw,
                Prediction.win_prob_away,
            )
            .join(Match, Prediction.match_id == Match.id)
            .filter(Match.status == "played")
            .filter(Match.result.isnot(None))
            .all()
        )

    records = []
    for model_name, model_version, result, p_home, p_draw, p_away in rows:
        probs = [p_home, p_draw, p_away]
        if any(p is None for p in probs):
            continue
        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        records.append(
            {
                "model_name": model_name,
                "model_version": model_version,
                "y_true": result,
                "y_pred": LABELS[pred_idx],
                "p_home": float(p_home),
                "p_draw": float(p_draw),
                "p_away": float(p_away),
            }
        )

    return pd.DataFrame(records)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (model_name, model_version), group in df.groupby([
        "model_name",
        "model_version",
    ]):
        y_true = group["y_true"]
        y_pred = group["y_pred"]
        prob_matrix = group[["p_home", "p_draw", "p_away"]].to_numpy()

        metrics = {
            "model_name": model_name,
            "model_version": model_version,
            "samples": len(group),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }
        try:
            metrics["log_loss"] = float(log_loss(y_true, prob_matrix, labels=LABELS))
        except ValueError:
            metrics["log_loss"] = float("nan")

        summaries.append(metrics)
    return pd.DataFrame(summaries)


def plot_bars(summary: pd.DataFrame, column: str, output: Path) -> None:
    if summary.empty:
        return
    summary = summary.sort_values(column, ascending=False)
    labels = summary.apply(
        lambda row: f"{row['model_name']}\n{row['model_version']}", axis=1
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(x=labels, y=summary[column], palette="Blues_d")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(column.replace("_", " "))
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def plot_confusion(group: pd.DataFrame, output: Path) -> None:
    y_true = group["y_true"]
    y_pred = group["y_pred"]
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def build_report(output_dir: Path) -> None:
    df = load_predictions()
    if df.empty:
        raise RuntimeError("No predictions with probabilities found.")

    summary = compute_metrics(df)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "prediction_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_bars(summary, "accuracy", output_dir / "accuracy_by_model.png")
    plot_bars(summary, "log_loss", output_dir / "logloss_by_model.png")

    for (model_name, model_version), group in df.groupby([
        "model_name",
        "model_version",
    ]):
        safe_name = f"{model_name}_{model_version}".replace(":", "-")
        plot_confusion(group, output_dir / f"confusion_{safe_name}.png")

    print(f"Report saved to {output_dir}")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: MODEL_DIR/reports/<timestamp>)",
    )
    args = parser.parse_args()

    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = REPORT_DIR / timestamp

    build_report(output_dir)


if __name__ == "__main__":
    main()
