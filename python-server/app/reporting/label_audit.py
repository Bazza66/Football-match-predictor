from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns

from app.config import REPORT_DIR
from app.db.models import ModelRun
from app.db.session import SessionLocal
from app.ml.labels import LABELS, LABEL_TO_ID


def _render_markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _plot_pred_vs_true(
    true_counts: Mapping[str, int],
    pred_counts: Mapping[str, int],
    output: Path,
) -> None:
    labels = LABELS
    true_values = [int(true_counts.get(label, 0)) for label in labels]
    pred_values = [int(pred_counts.get(label, 0)) for label in labels]

    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar([val - width / 2 for val in x], true_values, width, label="True")
    plt.bar([val + width / 2 for val in x], pred_values, width, label="Predicted")
    plt.xticks(list(x), labels)
    plt.ylabel("Count")
    plt.title("Predicted vs True Class Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = []
    lines.append(",".join(headers))
    for row in rows:
        lines.append(",".join(row))
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_confusion_matrix(
    confusion_matrix: list[list[int]],
    output: Path,
) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _update_index(entries: list[dict]) -> None:
    index_path = REPORT_DIR / "label_audits_index.json"
    index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _load_index() -> list[dict]:
    index_path = REPORT_DIR / "label_audits_index.json"
    if not index_path.exists():
        return []
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _compute_accuracy(confusion_matrix: list[list[int]]) -> float:
    total = sum(sum(int(val) for val in row) for row in confusion_matrix)
    correct = sum(int(confusion_matrix[idx][idx]) for idx in range(len(LABELS)))
    return float(correct / total) if total else 0.0


def write_label_audit(
    *,
    run_id: str,
    label_counts_train: Mapping[str, int],
    label_counts_val: Mapping[str, int],
    confusion_matrix: list[list[int]],
    metrics: Mapping[str, float] | None,
    league_code: str | None = None,
    season_label: str | None = None,
) -> dict[str, str]:
    report_dir = REPORT_DIR / f"label_audit_{run_id}"
    report_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(tz=timezone.utc)

    mapping_rows = [[label, str(LABEL_TO_ID[label])] for label in LABELS]
    mapping_table = _render_markdown_table(mapping_rows, ["Label", "Id"])

    count_rows = []
    for label in LABELS:
        count_rows.append(["train", label, str(int(label_counts_train.get(label, 0)))])
        count_rows.append(["val", label, str(int(label_counts_val.get(label, 0)))])
    counts_table = _render_markdown_table(count_rows, ["Split", "Label", "Count"])
    counts_csv_rows = [["train", label, str(int(label_counts_train.get(label, 0)))] for label in LABELS]
    counts_csv_rows += [["val", label, str(int(label_counts_val.get(label, 0)))] for label in LABELS]

    cm_rows = []
    for row_label, row_values in zip(LABELS, confusion_matrix):
        cm_rows.append([row_label] + [str(int(value)) for value in row_values])
    cm_headers = ["True \\ Pred", *LABELS]
    cm_table = _render_markdown_table(cm_rows, cm_headers)

    confusion_csv_rows = []
    for row_label, row_values in zip(LABELS, confusion_matrix):
        confusion_csv_rows.append([row_label] + [str(int(value)) for value in row_values])

    chart_path = report_dir / "predicted_vs_true.png"
    confusion_png_path = report_dir / "confusion_matrix.png"
    _plot_pred_vs_true(label_counts_val, _pred_counts_from_confusion(confusion_matrix), chart_path)
    _plot_confusion_matrix(confusion_matrix, confusion_png_path)

    class_counts_path = report_dir / "class_counts.csv"
    _write_csv(class_counts_path, ["split", "label", "count"], counts_csv_rows)

    confusion_path = report_dir / "confusion_matrix.csv"
    _write_csv(confusion_path, cm_headers, confusion_csv_rows)

    metrics_payload = {
        "val_accuracy": float(metrics.get("val_accuracy")) if metrics else None,
        "val_f1_macro": float(metrics.get("val_f1_macro")) if metrics else None,
        "val_log_loss": float(metrics.get("val_log_loss")) if metrics else None,
        "val_brier_score": float(metrics.get("val_brier_score")) if metrics else None,
        "accuracy_from_confusion": _compute_accuracy(confusion_matrix),
    }
    if metrics and metrics_payload["val_accuracy"] is not None:
        if abs(metrics_payload["val_accuracy"] - metrics_payload["accuracy_from_confusion"]) > 1e-6:
            raise RuntimeError("Confusion matrix accuracy does not match val_accuracy.")

    metrics_path = report_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    report_path = report_dir / "label_audit.md"
    report_path.write_text(
        "\n".join(
            [
                "# Label Audit Report",
                "",
                f"- Run id: {run_id}",
                f"- Created: {created_at.isoformat()}",
                "",
                "## Label Mapping",
                mapping_table,
                "",
                "## Class Counts (Train/Val)",
                counts_table,
                "",
                "## Confusion Matrix (Validation)",
                cm_table,
                "",
                "## Metrics (Validation)",
                f"- accuracy: {metrics_payload['val_accuracy']}",
                f"- f1_macro: {metrics_payload['val_f1_macro']}",
                f"- log_loss: {metrics_payload['val_log_loss']}",
                f"- brier_score: {metrics_payload['val_brier_score']}",
                "",
                "## Files",
                f"- class_counts.csv: {class_counts_path}",
                f"- confusion_matrix.csv: {confusion_path}",
                f"- metrics.json: {metrics_path}",
                f"- predicted_vs_true.png: {chart_path}",
                f"- confusion_matrix.png: {confusion_png_path}",
            ]
        ),
        encoding="utf-8",
    )

    entries = _load_index()
    entry = {
        "report_id": report_dir.name,
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "league_code": league_code,
        "season_label": season_label,
        "report_file": report_path.name,
        "chart_file": chart_path.name,
        "confusion_file": confusion_png_path.name,
    }
    entries = [item for item in entries if item.get("report_id") != entry["report_id"]]
    entries.append(entry)
    entries.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    _update_index(entries)

    return {
        "report": str(report_path),
        "chart": str(chart_path),
        "confusion": str(confusion_png_path),
        "counts": str(class_counts_path),
        "matrix": str(confusion_path),
        "metrics": str(metrics_path),
    }


def _pred_counts_from_confusion(confusion_matrix: list[list[int]]) -> dict[str, int]:
    pred_counts: dict[str, int] = {label: 0 for label in LABELS}
    for col_idx, label in enumerate(LABELS):
        pred_counts[label] = int(sum(row[col_idx] for row in confusion_matrix))
    return pred_counts


def build_label_audit_from_run(
    *,
    run_id: str,
    metrics: Mapping[str, float],
    label_counts: Mapping[str, Mapping[str, int]] | None,
    league_code: str | None = None,
    season_label: str | None = None,
) -> dict[str, str]:
    confusion_matrix = metrics.get("val_confusion_matrix")
    if confusion_matrix is None:
        raise RuntimeError("Missing val_confusion_matrix in metrics.")
    label_counts = label_counts or {}
    return write_label_audit(
        run_id=run_id,
        label_counts_train=label_counts.get("train", {}),
        label_counts_val=label_counts.get("val", {}),
        confusion_matrix=confusion_matrix,
        metrics=metrics,
        league_code=league_code,
        season_label=season_label,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use latest model run if --run-id is not provided.",
    )
    args = parser.parse_args()

    run_id = args.run_id
    with SessionLocal() as session:
        if not run_id:
            if not args.latest:
                raise RuntimeError("Provide --run-id or use --latest.")
            latest = (
                session.query(ModelRun)
                .order_by(ModelRun.created_at.desc())
                .first()
            )
            if not latest:
                raise RuntimeError("No model runs found.")
            run_id = latest.run_id
        run = session.query(ModelRun).filter(ModelRun.run_id == run_id).one_or_none()
        if not run:
            raise RuntimeError(f"Run id {run_id} not found.")
        metrics = run.metrics or {}
        label_counts = None
        if run.data_summary and isinstance(run.data_summary, dict):
            label_counts = run.data_summary.get("label_counts")

    build_label_audit_from_run(
        run_id=run_id,
        metrics=metrics,
        label_counts=label_counts,
        league_code=run.league_code,
        season_label=run.season_label,
    )
    print(f"Label audit written for run {run_id}.")


if __name__ == "__main__":
    main()
