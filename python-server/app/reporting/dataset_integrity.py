from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from app.config import REPORT_DIR
from app.db.models import Feature, League, Match, Season
from app.db.session import SessionLocal
from app.ml.dataset import (
    CANONICAL_KEY,
    DATE_COLUMN,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    assert_no_reorder_or_mismatch,
    temporal_split,
)


def load_raw_dataset(
    version: str,
    *,
    league_code: str | None,
    season_label: str | None,
) -> pd.DataFrame:
    feature_columns = [getattr(Feature, col) for col in FEATURE_COLUMNS]
    with SessionLocal() as session:
        query = (
            session.query(
                Match.id,
                Match.match_date,
                Match.result,
                Match.status,
                Match.home_team_id,
                Match.away_team_id,
                Match.home_goals,
                Match.away_goals,
                Season.label,
                League.code,
                *feature_columns,
            )
            .join(Feature, Feature.match_id == Match.id)
            .join(Season, Match.season_id == Season.id)
            .join(League, Season.league_id == League.id)
            .filter(Feature.version == version)
        )
        if league_code:
            query = query.filter(League.code == league_code)
        if season_label:
            query = query.filter(Season.label == season_label)
        rows = query.all()

    column_names = [
        CANONICAL_KEY,
        DATE_COLUMN,
        LABEL_COLUMN,
        "status",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
        "season_label",
        "league_code",
        *FEATURE_COLUMNS,
    ]
    return pd.DataFrame(rows, columns=column_names)


def summarize_step(df: pd.DataFrame, step: str) -> dict[str, object]:
    key = df[CANONICAL_KEY]
    feature_null_rows = int(df[FEATURE_COLUMNS].isna().any(axis=1).sum())
    feature_null_cells = int(df[FEATURE_COLUMNS].isna().sum().sum())
    return {
        "step": step,
        "rows": int(len(df)),
        "unique_keys": int(key.nunique()),
        "duplicate_keys": int(key.duplicated().sum()),
        "missing_keys": int(key.isna().sum()),
        "missing_labels": int(df[LABEL_COLUMN].isna().sum()),
        "feature_null_rows": feature_null_rows,
        "feature_null_cells": feature_null_cells,
    }


def render_markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = [str(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_missingness(
    split,
    output: Path,
) -> None:
    missing = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "train_missing": split.X_train.isna().sum().values,
            "val_missing": split.X_val.isna().sum().values,
        }
    )
    melted = missing.melt(
        id_vars="feature",
        value_vars=["train_missing", "val_missing"],
        var_name="split",
        value_name="missing",
    )

    height = max(6, int(len(FEATURE_COLUMNS) * 0.25))
    plt.figure(figsize=(12, height))
    sns.barplot(data=melted, y="feature", x="missing", hue="split")
    plt.xlabel("Missing values")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def build_report(
    *,
    version: str,
    league_code: str | None,
    season_label: str | None,
    train_ratio: float,
    val_ratio: float,
    only_played: bool,
    sample_size: int,
    seed: int,
    output_dir: Path,
) -> None:
    raw_df = load_raw_dataset(
        version,
        league_code=league_code,
        season_label=season_label,
    )
    if raw_df.empty:
        raise RuntimeError("No feature rows available. Run the ETL first.")

    steps: list[dict[str, object]] = []

    def add_step(name: str, frame: pd.DataFrame) -> pd.DataFrame:
        assert_no_reorder_or_mismatch(step=name, df=frame, key_name=CANONICAL_KEY)
        steps.append(summarize_step(frame, name))
        return frame

    df = add_step("raw_join", raw_df)
    df = add_step("filter_match_date", df[df[DATE_COLUMN].notna()].copy())
    df = add_step("filter_result", df[df[LABEL_COLUMN].notna()].copy())
    if only_played:
        df = add_step("filter_status_played", df[df["status"] == "played"].copy())
    df = add_step(
        "ordered",
        df.sort_values([DATE_COLUMN, CANONICAL_KEY]).reset_index(drop=True),
    )

    split = temporal_split(df, train_ratio=train_ratio, val_ratio=val_ratio)

    sorted_df = df.sort_values([DATE_COLUMN, CANONICAL_KEY]).reset_index(drop=True)
    train_df = sorted_df.loc[split.X_train.index]
    val_df = sorted_df.loc[split.X_val.index]
    test_df = sorted_df.loc[split.X_test.index]
    add_step("split_train", train_df)
    add_step("split_val", val_df)
    add_step("split_test", test_df)

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_columns = [
        CANONICAL_KEY,
        DATE_COLUMN,
        "season_label",
        "league_code",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
        LABEL_COLUMN,
        *FEATURE_COLUMNS[:6],
    ]
    sample_count = min(sample_size, len(df))
    sample = (
        df[sample_columns]
        .sample(n=sample_count, random_state=seed)
        .sort_values(DATE_COLUMN)
    )
    golden_path = output_dir / "golden_sample.csv"
    sample.to_csv(golden_path, index=False)

    chart_path = output_dir / "missingness_train_val.png"
    plot_missingness(split, chart_path)

    table_columns = [
        "step",
        "rows",
        "unique_keys",
        "duplicate_keys",
        "missing_keys",
        "missing_labels",
        "feature_null_rows",
        "feature_null_cells",
    ]
    table = render_markdown_table(steps, table_columns)

    report_path = output_dir / "dataset_integrity.md"
    report_lines = [
        "# Dataset Integrity Audit",
        "",
        f"- Generated: {datetime.utcnow().isoformat()}Z",
        f"- Feature version: {version}",
        f"- League filter: {league_code or 'all'}",
        f"- Season filter: {season_label or 'all'}",
        f"- Canonical key: {CANONICAL_KEY}",
        f"- Train ratio: {train_ratio}",
        f"- Val ratio: {val_ratio}",
        "",
        "## ETL Step Summary",
        table,
        "",
        "## Artifacts",
        f"- Golden sample: {golden_path}",
        f"- Missingness chart: {chart_path}",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Report saved to {report_path}")
    print(f"Golden sample saved to {golden_path}")
    print(f"Missingness chart saved to {chart_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--league", default=None)
    parser.add_argument("--season", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--only-played",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter to matches with status=played (default: enabled).",
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: MODEL_DIR/reports/<timestamp>)",
    )
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 0.95:
        raise ValueError("train_ratio + val_ratio must be < 0.95")

    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = REPORT_DIR / timestamp

    build_report(
        version=args.version,
        league_code=args.league,
        season_label=args.season,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        only_played=args.only_played,
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
