# aggregate_leagues_spyder.py
# Run in Spyder. Reads artifacts_sweep/* created by your generator and produces global PNG summaries.
# Output: artifacts_sweep/_global/

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# HARD CONFIG
# =========================
ARTIFACTS_DIRNAME = "artifacts_sweep"
GLOBAL_DIRNAME = "_global"

# how many top/bottom leagues to show in some charts
TOPN = 20

# choose which metric you treat as "primary" for ranking leagues
PRIMARY_METRIC = "best_rank_value"   # from summary_best_by_league.csv (usually best val_log_loss)
PRIMARY_LOWER_IS_BETTER = True       # val_log_loss / logloss / brier => True

# =========================
# Helpers
# =========================
def _base_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def save_bar(df: pd.DataFrame, x: str, y: str, out: Path, title: str, topn: int = 20, ascending: bool = True) -> None:
    d = df[[x, y]].dropna().copy()
    d = d.sort_values(y, ascending=ascending).head(topn)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(d[x].astype(str).values, d[y].astype(float).values)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_scatter(df: pd.DataFrame, x: str, y: str, out: Path, title: str) -> None:
    d = df[[x, y]].dropna().copy()
    xv = d[x].astype(float).values
    yv = d[y].astype(float).values
    corr_txt = ""
    if len(xv) >= 3 and np.std(xv) > 0 and np.std(yv) > 0:
        corr = np.corrcoef(xv, yv)[0, 1]
        corr_txt = f" (corr={corr:.3f})"

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xv, yv, s=25)
    ax.set_title(title + corr_txt)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================
# Core
# =========================
def load_best_summary(artifacts_dir: Path) -> pd.DataFrame:
    p = artifacts_dir / "summary_best_by_league.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run generator first.")
    df = pd.read_csv(p)
    return df


def load_trials_for_league(league_dir: Path) -> pd.DataFrame:
    # parquet preferred
    pq = league_dir / "trials_clean.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = league_dir / "trials_clean.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No trials file in {league_dir}")


def compute_val_test_corr_per_league(artifacts_dir: Path) -> pd.DataFrame:
    rows = []
    for league_dir in sorted(artifacts_dir.glob("league=*")):
        league = league_dir.name.split("league=")[-1]
        try:
            df = load_trials_for_league(league_dir)
        except Exception:
            continue

        # columns (support your aliases)
        val_ll = pick_col(df, ["val_log_loss", "val_log_lo", "val_logloss"])
        test_ll = pick_col(df, ["test_log_loss", "test_log_lo", "test_logloss"])
        val_f1 = pick_col(df, ["val_f1_macro", "val_f1_ma", "val_f1_m", "val_f1"])
        test_f1 = pick_col(df, ["test_f1_macro", "test_f1_m", "test_f1"])
        val_acc = pick_col(df, ["val_accuracy", "val_accur", "val_acc"])
        test_acc = pick_col(df, ["test_accuracy", "test_accur", "test_acc", "test_accurac"])

        def corr(a: Optional[str], b: Optional[str]) -> float:
            if not a or not b:
                return float("nan")
            d = df[[a, b]].dropna()
            if len(d) < 5:
                return float("nan")
            x = pd.to_numeric(d[a], errors="coerce").values
            y = pd.to_numeric(d[b], errors="coerce").values
            if np.std(x) == 0 or np.std(y) == 0:
                return float("nan")
            return float(np.corrcoef(x, y)[0, 1])

        rows.append({
            "league_code": league,
            "corr_val_test_log_loss": corr(val_ll, test_ll),
            "corr_val_test_f1_macro": corr(val_f1, test_f1),
            "corr_val_test_accuracy": corr(val_acc, test_acc),
            "n_trials": int(df.shape[0]),
            "train_size": float(pd.to_numeric(df.get("train_size"), errors="coerce").median()) if "train_size" in df.columns else float("nan"),
            "val_size": float(pd.to_numeric(df.get("val_size"), errors="coerce").median()) if "val_size" in df.columns else float("nan"),
            "test_size": float(pd.to_numeric(df.get("test_size"), errors="coerce").median()) if "test_size" in df.columns else float("nan"),
        })

    return pd.DataFrame(rows)


def main():
    base = _base_dir()
    artifacts_dir = base / ARTIFACTS_DIRNAME
    global_dir = artifacts_dir / GLOBAL_DIRNAME
    mkdirp(global_dir)

    best = load_best_summary(artifacts_dir)

    # Normalize numeric columns we care about
    for c in ["best_rank_value", "val_log_loss", "test_log_loss", "val_accuracy", "test_accuracy", "val_f1_macro", "test_f1_macro", "test_brier"]:
        if c in best.columns:
            best[c] = pd.to_numeric(best[c], errors="coerce")

    # --- Global league ranking charts ---
    if PRIMARY_METRIC in best.columns:
        save_bar(
            best,
            x="league_code",
            y=PRIMARY_METRIC,
            out=global_dir / f"leagues_top{TOPN}_by_{PRIMARY_METRIC}.png",
            title=f"Top {TOPN} leagues by {PRIMARY_METRIC}",
            topn=TOPN,
            ascending=PRIMARY_LOWER_IS_BETTER,
        )

    # best test_log_loss (lower better)
    if "test_log_loss" in best.columns:
        save_bar(
            best,
            x="league_code",
            y="test_log_loss",
            out=global_dir / f"leagues_top{TOPN}_by_test_log_loss.png",
            title=f"Top {TOPN} leagues by test_log_loss (lower is better)",
            topn=TOPN,
            ascending=True,
        )

    # best test_f1_macro (higher better)
    if "test_f1_macro" in best.columns:
        save_bar(
            best,
            x="league_code",
            y="test_f1_macro",
            out=global_dir / f"leagues_top{TOPN}_by_test_f1_macro.png",
            title=f"Top {TOPN} leagues by test_f1_macro (higher is better)",
            topn=TOPN,
            ascending=False,
        )

    # best test_accuracy (higher better)
    if "test_accuracy" in best.columns:
        save_bar(
            best,
            x="league_code",
            y="test_accuracy",
            out=global_dir / f"leagues_top{TOPN}_by_test_accuracy.png",
            title=f"Top {TOPN} leagues by test_accuracy (higher is better)",
            topn=TOPN,
            ascending=False,
        )

    # Gap charts: test - val (logloss)
    if "val_log_loss" in best.columns and "test_log_loss" in best.columns:
        best["gap_test_minus_val_log_loss"] = best["test_log_loss"] - best["val_log_loss"]
        save_bar(
            best.sort_values("gap_test_minus_val_log_loss", ascending=False),
            x="league_code",
            y="gap_test_minus_val_log_loss",
            out=global_dir / f"leagues_top{TOPN}_by_gap_test_minus_val_log_loss.png",
            title=f"Top {TOPN} leagues by (test_log_loss - val_log_loss) (higher = worse generalization)",
            topn=TOPN,
            ascending=False,
        )

        # Scatter val vs test for best models (one point per league)
        save_scatter(
            best.dropna(subset=["val_log_loss", "test_log_loss"]),
            x="val_log_loss",
            y="test_log_loss",
            out=global_dir / "scatter_best_per_league_val_vs_test_log_loss.png",
            title="Best-per-league: val_log_loss vs test_log_loss",
        )

    # --- Per league "val->test" correlation across trials ---
    corr_df = compute_val_test_corr_per_league(artifacts_dir)
    if not corr_df.empty:
        corr_df.to_csv(global_dir / "val_test_correlation_by_league.csv", index=False)

        # Show most stable leagues (highest corr)
        if "corr_val_test_log_loss" in corr_df.columns:
            save_bar(
                corr_df,
                x="league_code",
                y="corr_val_test_log_loss",
                out=global_dir / f"leagues_top{TOPN}_by_corr_val_test_log_loss.png",
                title=f"Top {TOPN} leagues by corr(val_log_loss, test_log_loss) across trials",
                topn=TOPN,
                ascending=False,
            )

    # Merge best + corr into one table
    if not corr_df.empty and "league_code" in corr_df.columns:
        merged = best.merge(corr_df, on="league_code", how="left")
        merged.to_csv(global_dir / "summary_best_plus_corr.csv", index=False)

    print(f"[OK] Global summaries written to: {global_dir}")


if __name__ == "__main__":
    main()
