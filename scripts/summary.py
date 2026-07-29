# generate_artifacts_spyder.py
# Hardcoded runner for Spyder: reads sweep_reesults.csv (or sweep_results.csv),
# generates ONLY tables/PNG/charts under artifacts_sweep/.

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# HARD-CODED CONFIG (edit here)
# ============================================================

CSV_FILENAMES = [
    "sweep_reesults.csv",  # <- your name (as you wrote)
    "sweep_results.csv",   # <- common corrected name fallback
]

OUTDIR_NAME = "artifacts_sweep"

TOPK = 10
RANK_METRIC = "val_log_loss"   # e.g. "val_f1_macro", "val_accuracy"
LEAGUE_FILTER = None           # e.g. "EPL" or None for all

# If None: auto infer (loss/brier => ascending, acc/f1 => descending)
RANK_ASCENDING = None

# ============================================================
# Helpers: parsing + sanity
# ============================================================

def _base_dir() -> Path:
    # In Spyder, __file__ is usually defined when running a file.
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def safe_read_csv(path: Path) -> pd.DataFrame:
    """
    Reads CSV with auto separator detection (comma/semicolon/tab),
    handles common encodings.
    """
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path, sep=",")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_parse_params_cell(x: Any) -> Dict[str, Any]:
    """
    Parses 'params' column values into dict.
    Accepts JSON strings, python dict-like strings, or already-dict.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, (list, tuple)):
        return {"_list": list(x)}

    s = str(x).strip()
    if not s:
        return {}

    # JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # python literal
    try:
        v = ast.literal_eval(s)
        if isinstance(v, dict):
            return v
        return {"_value": v}
    except Exception:
        pass

    return {"_raw": s}


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pick_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Table-to-PNG (matplotlib)
# ============================================================

def dataframe_to_png_table(
    df: pd.DataFrame,
    out_path: Path,
    title: str = "",
    max_rows: int = 25,
    max_cols: int = 18,
    float_fmt: str = "{:.4f}",
) -> None:
    """
    Saves a DataFrame as a PNG table (matplotlib).
    Keep it readable by limiting rows/cols.
    """
    df_disp = df.copy()

    if df_disp.shape[0] > max_rows:
        df_disp = df_disp.head(max_rows)

    if df_disp.shape[1] > max_cols:
        df_disp = df_disp.iloc[:, :max_cols]

    for c in df_disp.columns:
        if pd.api.types.is_float_dtype(df_disp[c]):
            df_disp[c] = df_disp[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))

    df_disp = df_disp.astype(str)

    nrows, ncols = df_disp.shape
    fig_w = max(10, min(24, 0.9 * ncols + 6))
    fig_h = max(2.5, min(18, 0.45 * nrows + 2.5))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, pad=12)

    table = ax.table(
        cellText=df_disp.values,
        colLabels=df_disp.columns.tolist(),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(10)
            cell.set_height(cell.get_height() * 1.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plot helpers
# ============================================================

def save_line_plot(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str) -> None:
    d = df[[x_col, y_col]].dropna().sort_values(x_col)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(d[x_col].values, d[y_col].values, marker="o", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_hist_plot(series: pd.Series, out_path: Path, title: str, bins: int = 25) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(s.values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(series.name or "value")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str) -> None:
    d = df[[x_col, y_col]].dropna()
    x = d[x_col].values
    y = d[y_col].values

    corr_txt = ""
    if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
        corr = np.corrcoef(x, y)[0, 1]
        corr_txt = f"  (corr={corr:.3f})"

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=25)
    ax.set_title(title + corr_txt)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Core processing
# ============================================================

@dataclass
class RunOptions:
    csv_path: Path
    outdir: Path
    topk: int
    rank_metric: str
    rank_ascending: bool
    league_filter: Optional[str]


def infer_required_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    league_col = pick_existing(df, ["league_code", "league", "liga", "leagueId"])
    iteration_col = pick_existing(df, ["iteration", "iter", "trial", "trial_id"])
    params_col = pick_existing(df, ["params", "parameters", "param_grid", "hyperparams"])

    if league_col is None:
        raise ValueError("Cannot find league column. Expected something like: league_code / league.")
    if iteration_col is None:
        raise ValueError("Cannot find iteration column. Expected: iteration / iter / trial.")
    if params_col is None:
        params_col = "params"
        if "params" not in df.columns:
            df["params"] = None

    return league_col, iteration_col, params_col


def build_clean_trials(df: pd.DataFrame, iteration_col: str, params_col: str) -> pd.DataFrame:
    df = df.copy()
    df[iteration_col] = pd.to_numeric(df[iteration_col], errors="coerce")

    df["_params_dict"] = df[params_col].apply(safe_parse_params_cell)

    common_keys = [
        "learning_rate", "max_depth", "n_estimators", "subsample",
        "colsample_bytree", "gamma", "min_child_weight", "reg_alpha", "reg_lambda"
    ]
    for k in common_keys:
        df[f"param_{k}"] = df["_params_dict"].apply(lambda d: d.get(k, np.nan))

    metric_candidates = [
        "val_accuracy", "val_acc", "val_accur",
        "val_log_loss", "val_log_lo", "val_logloss",
        "val_f1_macro", "val_f1_ma", "val_f1",
        "test_accuracy", "test_accur", "test_acc", "test_accurac",
        "test_log_loss", "test_log_lo", "test_logloss",
        "test_f1_macro", "test_f1_m", "test_f1",
        "test_brier", "test_brier_score",
        "train_size", "val_size", "test_size",
    ]
    df = ensure_numeric(df, metric_candidates)
    return df


def select_metric_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested

    aliases = {
        "val_log_loss": ["val_log_loss", "val_log_lo", "val_logloss"],
        "val_accuracy": ["val_accuracy", "val_accur", "val_acc"],
        "val_f1_macro": ["val_f1_macro", "val_f1_ma", "val_f1_m", "val_f1"],
        "test_accuracy": ["test_accuracy", "test_accur", "test_acc", "test_accurac"],
        "test_log_loss": ["test_log_loss", "test_log_lo", "test_logloss"],
        "test_f1_macro": ["test_f1_macro", "test_f1_m", "test_f1"],
        "test_brier": ["test_brier", "test_brier_score"],
    }

    if requested in aliases:
        for c in aliases[requested]:
            if c in df.columns:
                return c

    for c in df.columns:
        if requested.lower() in c.lower():
            return c

    raise ValueError(f"Requested metric '{requested}' not found in CSV columns.")


def best_row(df_league: pd.DataFrame, metric_col: str, ascending: bool) -> pd.Series:
    d = df_league.dropna(subset=[metric_col]).sort_values(metric_col, ascending=ascending)
    if d.empty:
        return df_league.iloc[0]
    return d.iloc[0]


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def generate_league_artifacts(
    df_league: pd.DataFrame,
    league_code: str,
    league_dir: Path,
    iteration_col: str,
    metric_rank_col: str,
    rank_ascending: bool,
    topk: int,
) -> Dict[str, Any]:
    mkdirp(league_dir)

    # Save cleaned trials for this league (parquet if available)
    try:
        df_league.to_parquet(league_dir / "trials_clean.parquet", index=False)
    except Exception:
        df_league.to_csv(league_dir / "trials_clean.csv", index=False)

    # TopK table
    top = (
        df_league.dropna(subset=[metric_rank_col])
        .sort_values(metric_rank_col, ascending=rank_ascending)
        .head(topk)
    )

    top_table_cols: List[str] = []
    preferred_cols = [
        iteration_col,
        metric_rank_col,
        "val_accuracy", "val_accur", "val_f1_macro", "val_f1_ma",
        "test_accuracy", "test_accur", "test_accurac", "test_f1_macro", "test_f1_m",
        "test_log_loss", "test_log_lo",
        "test_brier",
        "train_size", "val_size", "test_size",
        "param_learning_rate", "param_max_depth", "param_colsample_bytree", "param_gamma",
    ]
    for c in preferred_cols:
        if c in top.columns and c not in top_table_cols:
            top_table_cols.append(c)
    if not top_table_cols:
        top_table_cols = top.columns.tolist()[:12]

    dataframe_to_png_table(
        top[top_table_cols],
        league_dir / f"top{topk}.png",
        title=f"Top {topk} — league={league_code} (rank by {metric_rank_col})",
        max_rows=topk,
        max_cols=18,
    )
    top[top_table_cols].to_csv(league_dir / f"top{topk}.csv", index=False)

    # Best row JSON
    br = best_row(df_league, metric_rank_col, ascending=rank_ascending)
    br_dict = br.to_dict()
    for k, v in list(br_dict.items()):
        if isinstance(v, (np.integer, np.int64)):
            br_dict[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            br_dict[k] = float(v)
        elif isinstance(v, (np.ndarray,)):
            br_dict[k] = v.tolist()
    write_json(league_dir / "best_row.json", br_dict)

    # Plots
    plot_specs = [
        ("val_log_loss", "line_val_log_loss.png", "val log loss over iterations"),
        ("val_accuracy", "line_val_accuracy.png", "val accuracy over iterations"),
        ("val_f1_macro", "line_val_f1_macro.png", "val f1-macro over iterations"),
        ("test_accuracy", "line_test_accuracy.png", "test accuracy over iterations"),
        ("test_log_loss", "line_test_log_loss.png", "test log loss over iterations"),
        ("test_f1_macro", "line_test_f1_macro.png", "test f1-macro over iterations"),
        ("test_brier", "line_test_brier.png", "test brier over iterations"),
    ]

    for logical, fname, ttl in plot_specs:
        try:
            col = select_metric_column(df_league, logical)
        except Exception:
            continue
        if iteration_col in df_league.columns and col in df_league.columns:
            save_line_plot(
                df_league.dropna(subset=[iteration_col, col]),
                x_col=iteration_col,
                y_col=col,
                out_path=league_dir / fname,
                title=f"{ttl} (league={league_code})",
            )
            save_hist_plot(
                df_league[col],
                out_path=league_dir / f"hist_{col}.png",
                title=f"Histogram: {col} (league={league_code})",
                bins=25,
            )

    # Scatter: val vs test logloss
    try:
        vll = select_metric_column(df_league, "val_log_loss")
        tll = select_metric_column(df_league, "test_log_loss")
        save_scatter_plot(
            df_league,
            x_col=vll,
            y_col=tll,
            out_path=league_dir / "scatter_val_vs_test_log_loss.png",
            title=f"val vs test log loss (league={league_code})",
        )
    except Exception:
        pass

    # Scatter: val vs test f1-macro
    try:
        vf1 = select_metric_column(df_league, "val_f1_macro")
        tf1 = select_metric_column(df_league, "test_f1_macro")
        save_scatter_plot(
            df_league,
            x_col=vf1,
            y_col=tf1,
            out_path=league_dir / "scatter_val_f1_macro_vs_test_f1_macro.png",
            title=f"val vs test f1-macro (league={league_code})",
        )
    except Exception:
        pass

    # Summary row
    def _maybe(logical: str, candidates: List[str]) -> Any:
        if any(k in df_league.columns for k in candidates):
            try:
                return br.get(select_metric_column(df_league, logical), np.nan)
            except Exception:
                return np.nan
        return np.nan

    return {
        "league_code": league_code,
        "best_iteration": br.get(iteration_col, np.nan),
        "rank_metric": metric_rank_col,
        "best_rank_value": br.get(metric_rank_col, np.nan),
        "val_log_loss": _maybe("val_log_loss", ["val_log_loss", "val_log_lo", "val_logloss"]),
        "val_accuracy": _maybe("val_accuracy", ["val_accuracy", "val_accur", "val_acc"]),
        "val_f1_macro": _maybe("val_f1_macro", ["val_f1_macro", "val_f1_ma", "val_f1_m", "val_f1"]),
        "test_accuracy": _maybe("test_accuracy", ["test_accuracy", "test_accur", "test_acc", "test_accurac"]),
        "test_log_loss": _maybe("test_log_loss", ["test_log_loss", "test_log_lo", "test_logloss"]),
        "test_f1_macro": _maybe("test_f1_macro", ["test_f1_macro", "test_f1_m", "test_f1"]),
        "test_brier": _maybe("test_brier", ["test_brier", "test_brier_score"]),
        "train_size": br.get("train_size", np.nan),
        "val_size": br.get("val_size", np.nan),
        "test_size": br.get("test_size", np.nan),
    }


def generate_global_artifacts(summary_rows: List[Dict[str, Any]], outdir: Path, rank_metric: str) -> None:
    if not summary_rows:
        return
    df_sum = pd.DataFrame(summary_rows)

    df_sum = df_sum.sort_values("best_rank_value", ascending=True, na_position="last")
    df_sum.to_csv(outdir / "summary_best_by_league.csv", index=False)

    cols = [c for c in [
        "league_code",
        "best_iteration",
        "best_rank_value",
        "val_log_loss",
        "val_accuracy",
        "val_f1_macro",
        "test_accuracy",
        "test_log_loss",
        "test_f1_macro",
        "test_brier",
        "train_size",
        "val_size",
        "test_size",
    ] if c in df_sum.columns]

    dataframe_to_png_table(
        df_sum[cols],
        outdir / "summary_best_by_league.png",
        title=f"Best trial per league (rank metric: {rank_metric})",
        max_rows=80,
        max_cols=18,
        float_fmt="{:.5f}",
    )

    d = df_sum.dropna(subset=["best_rank_value"]).copy()
    if not d.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(d["league_code"].astype(str).values, d["best_rank_value"].astype(float).values)
        ax.set_title(f"Best {rank_metric} by league (lower is better)")
        ax.set_xlabel("league")
        ax.set_ylabel(rank_metric)
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(outdir / f"global_best_{rank_metric}_by_league.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def run(opts: RunOptions) -> None:
    mkdirp(opts.outdir)

    df_raw = safe_read_csv(opts.csv_path)
    df_raw = normalize_columns(df_raw)

    league_col, iteration_col, params_col = infer_required_columns(df_raw)
    df = build_clean_trials(df_raw, iteration_col=iteration_col, params_col=params_col)

    metric_rank_col = select_metric_column(df, opts.rank_metric)

    leagues = sorted(df[league_col].dropna().astype(str).unique().tolist())
    if opts.league_filter:
        leagues = [l for l in leagues if l == opts.league_filter]

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "csv_path": str(opts.csv_path),
        "outdir": str(opts.outdir),
        "topk": opts.topk,
        "rank_metric_requested": opts.rank_metric,
        "rank_metric_used": metric_rank_col,
        "rank_ascending": opts.rank_ascending,
        "league_col": league_col,
        "iteration_col": iteration_col,
        "params_col": params_col,
        "leagues": leagues,
        "n_rows_total": int(df.shape[0]),
        "columns": df.columns.tolist(),
    }
    write_json(opts.outdir / "manifest.json", manifest)

    summary_rows: List[Dict[str, Any]] = []
    for league in leagues:
        df_league = df[df[league_col].astype(str) == str(league)].copy()
        league_dir = opts.outdir / f"league={league}"
        row = generate_league_artifacts(
            df_league=df_league,
            league_code=str(league),
            league_dir=league_dir,
            iteration_col=iteration_col,
            metric_rank_col=metric_rank_col,
            rank_ascending=opts.rank_ascending,
            topk=opts.topk,
        )
        summary_rows.append(row)

    generate_global_artifacts(summary_rows=summary_rows, outdir=opts.outdir, rank_metric=metric_rank_col)

    print(f"[OK] Generated artifacts in: {opts.outdir}")
    print(f"[OK] Leagues: {', '.join(leagues) if leagues else '(none)'}")


# ============================================================
# Spyder entrypoint (NO CLI)
# ============================================================

if __name__ == "__main__":
    base = _base_dir()

    csv_path = None
    for name in CSV_FILENAMES:
        p = (base / name)
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        # Helpful error with directory listing
        files = [x.name for x in base.glob("*.csv")]
        raise FileNotFoundError(
            f"Cannot find any of: {CSV_FILENAMES}\n"
            f"Looked in: {base}\n"
            f"CSV files present: {files}"
        )

    outdir = base / OUTDIR_NAME

    # Auto-rank direction if not provided
    if RANK_ASCENDING is None:
        rm = str(RANK_METRIC).lower()
        if "loss" in rm or "brier" in rm:
            rank_ascending = True
        else:
            rank_ascending = False
    else:
        rank_ascending = bool(RANK_ASCENDING)

    opts = RunOptions(
        csv_path=csv_path,
        outdir=outdir,
        topk=int(TOPK),
        rank_metric=str(RANK_METRIC),
        rank_ascending=rank_ascending,
        league_filter=LEAGUE_FILTER,
    )

    run(opts)
