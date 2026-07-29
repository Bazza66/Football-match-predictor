from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import Feature, Match
from app.db.session import SessionLocal

CANONICAL_KEY = "match_id"
LABEL_COLUMN = "result"
DATE_COLUMN = "match_date"

FEATURE_COLUMNS = [
    "home_points_last_5",
    "home_points_last_10",
    "away_points_last_5",
    "away_points_last_10",
    "home_wins_last_5",
    "home_draws_last_5",
    "home_losses_last_5",
    "away_wins_last_5",
    "away_draws_last_5",
    "away_losses_last_5",
    "home_wins_last_10",
    "home_draws_last_10",
    "home_losses_last_10",
    "away_wins_last_10",
    "away_draws_last_10",
    "away_losses_last_10",
    "home_goals_for_last_5",
    "home_goals_against_last_5",
    "home_goal_diff_last_5",
    "away_goals_for_last_5",
    "away_goals_against_last_5",
    "away_goal_diff_last_5",
    "home_goals_for_last_10",
    "home_goals_against_last_10",
    "home_goal_diff_last_10",
    "away_goals_for_last_10",
    "away_goals_against_last_10",
    "away_goal_diff_last_10",
    "home_points_last_5_home_games",
    "away_points_last_5_away_games",
    "home_goals_for_home_last_5",
    "away_goals_for_away_last_5",
    "points_season_so_far_home",
    "points_season_so_far_away",
    "goal_diff_season_so_far_home",
    "goal_diff_season_so_far_away",
    "h2h_points_last_3",
    "h2h_goal_diff_last_3",
    "days_since_last_match_home",
    "days_since_last_match_away",
    "diff_points_last_5",
    "diff_goal_diff_last_5",
    "diff_days_rest",
    "home_goals_for_rate_last_5",
    "home_goals_against_rate_last_5",
    "away_goals_for_rate_last_5",
    "away_goals_against_rate_last_5",
    "home_goals_for_rate_last_10",
    "home_goals_against_rate_last_10",
    "away_goals_for_rate_last_10",
    "away_goals_against_rate_last_10",
    "home_elo_overall",
    "away_elo_overall",
    "home_elo_home",
    "away_elo_away",
    "elo_diff_overall",
    "elo_diff_home_away",
    "home_form_ema_points",
    "away_form_ema_points",
    "home_form_ema_goal_diff",
    "away_form_ema_goal_diff",
    "diff_form_ema_points",
    "diff_form_ema_goal_diff",
    "home_win_streak",
    "away_win_streak",
    "home_draw_streak",
    "away_draw_streak",
    "home_loss_streak",
    "away_loss_streak",
    "home_unbeaten_streak",
    "away_unbeaten_streak",
    "home_home_streak",
    "away_away_streak",
]


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def assert_no_reorder_or_mismatch(
    *,
    step: str,
    key_name: str = CANONICAL_KEY,
    df: pd.DataFrame | None = None,
    keys: pd.Series | None = None,
    expected_keys: pd.Series | None = None,
    allow_reorder: bool = False,
    X: pd.DataFrame | None = None,
    y: pd.Series | pd.DataFrame | None = None,
) -> None:
    if keys is None:
        if df is not None:
            if key_name not in df.columns:
                raise RuntimeError(f"{step}: missing {key_name} column for integrity check.")
            keys = df[key_name]
        elif X is not None and isinstance(X, pd.DataFrame) and key_name in X.columns:
            keys = X[key_name]
        elif isinstance(y, pd.DataFrame) and key_name in y.columns:
            keys = y[key_name]
        else:
            raise RuntimeError(f"{step}: no keys or dataframe supplied for integrity check.")

    missing_keys = int(keys.isna().sum())
    if missing_keys:
        raise RuntimeError(f"{step}: {missing_keys} missing {key_name} values detected.")

    dup_mask = keys.duplicated()
    dup_count = int(dup_mask.sum())
    if dup_count:
        sample_dups = keys[dup_mask].head(5).tolist()
        raise RuntimeError(
            f"{step}: {dup_count} duplicate {key_name} values detected "
            f"(sample: {sample_dups})."
        )

    if expected_keys is not None:
        actual_keys = keys.reset_index(drop=True)
        expected_keys = expected_keys.reset_index(drop=True)
        if len(actual_keys) != len(expected_keys):
            raise RuntimeError(
                f"{step}: key length changed from {len(expected_keys)} to {len(actual_keys)}."
            )
        if allow_reorder:
            expected_index = pd.Index(expected_keys)
            actual_index = pd.Index(actual_keys)
            missing = expected_index.difference(actual_index)
            extra = actual_index.difference(expected_index)
            if not missing.empty or not extra.empty:
                raise RuntimeError(
                    f"{step}: key set mismatch (missing={len(missing)}, extra={len(extra)})."
                )
        else:
            if not actual_keys.equals(expected_keys):
                raise RuntimeError(f"{step}: key order mismatch detected.")

    if X is not None and y is not None:
        if len(X) != len(y):
            raise RuntimeError(
                f"{step}: X/y length mismatch ({len(X)} vs {len(y)})."
            )
        if X.index.equals(y.index):
            return
        x_keys = X[key_name] if key_name in X.columns else None
        y_keys = None
        if isinstance(y, pd.DataFrame) and key_name in y.columns:
            y_keys = y[key_name]
        if x_keys is None or y_keys is None:
            raise RuntimeError(
                f"{step}: X/y index mismatch and {key_name} columns not available."
            )
        if not x_keys.reset_index(drop=True).equals(y_keys.reset_index(drop=True)):
            raise RuntimeError(f"{step}: X/y {key_name} mismatch detected.")


def _rows_to_frame(rows: list[tuple], include_match_id: bool) -> pd.DataFrame:
    records: list[dict] = []
    for row in rows:
        offset = 0
        data: dict = {}
        if include_match_id:
            data[CANONICAL_KEY] = row[offset]
            offset += 1
        data[DATE_COLUMN] = row[offset]
        offset += 1
        data[LABEL_COLUMN] = row[offset]
        offset += 1
        for idx, col in enumerate(FEATURE_COLUMNS):
            data[col] = row[offset + idx]
        records.append(data)
    return pd.DataFrame(records)


def load_feature_dataset(
    version: str,
    only_played: bool = True,
    league_code: str | None = None,
    season_label: str | None = None,
    include_match_id: bool = True,
) -> pd.DataFrame:
    columns = [getattr(Feature, col) for col in FEATURE_COLUMNS]
    with SessionLocal() as session:
        if include_match_id:
            query = session.query(Match.id, Match.match_date, Match.result, *columns)
        else:
            query = session.query(Match.match_date, Match.result, *columns)
        query = query.join(Feature, Feature.match_id == Match.id).filter(
            Feature.version == version
        )
        if league_code or season_label:
            from app.db.models import League, Season

            query = query.join(Season, Match.season_id == Season.id).join(
                League, Season.league_id == League.id
            )
            if league_code:
                query = query.filter(League.code == league_code)
            if season_label:
                query = query.filter(Season.label == season_label)
        query = query.filter(Match.match_date.isnot(None)).filter(
            Match.result.isnot(None)
        )
        if only_played:
            query = query.filter(Match.status == "played")
        rows = query.order_by(Match.match_date.asc(), Match.id.asc()).all()
    df = _rows_to_frame(rows, include_match_id=include_match_id)
    if include_match_id:
        assert_no_reorder_or_mismatch(step="load_feature_dataset", df=df)
    return df


def load_features_for_prediction(
    session: Session,
    version: str,
    only_played: bool = False,
    include_match_id: bool = True,
):
    columns = [getattr(Feature, col) for col in FEATURE_COLUMNS]
    query = (
        session.query(Match.id, Match.match_date, Match.result, *columns)
        .join(Feature, Feature.match_id == Match.id)
        .filter(Feature.version == version)
        .filter(Match.match_date.isnot(None))
    )
    if only_played:
        query = query.filter(Match.status == "played")
    rows = query.order_by(Match.match_date.asc(), Match.id.asc()).all()
    df = _rows_to_frame(rows, include_match_id=include_match_id)
    if include_match_id:
        assert_no_reorder_or_mismatch(step="load_features_for_prediction", df=df)
    return df


def temporal_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> DatasetSplit:
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

    train_end, val_end = _temporal_split_indices(n, train_ratio, val_ratio)

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

    return DatasetSplit(X_train, y_train, X_val, y_val, X_test, y_test)


def _temporal_split_indices(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    val_end = min(val_end, n - 1)
    return train_end, val_end


def temporal_split_date_ranges(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> dict[str, tuple[str | None, str | None]]:
    if DATE_COLUMN not in df.columns:
        raise RuntimeError(
            f"temporal_split: missing {DATE_COLUMN} column (date column required)."
        )
    if CANONICAL_KEY not in df.columns:
        raise RuntimeError(
            f"temporal_split: missing {CANONICAL_KEY} column (canonical key required)."
        )
    sorted_df = df.sort_values([DATE_COLUMN, CANONICAL_KEY]).reset_index(drop=True)
    n = len(sorted_df)
    if n < 1:
        return {
            "train": (None, None),
            "val": (None, None),
            "test": (None, None),
        }

    train_end, val_end = _temporal_split_indices(n, train_ratio, val_ratio)
    train = sorted_df.iloc[:train_end]
    val = sorted_df.iloc[train_end:val_end]
    test = sorted_df.iloc[val_end:]

    def _range(frame: pd.DataFrame) -> tuple[str | None, str | None]:
        if frame.empty:
            return None, None
        return str(frame[DATE_COLUMN].min()), str(frame[DATE_COLUMN].max())

    return {
        "train": _range(train),
        "val": _range(val),
        "test": _range(test),
    }
