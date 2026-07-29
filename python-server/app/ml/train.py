from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

from app.config import DATA_DIR, MODEL_DIR
from sqlalchemy.orm import aliased

from app.db.models import League, Match, Season, Team
from app.db.session import SessionLocal
from app.etl.load_football_json import iter_match_rows
from app.ml.artifacts import git_sha, save_model
from app.ml.tracking import build_artifact_info, record_run


def load_matches_from_db() -> pd.DataFrame:
    with SessionLocal() as session:
        home_team = aliased(Team)
        away_team = aliased(Team)
        rows = (
            session.query(Match, Season, League, home_team, away_team)
            .join(Season, Match.season_id == Season.id)
            .join(League, Season.league_id == League.id)
            .join(home_team, Match.home_team_id == home_team.id)
            .join(away_team, Match.away_team_id == away_team.id)
            .all()
        )
    data = []
    for match, season, league, home, away in rows:
        data.append(
            {
                "league": league.code,
                "season": season.label,
                "round": match.round,
                "date": match.match_date,
                "home_team": home.name,
                "away_team": away.name,
                "home_score": match.home_goals,
                "away_score": match.away_goals,
            }
        )
    return pd.DataFrame(data)


def load_matches_from_files(data_dir: Path) -> pd.DataFrame:
    if not data_dir.exists():
        return pd.DataFrame()
    records = list(iter_match_rows(data_dir))
    df = pd.DataFrame(records)
    if "home_goals" in df.columns:
        df = df.rename(columns={"home_goals": "home_score", "away_goals": "away_score"})
    return df


def build_pipeline(model_name: str) -> Pipeline:
    preprocess = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["home_team", "away_team"])],
        remainder="drop",
    )

    if model_name == "xgboost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    else:
        estimator = Ridge(alpha=1.0)

    model = MultiOutputRegressor(estimator)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def train(model_name: str, limit: int | None = None) -> None:
    df = load_matches_from_db()
    if df.empty:
        df = load_matches_from_files(DATA_DIR)

    df = df.dropna(subset=["home_score", "away_score"]).copy()
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42)

    if df.empty:
        raise RuntimeError("No training data available.")

    X = df[["home_team", "away_team"]]
    y = df[["home_score", "away_score"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(model_name)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    model_key = f"score_{model_name}"
    run_id = uuid.uuid4().hex
    model_path, meta, artifact_paths = save_model(
        pipeline,
        model_name=model_key,
        model_dir=MODEL_DIR,
        metrics={"mae": float(mae)},
        extra_meta={
            "model_type": "score_regressor",
        },
        feature_schema={"columns": ["home_team", "away_team"]},
        run_id=run_id,
    )

    artifacts = []
    for path in artifact_paths:
        name = path.name
        if name.endswith("_schema.json"):
            artifact_type = "feature_schema"
        elif path.suffix == ".pkl":
            artifact_type = "model"
        else:
            artifact_type = "model_meta"
        artifacts.append(build_artifact_info(path, artifact_type, MODEL_DIR))
    record_run(
        run_id=run_id,
        model_name=model_key,
        model_version=meta["model_version"],
        feature_version="raw",
        commit_hash=git_sha(),
        league_code=None,
        season_label=None,
        metrics={"mae": float(mae)},
        params={"model_type": model_name},
        data_summary={"train_size": int(len(X_train)), "test_size": int(len(X_test))},
        artifacts=artifacts,
    )

    print(f"Saved model to {model_path}")
    print(f"MAE: {mae:.4f}")
    print(f"Meta: {meta}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "baseline"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    train(args.model, args.limit)


if __name__ == "__main__":
    main()
