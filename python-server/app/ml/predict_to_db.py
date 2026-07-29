from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from app.config import MODEL_DIR
from app.db.models import Prediction
from app.db.session import SessionLocal
from app.ml.dataset import FEATURE_COLUMNS, load_features_for_prediction
from app.ml.labels import LABELS
from app.ml.registry import load_latest_model_by_name


def resolve_class_order(meta: dict[str, Any] | None) -> list[str]:
    if meta and isinstance(meta.get("classes"), list):
        return [str(item) for item in meta["classes"]]
    return LABELS


def predict_to_db(
    version: str,
    model_name: str,
    only_missing: bool = True,
) -> int:
    model, meta = load_latest_model_by_name(MODEL_DIR, model_name)
    if model is None:
        raise RuntimeError(f"No model artifacts found for {model_name} in {MODEL_DIR}.")

    class_order = resolve_class_order(meta)
    class_index = {label: idx for idx, label in enumerate(class_order)}
    if meta and meta.get("feature_version") and meta.get("feature_version") != version:
        print(
            f"Warning: model was trained on feature version {meta.get('feature_version')} "
            f"but predictions requested for {version}."
        )

    with SessionLocal() as session:
        model_version = meta.get("model_version", "unknown")
        if not only_missing:
            session.query(Prediction).filter(
                Prediction.model_name == model_name,
                Prediction.model_version == model_version,
            ).delete(synchronize_session=False)
            session.commit()

        df = load_features_for_prediction(session, version, include_match_id=True)
        if df.empty:
            print("No feature rows available.")
            return 0

        if only_missing:
            existing_ids = {
                row[0]
                for row in session.query(Prediction.match_id)
                .filter(
                    Prediction.model_name == model_name,
                    Prediction.model_version == model_version,
                )
                .all()
            }
            if existing_ids:
                df = df[~df["match_id"].isin(existing_ids)]

        if df.empty:
            print("No matches to score.")
            return 0

        X = df[FEATURE_COLUMNS]
        probs = model.predict_proba(X)

        inserted = 0
        for idx, match_id in enumerate(df["match_id"].astype(int).tolist()):
            home_prob = (
                float(probs[idx][class_index["H"]]) if "H" in class_index else None
            )
            draw_prob = (
                float(probs[idx][class_index["D"]]) if "D" in class_index else None
            )
            away_prob = (
                float(probs[idx][class_index["A"]]) if "A" in class_index else None
            )

            prediction = Prediction(
                match_id=match_id,
                model_name=model_name,
                model_version=model_version,
                predicted_home_score=None,
                predicted_away_score=None,
                win_prob_home=home_prob,
                win_prob_draw=draw_prob,
                win_prob_away=away_prob,
            )
            session.add(prediction)
            inserted += 1

        session.commit()

    print(f"Inserted {inserted} predictions for model {model_name} ({model_version}).")
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--model", default="main_xgboost")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    predict_to_db(version=args.version, model_name=args.model, only_missing=not args.all)


if __name__ == "__main__":
    main()
