from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.config import MODEL_DIR
from app.ml.registry import load_latest_model_by_name


def score_to_probs(home_score: float, away_score: float) -> tuple[float, float, float]:
    draw_score = max(0.1, 1.5 - abs(home_score - away_score))
    scores = [home_score, draw_score, away_score]
    exp_scores = [math.exp(s) for s in scores]
    total = sum(exp_scores) or 1.0
    home_prob = exp_scores[0] / total
    draw_prob = exp_scores[1] / total
    away_prob = exp_scores[2] / total
    return home_prob, draw_prob, away_prob


def predict_match(home_team: str, away_team: str) -> tuple[dict[str, Any], dict[str, Any]]:
    model, meta = load_latest_model_by_name(MODEL_DIR, "score_xgboost")
    if model is None:
        model, meta = load_latest_model_by_name(MODEL_DIR, "score_baseline")
    if model is None:
        home_score = 1.2
        away_score = 1.0
        model_meta = {
            "model_name": "heuristic",
            "model_version": "nogit",
        }
    else:
        features = pd.DataFrame([{"home_team": home_team, "away_team": away_team}])
        preds = model.predict(features)
        home_score = float(preds[0][0])
        away_score = float(preds[0][1])
        model_meta = {
            "model_name": meta.get("model_name", "model"),
            "model_version": meta.get("model_version", "unknown"),
        }

    home_score = max(0.0, home_score)
    away_score = max(0.0, away_score)

    home_prob, draw_prob, away_prob = score_to_probs(home_score, away_score)

    prediction = {
        "predicted_home_score": home_score,
        "predicted_away_score": away_score,
        "win_prob_home": home_prob,
        "win_prob_draw": draw_prob,
        "win_prob_away": away_prob,
    }
    return prediction, model_meta
