#!/usr/bin/env python3

import csv
import os

from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import func

from app.db.session import SessionLocal
from app.db.models import Match, Prediction, Season


def prediction_label(home, draw, away):
    if home >= draw and home >= away:
        return "Home"
    elif draw >= home and draw >= away:
        return "Draw"
    else:
        return "Away"


def highest_probability(home, draw, away):
    return max(home, draw, away)


def margin(home, draw, away):
    probabilities = sorted([home, draw, away], reverse=True)
    return probabilities[0] - probabilities[1]


def strength_from_margin(value):
    if value >= 15:
        return "Strong"
    elif value >= 7.5:
        return "Moderate"
    else:
        return "Close"


def main():
    today = date.today()

    # Football fixture week: Tuesday -> Monday.
    # On Sunday/Monday, use the next Tuesday as the start of the
    # upcoming fixture week.
    week_start = today - timedelta(days=(today.weekday() - 1) % 7)

    if today.weekday() in (6, 0):  # Sunday or Monday
        week_start += timedelta(days=7)

    week_end = week_start + timedelta(days=6)

    output_dir = Path(
        os.getenv(
            "PREDICTIONS_OUTPUT_DIR",
            str(Path.home() / "Downloads"),
        )
    )
    output_path = (
        output_dir
        / f"football-predictions-{week_start.isoformat()}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = SessionLocal()

    try:
        # Find the latest prediction for each match/model combination.
        latest_xgb = (
            session.query(
                Prediction.match_id,
                func.max(Prediction.created_at).label("latest_created_at"),
            )
            .filter(Prediction.model_name == "main_xgboost")
            .group_by(Prediction.match_id)
            .subquery()
        )

        latest_baseline = (
            session.query(
                Prediction.match_id,
                func.max(Prediction.created_at).label("latest_created_at"),
            )
            .filter(Prediction.model_name == "baseline_classifier")
            .group_by(Prediction.match_id)
            .subquery()
        )

        # Get scheduled fixtures for the current football week.
        matches = (
            session.query(Match, Season)
            .join(Season, Season.id == Match.season_id)
            .filter(
                Match.status == "scheduled",
                Match.match_date >= week_start,
                Match.match_date <= week_end,
            )
            .order_by(
                Match.match_date,
                Match.match_time,
                Match.season_id,
            )
            .all()
        )

        # Look up the latest XGBoost predictions.
        xgb_predictions = {}

        xgb_rows = (
            session.query(Prediction)
            .join(
                latest_xgb,
                (Prediction.match_id == latest_xgb.c.match_id)
                & (
                    Prediction.created_at
                    == latest_xgb.c.latest_created_at
                ),
            )
            .filter(Prediction.model_name == "main_xgboost")
            .all()
        )

        for prediction in xgb_rows:
            xgb_predictions[prediction.match_id] = prediction

        # Look up the latest baseline predictions.
        baseline_predictions = {}

        baseline_rows = (
            session.query(Prediction)
            .join(
                latest_baseline,
                (Prediction.match_id == latest_baseline.c.match_id)
                & (
                    Prediction.created_at
                    == latest_baseline.c.latest_created_at
                ),
            )
            .filter(Prediction.model_name == "baseline_classifier")
            .all()
        )

        for prediction in baseline_rows:
            baseline_predictions[prediction.match_id] = prediction

        # Build export rows.
        export_rows = []

        for match, season in matches:
            xgb = xgb_predictions.get(match.id)
            baseline = baseline_predictions.get(match.id)

            # XGBoost prediction.
            if xgb:
                xgb_home = xgb.win_prob_home * 100
                xgb_draw = xgb.win_prob_draw * 100
                xgb_away = xgb.win_prob_away * 100

                xgb_prediction = prediction_label(
                    xgb_home, xgb_draw, xgb_away
                )
                xgb_highest = highest_probability(
                    xgb_home, xgb_draw, xgb_away
                )
            else:
                xgb_home = xgb_draw = xgb_away = None
                xgb_prediction = ""
                xgb_highest = None

            # Baseline prediction.
            if baseline:
                baseline_home = baseline.win_prob_home * 100
                baseline_draw = baseline.win_prob_draw * 100
                baseline_away = baseline.win_prob_away * 100

                baseline_prediction = prediction_label(
                    baseline_home, baseline_draw, baseline_away
                )
                baseline_highest = highest_probability(
                    baseline_home, baseline_draw, baseline_away
                )
            else:
                baseline_home = baseline_draw = baseline_away = None
                baseline_prediction = ""
                baseline_highest = None

            # Comparison and combined prediction.
            if xgb and baseline:
                combined_home = (xgb_home + baseline_home) / 2
                combined_draw = (xgb_draw + baseline_draw) / 2
                combined_away = (xgb_away + baseline_away) / 2

                combined_prediction = prediction_label(
                    combined_home, combined_draw, combined_away
                )
                combined_highest = highest_probability(
                    combined_home, combined_draw, combined_away
                )
                combined_margin = margin(
                    combined_home, combined_draw, combined_away
                )
                strength = strength_from_margin(combined_margin)

                models_agree = (
                    "Yes"
                    if xgb_prediction == baseline_prediction
                    else "No"
                )
                model_difference = abs(
                    xgb_highest - baseline_highest
                )
            else:
                combined_home = combined_draw = combined_away = None
                combined_prediction = ""
                combined_highest = None
                combined_margin = None
                strength = ""
                models_agree = "No"
                model_difference = None

            export_rows.append([
                match.match_date.isoformat(),
                (
                    match.match_time.strftime("%H:%M")
                    if match.match_time
                    else ""
                ),
                season.league.name,
                match.home_team.name,
                match.away_team.name,
                xgb_home,
                xgb_draw,
                xgb_away,
                xgb_prediction,
                xgb_highest,
                baseline_home,
                baseline_draw,
                baseline_away,
                baseline_prediction,
                baseline_highest,
                models_agree,
                model_difference,
                combined_home,
                combined_draw,
                combined_away,
                combined_prediction,
                combined_highest,
                combined_margin,
                strength,
            ])

        # Sort by combined highest probability, descending.
        export_rows.sort(
            key=lambda row: (
                row[21] is not None,
                row[21] if row[21] is not None else 0,
            ),
            reverse=True,
        )

        # Write CSV.
        with output_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow([
                "Date",
                "Time",
                "League",
                "Home",
                "Away",
                "XGB Home %",
                "XGB Draw %",
                "XGB Away %",
                "XGB Prediction",
                "XGB Highest %",
                "Baseline Home %",
                "Baseline Draw %",
                "Baseline Away %",
                "Baseline Prediction",
                "Baseline Highest %",
                "Models Agree?",
                "Model Difference %",
                "Combined Home %",
                "Combined Draw %",
                "Combined Away %",
                "Combined Prediction",
                "Combined Highest %",
                "Combined Margin %",
                "Strength",
            ])

            for row in export_rows:
                formatted = []

                for index, value in enumerate(row):
                    if index in (
                        5, 6, 7, 9,
                        10, 11, 12, 14,
                        16,
                        17, 18, 19, 21, 22,
                    ):
                        if value is None:
                            formatted.append("")
                        else:
                            formatted.append(f"{value:.1f}")
                    else:
                        formatted.append(value)

                writer.writerow(formatted)

        print(f"Week: {week_start} to {week_end}")
        print(f"Matches exported: {len(export_rows)}")
        print(f"XGB predictions available: {len(xgb_predictions)}")
        print(
            "Baseline predictions available: "
            f"{len(baseline_predictions)}"
        )
        print(f"CSV written: {output_path}")

    finally:
        session.close()


if __name__ == "__main__":
    main()
