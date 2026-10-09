#!/usr/bin/env python3

import os

from app.db.session import SessionLocal
from app.db.models import Season, League
from app.etl.feature_engineering import generate_features_for_season

SEASON_LABEL = os.getenv("SEASON_LABEL", "2026-27")

LEAGUE_CODES = [
    "english-premier-league",
    "english-championship",
    "english-league-one",
    "english-league-two",
    "english-conference",
]


def main():
    print(f"Generating features for season {SEASON_LABEL}")

    session = SessionLocal()

    try:
        seasons = (
            session.query(Season)
            .join(League, Season.league_id == League.id)
            .filter(
                Season.label == SEASON_LABEL,
                League.code.in_(LEAGUE_CODES),
            )
            .order_by(League.code)
            .all()
        )

        found_codes = {season.league.code for season in seasons}
        missing_codes = set(LEAGUE_CODES) - found_codes

        if missing_codes:
            missing = ", ".join(sorted(missing_codes))
            raise RuntimeError(
                f"Missing seasons for {SEASON_LABEL}: {missing}"
            )

        for season in seasons:
            result = generate_features_for_season(
                session,
                season,
                version="v1",
            )

            print(
                f"{season.id} | {season.label} | "
                f"{season.league.name} | {result}"
            )

        session.commit()
        print("\nFeatures committed successfully.")

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


if __name__ == "__main__":
    main()