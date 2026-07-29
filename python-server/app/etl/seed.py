from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import func, text

from app.config import DATA_DIR
from app.db.base import Base
from app.db.models import Feature, League, Match, Season, Team
from app.db.session import SessionLocal, engine
from app.etl.load_football_json import LoadSummary, run_etl


def print_report(load_summary: LoadSummary, feature_summary, feature_version: str) -> None:
    with SessionLocal() as session:
        league_count = session.query(func.count(League.id)).scalar() or 0
        season_count = session.query(func.count(Season.id)).scalar() or 0
        team_count = session.query(func.count(Team.id)).scalar() or 0
        match_total = session.query(func.count(Match.id)).scalar() or 0
        match_played = (
            session.query(func.count(Match.id))
            .filter(Match.status == "played")
            .scalar()
            or 0
        )
        feature_count = (
            session.query(func.count(Feature.match_id))
            .filter(Feature.version == feature_version)
            .scalar()
            or 0
        )

    coverage_denominator = feature_summary.total_matches or 1
    coverage = feature_count / coverage_denominator

    print("ETL summary")
    print(f"Leagues: {league_count}")
    print(f"Seasons: {season_count}")
    print(f"Teams: {team_count}")
    print(f"Matches: {match_total} (played {match_played})")
    print(
        f"Features ({feature_version}): {feature_count}/{feature_summary.total_matches} "
        f"({coverage:.1%} coverage, {feature_summary.eligible_matches} eligible)"
    )
    if feature_summary.leakage_warnings:
        print(f"Leakage warnings: {feature_summary.leakage_warnings}")

    print("Top issues")
    print(f"missing_date: {load_summary.missing_date}")
    print(f"missing_team: {load_summary.missing_team}")
    print(f"duplicate_matches: {load_summary.duplicate_matches}")



def seed(
    limit: int = 1000,
    data_dir: Path | None = None,
    force: bool = False,
    feature_version: str = "v1",
) -> int:
    data_dir = data_dir or DATA_DIR
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        existing = session.query(func.count(Match.id)).scalar() or 0
    if existing > 0 and not force:
        print(f"Seed skipped: {existing} matches already in DB.")
        return 0
    if force:
        with SessionLocal() as session:
            session.execute(
                text(
                    "TRUNCATE TABLE features, predictions, matches, teams, seasons, leagues "
                    "RESTART IDENTITY CASCADE"
                )
            )
            session.commit()
        print("Existing data cleared.")

    load_summary, feature_summary = run_etl(
        data_dir, limit=limit, feature_version=feature_version
    )
    print_report(load_summary, feature_summary, feature_version)
    return load_summary.matches_created


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--feature-version", default="v1")
    args = parser.parse_args()
    limit = None if args.all else args.limit
    seed(
        limit=limit,
        data_dir=Path(args.data_dir),
        force=args.force,
        feature_version=args.feature_version,
    )


if __name__ == "__main__":
    main()
