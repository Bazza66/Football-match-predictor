from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from app.config import DATA_DIR
from app.db.models import League, Match, Season, Team
from app.db.session import SessionLocal
from app.etl.feature_engineering import FeatureSummary, generate_features_for_all_seasons

ALIAS_MAP = {
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "manchester utd": "Manchester United",
    "psg": "Paris Saint-Germain",
    "spurs": "Tottenham Hotspur",
}


@dataclass
class LoadSummary:
    matches_processed: int
    matches_created: int
    missing_date: int
    missing_team: int
    duplicate_matches: int


def parse_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_time(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError:
        return None


def normalize_text(value: str | None) -> str | None:
    if not value:
        return None
    normalized = unicodedata.normalize("NFKC", value)
    normalized = " ".join(normalized.split())
    return normalized.strip() or None


def normalize_key(value: str) -> str:
    cleaned = normalize_text(value) or ""
    cleaned = cleaned.replace(".", "")
    return cleaned.casefold()


def canonical_team_name(value: str | None) -> str | None:
    normalized = normalize_text(value)
    if not normalized:
        return None
    key = normalize_key(normalized)
    return ALIAS_MAP.get(key, normalized)


def iter_match_rows(root_dir: Path) -> Iterable[dict]:
    for path in root_dir.rglob("*.json"):
        if path.name == "package.json":
            continue
        if path.parent == root_dir:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "matches" not in data:
            continue

        season_label = path.parent.name
        league_code = path.stem
        league_name = data.get("name", league_code)

        for match in data.get("matches", []):
            score = match.get("score", {})
            ft = score.get("ft") or [None, None]
            yield {
                "season": season_label,
                "league_code": league_code,
                "league_name": league_name,
                "round": match.get("round"),
                "date": parse_date(match.get("date")),
                "time": parse_time(match.get("time")),
                "home_team": match.get("team1"),
                "away_team": match.get("team2"),
                "home_goals": ft[0],
                "away_goals": ft[1],
            }


def get_or_create_league(session, cache: dict, code: str, name: str) -> League:
    league = cache.get(code)
    if league:
        return league
    league = session.query(League).filter_by(code=code).one_or_none()
    if league:
        cache[code] = league
        return league
    league = League(code=code, name=name)
    session.add(league)
    session.flush()
    cache[code] = league
    return league


def get_or_create_season(session, cache: dict, league_id: int, label: str) -> Season:
    key = (league_id, label)
    season = cache.get(key)
    if season:
        return season
    season = (
        session.query(Season)
        .filter_by(league_id=league_id, label=label)
        .one_or_none()
    )
    if season:
        cache[key] = season
        return season
    season = Season(league_id=league_id, label=label)
    session.add(season)
    session.flush()
    cache[key] = season
    return season


def get_or_create_team(session, cache: dict, league_id: int, name: str) -> Team:
    key = (league_id, normalize_key(name))
    team = cache.get(key)
    if team:
        return team
    team = (
        session.query(Team)
        .filter_by(league_id=league_id, name=name)
        .one_or_none()
    )
    if team:
        cache[key] = team
        return team
    team = Team(league_id=league_id, name=name)
    session.add(team)
    session.flush()
    cache[key] = team
    return team


def derive_result(home_goals: int | None, away_goals: int | None) -> str | None:
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def derive_status(match_date, home_goals: int | None, away_goals: int | None) -> str:
    if home_goals is not None and away_goals is not None:
        return "played"
    if match_date:
        return "scheduled"
    return "unknown"


def apply_match_update(match: Match, row: dict) -> None:
    match.round = row["round"] or match.round
    match.match_time = row["time"] or match.match_time
    if row["home_goals"] is not None:
        match.home_goals = row["home_goals"]
    if row["away_goals"] is not None:
        match.away_goals = row["away_goals"]
    match.result = derive_result(match.home_goals, match.away_goals)
    match.status = derive_status(match.match_date, match.home_goals, match.away_goals)


def upsert_match(
    session,
    season_id: int,
    home_team_id: int,
    away_team_id: int,
    row: dict,
    pending_matches: dict[tuple[int, date | None, int, int], Match],
):
    key = (season_id, row["date"], home_team_id, away_team_id)
    match = pending_matches.get(key)
    if match:
        apply_match_update(match, row)
        return match, False

    match = (
        session.query(Match)
        .filter_by(
            season_id=season_id,
            match_date=row["date"],
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )
        .one_or_none()
    )

    created = False
    if not match:
        match = Match(
            season_id=season_id,
            match_date=row["date"],
            match_time=row["time"],
            round=row["round"],
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=row["home_goals"],
            away_goals=row["away_goals"],
            result=derive_result(row["home_goals"], row["away_goals"]),
            status=derive_status(row["date"], row["home_goals"], row["away_goals"]),
        )
        session.add(match)
        created = True
    else:
        apply_match_update(match, row)

    pending_matches[key] = match
    return match, created


def load_to_db(data_dir: Path, limit: int | None = None) -> LoadSummary:
    league_cache: dict[str, League] = {}
    season_cache: dict[tuple[int, str], Season] = {}
    team_cache: dict[tuple[int, str], Team] = {}

    missing_date = 0
    missing_team = 0
    duplicate_matches = 0

    with SessionLocal() as session:
        processed = 0
        created_count = 0
        pending_matches: dict[tuple[int, date | None, int, int], Match] = {}
        for row in iter_match_rows(data_dir):
            if limit and processed >= limit:
                break
            home_name = canonical_team_name(row["home_team"])
            away_name = canonical_team_name(row["away_team"])
            if not home_name or not away_name:
                missing_team += 1
                continue
            if row["date"] is None:
                missing_date += 1
            league = get_or_create_league(session, league_cache, row["league_code"], row["league_name"])
            season = get_or_create_season(session, season_cache, league.id, row["season"])
            home_team = get_or_create_team(session, team_cache, league.id, home_name)
            away_team = get_or_create_team(session, team_cache, league.id, away_name)

            _, created = upsert_match(
                session,
                season.id,
                home_team.id,
                away_team.id,
                row,
                pending_matches,
            )
            if not created:
                duplicate_matches += 1
            else:
                created_count += 1
            processed += 1
            if processed % 1000 == 0:
                session.flush()
        session.commit()

    print(f"Loaded {created_count} matches from {processed} rows.")
    return LoadSummary(
        matches_processed=processed,
        matches_created=created_count,
        missing_date=missing_date,
        missing_team=missing_team,
        duplicate_matches=duplicate_matches,
    )


def run_etl(data_dir: Path, limit: int | None = None, feature_version: str = "v1") -> tuple[LoadSummary, FeatureSummary]:
    load_summary = load_to_db(data_dir, limit=limit)
    with SessionLocal() as session:
        feature_summary = generate_features_for_all_seasons(session, version=feature_version)
        session.commit()
    return load_summary, feature_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--feature-version", default="v1")
    args = parser.parse_args()
    if args.no_features:
        load_to_db(Path(args.data_dir), limit=args.limit)
        return
    run_etl(Path(args.data_dir), limit=args.limit, feature_version=args.feature_version)


if __name__ == "__main__":
    main()
