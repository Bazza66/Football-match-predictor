from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from sqlalchemy.orm import Session

from app.db.models import Feature, League, Match, Season
from app.etl.feature_engineering import build_record, summarize_h2h, summarize_last


@dataclass
class LeakageCheckResult:
    total_matches: int
    checked_matches: int
    mismatches: int
    date_order_violations: int
    negative_rest_days: int
    missing_date: int
    sample_mismatches: list[dict]


CHECK_COLUMNS = [
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
]


def run_leakage_checks(
    session: Session,
    *,
    version: str,
    league_code: str | None = None,
    season_label: str | None = None,
    sample_limit: int = 5,
) -> LeakageCheckResult:
    query = (
        session.query(Match, Feature, Season)
        .join(Feature, Feature.match_id == Match.id)
        .join(Season, Match.season_id == Season.id)
        .join(League, Season.league_id == League.id)
        .filter(Feature.version == version)
        .order_by(Season.id.asc(), Match.match_date.asc().nullslast(), Match.id.asc())
    )
    if league_code:
        query = query.filter(League.code == league_code)
    if season_label:
        query = query.filter(Season.label == season_label)

    rows = query.all()
    total_matches = len(rows)
    missing_date = 0
    matches_by_season: dict[int, list[tuple[Match, Feature]]] = defaultdict(list)
    for match, feature, season in rows:
        if not match.match_date:
            missing_date += 1
            continue
        matches_by_season[season.id].append((match, feature))

    checked_matches = 0
    mismatches = 0
    date_order_violations = 0
    negative_rest_days = 0
    sample_mismatches: list[dict] = []

    for season_id, season_rows in matches_by_season.items():
        matches_by_date: dict[date, list[tuple[Match, Feature]]] = defaultdict(list)
        for match, feature in season_rows:
            matches_by_date[match.match_date].append((match, feature))

        team_history: dict[int, list[dict]] = defaultdict(list)
        team_home_history: dict[int, list[dict]] = defaultdict(list)
        team_away_history: dict[int, list[dict]] = defaultdict(list)
        team_last_date: dict[int, date] = {}
        season_points: dict[int, int] = defaultdict(int)
        season_goal_diff: dict[int, int] = defaultdict(int)
        pair_history: dict[tuple[int, int], list[dict]] = defaultdict(list)
        history_latest_date: date | None = None

        for match_date in sorted(matches_by_date.keys()):
            if history_latest_date and history_latest_date >= match_date:
                date_order_violations += 1
            todays_matches = matches_by_date[match_date]
            for match, feature in todays_matches:
                checked_matches += 1
                home_id = match.home_team_id
                away_id = match.away_team_id

                home_last_5 = summarize_last(team_history[home_id], 5)
                home_last_10 = summarize_last(team_history[home_id], 10)
                away_last_5 = summarize_last(team_history[away_id], 5)
                away_last_10 = summarize_last(team_history[away_id], 10)

                home_home_last_5 = summarize_last(team_home_history[home_id], 5)
                away_away_last_5 = summarize_last(team_away_history[away_id], 5)

                h2h_key = tuple(sorted((home_id, away_id)))
                h2h_points, h2h_goal_diff = summarize_h2h(
                    pair_history[h2h_key], home_id
                )

                last_home_date = team_last_date.get(home_id)
                last_away_date = team_last_date.get(away_id)
                days_since_home = (
                    (match_date - last_home_date).days if last_home_date else None
                )
                days_since_away = (
                    (match_date - last_away_date).days if last_away_date else None
                )
                diff_days_rest = (
                    days_since_home - days_since_away
                    if days_since_home is not None and days_since_away is not None
                    else None
                )

                if days_since_home is not None and days_since_home < 0:
                    negative_rest_days += 1
                if days_since_away is not None and days_since_away < 0:
                    negative_rest_days += 1

                expected = {
                    "home_points_last_5": home_last_5["points"],
                    "home_points_last_10": home_last_10["points"],
                    "away_points_last_5": away_last_5["points"],
                    "away_points_last_10": away_last_10["points"],
                    "home_wins_last_5": home_last_5["wins"],
                    "home_draws_last_5": home_last_5["draws"],
                    "home_losses_last_5": home_last_5["losses"],
                    "away_wins_last_5": away_last_5["wins"],
                    "away_draws_last_5": away_last_5["draws"],
                    "away_losses_last_5": away_last_5["losses"],
                    "home_wins_last_10": home_last_10["wins"],
                    "home_draws_last_10": home_last_10["draws"],
                    "home_losses_last_10": home_last_10["losses"],
                    "away_wins_last_10": away_last_10["wins"],
                    "away_draws_last_10": away_last_10["draws"],
                    "away_losses_last_10": away_last_10["losses"],
                    "home_goals_for_last_5": home_last_5["goals_for"],
                    "home_goals_against_last_5": home_last_5["goals_against"],
                    "home_goal_diff_last_5": home_last_5["goal_diff"],
                    "away_goals_for_last_5": away_last_5["goals_for"],
                    "away_goals_against_last_5": away_last_5["goals_against"],
                    "away_goal_diff_last_5": away_last_5["goal_diff"],
                    "home_goals_for_last_10": home_last_10["goals_for"],
                    "home_goals_against_last_10": home_last_10["goals_against"],
                    "home_goal_diff_last_10": home_last_10["goal_diff"],
                    "away_goals_for_last_10": away_last_10["goals_for"],
                    "away_goals_against_last_10": away_last_10["goals_against"],
                    "away_goal_diff_last_10": away_last_10["goal_diff"],
                    "home_points_last_5_home_games": home_home_last_5["points"],
                    "away_points_last_5_away_games": away_away_last_5["points"],
                    "home_goals_for_home_last_5": home_home_last_5["goals_for"],
                    "away_goals_for_away_last_5": away_away_last_5["goals_for"],
                    "points_season_so_far_home": season_points[home_id],
                    "points_season_so_far_away": season_points[away_id],
                    "goal_diff_season_so_far_home": season_goal_diff[home_id],
                    "goal_diff_season_so_far_away": season_goal_diff[away_id],
                    "h2h_points_last_3": h2h_points,
                    "h2h_goal_diff_last_3": h2h_goal_diff,
                    "days_since_last_match_home": days_since_home,
                    "days_since_last_match_away": days_since_away,
                    "diff_points_last_5": home_last_5["points"] - away_last_5["points"],
                    "diff_goal_diff_last_5": home_last_5["goal_diff"]
                    - away_last_5["goal_diff"],
                    "diff_days_rest": diff_days_rest,
                }

                mismatch_fields = []
                for col in CHECK_COLUMNS:
                    if getattr(feature, col) != expected[col]:
                        mismatch_fields.append(col)
                if mismatch_fields:
                    mismatches += 1
                    if len(sample_mismatches) < sample_limit:
                        sample_mismatches.append(
                            {
                                "match_id": match.id,
                                "match_date": str(match.match_date),
                                "mismatch_fields": mismatch_fields,
                                "season_id": season_id,
                            }
                        )

            for match, _feature in todays_matches:
                if match.home_goals is None or match.away_goals is None:
                    continue
                home_record = build_record(match.home_goals, match.away_goals)
                away_record = build_record(match.away_goals, match.home_goals)

                team_history[match.home_team_id].append(home_record)
                team_history[match.away_team_id].append(away_record)
                team_home_history[match.home_team_id].append(home_record)
                team_away_history[match.away_team_id].append(away_record)

                season_points[match.home_team_id] += home_record["points"]
                season_points[match.away_team_id] += away_record["points"]
                season_goal_diff[match.home_team_id] += (
                    home_record["goals_for"] - home_record["goals_against"]
                )
                season_goal_diff[match.away_team_id] += (
                    away_record["goals_for"] - away_record["goals_against"]
                )

                team_last_date[match.home_team_id] = match_date
                team_last_date[match.away_team_id] = match_date

                pair_key = tuple(sorted((match.home_team_id, match.away_team_id)))
                pair_history[pair_key].append(
                    {
                        "home_team_id": match.home_team_id,
                        "away_team_id": match.away_team_id,
                        "home_goals": match.home_goals,
                        "away_goals": match.away_goals,
                    }
                )

            history_latest_date = match_date

    return LeakageCheckResult(
        total_matches=total_matches,
        checked_matches=checked_matches,
        mismatches=mismatches,
        date_order_violations=date_order_violations,
        negative_rest_days=negative_rest_days,
        missing_date=missing_date,
        sample_mismatches=sample_mismatches,
    )
