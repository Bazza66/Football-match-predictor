from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from math import pow

from sqlalchemy.orm import Session

from app.db.models import Feature, Match, Season

ELO_BASE = 1500.0
ELO_K = 20.0
EMA_ALPHA = 0.2


@dataclass
class FeatureSummary:
    total_matches: int
    eligible_matches: int
    features_written: int
    missing_date: int
    leakage_warnings: int


def build_record(goals_for: int, goals_against: int) -> dict:
    if goals_for > goals_against:
        points = 3
        win = 1
        draw = 0
        loss = 0
    elif goals_for < goals_against:
        points = 0
        win = 0
        draw = 0
        loss = 1
    else:
        points = 1
        win = 0
        draw = 1
        loss = 0
    return {
        "points": points,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "win": win,
        "draw": draw,
        "loss": loss,
    }


def summarize_last(records: list[dict], window: int) -> dict:
    if not records:
        return {
            "points": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
        }
    last = records[-window:] if window else records
    points = sum(r["points"] for r in last)
    wins = sum(r["win"] for r in last)
    draws = sum(r["draw"] for r in last)
    losses = sum(r["loss"] for r in last)
    goals_for = sum(r["goals_for"] for r in last)
    goals_against = sum(r["goals_against"] for r in last)
    return {
        "points": points,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_diff": goals_for - goals_against,
    }


def summarize_h2h(records: list[dict], home_team_id: int) -> tuple[int, int]:
    if not records:
        return 0, 0
    last = records[-3:]
    points = 0
    goal_diff = 0
    for record in last:
        if record["home_team_id"] == home_team_id:
            goals_for = record["home_goals"]
            goals_against = record["away_goals"]
        else:
            goals_for = record["away_goals"]
            goals_against = record["home_goals"]
        if goals_for > goals_against:
            points += 3
        elif goals_for == goals_against:
            points += 1
        goal_diff += goals_for - goals_against
    return points, goal_diff


def rate_from_history(records: list[dict], window: int, key: str) -> float:
    if not records:
        return 0.0
    last = records[-window:] if window else records
    denominator = len(last) or 1
    total = sum(r[key] for r in last)
    return float(total / denominator)


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float) -> tuple[float, float]:
    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1.0 - exp_a
    new_a = rating_a + ELO_K * (score_a - exp_a)
    new_b = rating_b + ELO_K * ((1.0 - score_a) - exp_b)
    return new_a, new_b


def generate_features_for_all_seasons(session: Session, version: str = "v1") -> FeatureSummary:
    total_matches = 0
    eligible_matches = 0
    features_written = 0
    missing_date = 0
    leakage_warnings = 0

    seasons = session.query(Season).all()
    for season in seasons:
        season_summary = generate_features_for_season(session, season, version=version)
        total_matches += season_summary.total_matches
        eligible_matches += season_summary.eligible_matches
        features_written += season_summary.features_written
        missing_date += season_summary.missing_date
        leakage_warnings += season_summary.leakage_warnings

    return FeatureSummary(
        total_matches=total_matches,
        eligible_matches=eligible_matches,
        features_written=features_written,
        missing_date=missing_date,
        leakage_warnings=leakage_warnings,
    )


def generate_features_for_season(session: Session, season: Season, version: str = "v1") -> FeatureSummary:
    matches = (
        session.query(Match)
        .filter(Match.season_id == season.id)
        .order_by(Match.match_date.asc().nullslast(), Match.id.asc())
        .all()
    )

    total_matches = len(matches)
    matches_by_date: dict[date, list[Match]] = defaultdict(list)
    missing_date = 0
    for match in matches:
        if not match.match_date:
            missing_date += 1
            continue
        matches_by_date[match.match_date].append(match)

    team_history: dict[int, list[dict]] = defaultdict(list)
    team_home_history: dict[int, list[dict]] = defaultdict(list)
    team_away_history: dict[int, list[dict]] = defaultdict(list)
    team_last_date: dict[int, date] = {}
    season_points: dict[int, int] = defaultdict(int)
    season_goal_diff: dict[int, int] = defaultdict(int)
    pair_history: dict[tuple[int, int], list[dict]] = defaultdict(list)
    team_elo_overall: dict[int, float] = defaultdict(lambda: ELO_BASE)
    team_elo_home: dict[int, float] = defaultdict(lambda: ELO_BASE)
    team_elo_away: dict[int, float] = defaultdict(lambda: ELO_BASE)
    team_form_ema_points: dict[int, float] = defaultdict(float)
    team_form_ema_goal_diff: dict[int, float] = defaultdict(float)
    team_win_streak: dict[int, int] = defaultdict(int)
    team_draw_streak: dict[int, int] = defaultdict(int)
    team_loss_streak: dict[int, int] = defaultdict(int)
    team_unbeaten_streak: dict[int, int] = defaultdict(int)
    team_home_streak: dict[int, int] = defaultdict(int)
    team_away_streak: dict[int, int] = defaultdict(int)

    features_written = 0
    eligible_matches = 0
    leakage_warnings = 0
    history_latest_date: date | None = None

    for match_date in sorted(matches_by_date.keys()):
        if history_latest_date and history_latest_date >= match_date:
            leakage_warnings += 1
        todays_matches = matches_by_date[match_date]

        for match in todays_matches:
            eligible_matches += 1
            home_id = match.home_team_id
            away_id = match.away_team_id

            home_last_5 = summarize_last(team_history[home_id], 5)
            home_last_10 = summarize_last(team_history[home_id], 10)
            away_last_5 = summarize_last(team_history[away_id], 5)
            away_last_10 = summarize_last(team_history[away_id], 10)

            home_home_last_5 = summarize_last(team_home_history[home_id], 5)
            away_away_last_5 = summarize_last(team_away_history[away_id], 5)

            h2h_key = tuple(sorted((home_id, away_id)))
            h2h_points, h2h_goal_diff = summarize_h2h(pair_history[h2h_key], home_id)

            last_home_date = team_last_date.get(home_id)
            last_away_date = team_last_date.get(away_id)
            days_since_home = (match_date - last_home_date).days if last_home_date else None
            days_since_away = (match_date - last_away_date).days if last_away_date else None

            diff_days_rest = (
                days_since_home - days_since_away
                if days_since_home is not None and days_since_away is not None
                else None
            )

            home_goals_for_rate_last_5 = rate_from_history(
                team_history[home_id], 5, "goals_for"
            )
            home_goals_against_rate_last_5 = rate_from_history(
                team_history[home_id], 5, "goals_against"
            )
            away_goals_for_rate_last_5 = rate_from_history(
                team_history[away_id], 5, "goals_for"
            )
            away_goals_against_rate_last_5 = rate_from_history(
                team_history[away_id], 5, "goals_against"
            )
            home_goals_for_rate_last_10 = rate_from_history(
                team_history[home_id], 10, "goals_for"
            )
            home_goals_against_rate_last_10 = rate_from_history(
                team_history[home_id], 10, "goals_against"
            )
            away_goals_for_rate_last_10 = rate_from_history(
                team_history[away_id], 10, "goals_for"
            )
            away_goals_against_rate_last_10 = rate_from_history(
                team_history[away_id], 10, "goals_against"
            )

            home_elo_overall = team_elo_overall[home_id]
            away_elo_overall = team_elo_overall[away_id]
            home_elo_home = team_elo_home[home_id]
            away_elo_away = team_elo_away[away_id]
            elo_diff_overall = home_elo_overall - away_elo_overall
            elo_diff_home_away = home_elo_home - away_elo_away

            home_form_ema_points = team_form_ema_points[home_id]
            away_form_ema_points = team_form_ema_points[away_id]
            home_form_ema_goal_diff = team_form_ema_goal_diff[home_id]
            away_form_ema_goal_diff = team_form_ema_goal_diff[away_id]

            feature = Feature(
                match_id=match.id,
                version=version,
                home_points_last_5=home_last_5["points"],
                home_points_last_10=home_last_10["points"],
                away_points_last_5=away_last_5["points"],
                away_points_last_10=away_last_10["points"],
                home_wins_last_5=home_last_5["wins"],
                home_draws_last_5=home_last_5["draws"],
                home_losses_last_5=home_last_5["losses"],
                away_wins_last_5=away_last_5["wins"],
                away_draws_last_5=away_last_5["draws"],
                away_losses_last_5=away_last_5["losses"],
                home_wins_last_10=home_last_10["wins"],
                home_draws_last_10=home_last_10["draws"],
                home_losses_last_10=home_last_10["losses"],
                away_wins_last_10=away_last_10["wins"],
                away_draws_last_10=away_last_10["draws"],
                away_losses_last_10=away_last_10["losses"],
                home_goals_for_last_5=home_last_5["goals_for"],
                home_goals_against_last_5=home_last_5["goals_against"],
                home_goal_diff_last_5=home_last_5["goal_diff"],
                away_goals_for_last_5=away_last_5["goals_for"],
                away_goals_against_last_5=away_last_5["goals_against"],
                away_goal_diff_last_5=away_last_5["goal_diff"],
                home_goals_for_last_10=home_last_10["goals_for"],
                home_goals_against_last_10=home_last_10["goals_against"],
                home_goal_diff_last_10=home_last_10["goal_diff"],
                away_goals_for_last_10=away_last_10["goals_for"],
                away_goals_against_last_10=away_last_10["goals_against"],
                away_goal_diff_last_10=away_last_10["goal_diff"],
                home_points_last_5_home_games=home_home_last_5["points"],
                away_points_last_5_away_games=away_away_last_5["points"],
                home_goals_for_home_last_5=home_home_last_5["goals_for"],
                away_goals_for_away_last_5=away_away_last_5["goals_for"],
                points_season_so_far_home=season_points[home_id],
                points_season_so_far_away=season_points[away_id],
                goal_diff_season_so_far_home=season_goal_diff[home_id],
                goal_diff_season_so_far_away=season_goal_diff[away_id],
                h2h_points_last_3=h2h_points,
                h2h_goal_diff_last_3=h2h_goal_diff,
                days_since_last_match_home=days_since_home,
                days_since_last_match_away=days_since_away,
                diff_points_last_5=home_last_5["points"] - away_last_5["points"],
                diff_goal_diff_last_5=home_last_5["goal_diff"] - away_last_5["goal_diff"],
                diff_days_rest=diff_days_rest,
                home_goals_for_rate_last_5=home_goals_for_rate_last_5,
                home_goals_against_rate_last_5=home_goals_against_rate_last_5,
                away_goals_for_rate_last_5=away_goals_for_rate_last_5,
                away_goals_against_rate_last_5=away_goals_against_rate_last_5,
                home_goals_for_rate_last_10=home_goals_for_rate_last_10,
                home_goals_against_rate_last_10=home_goals_against_rate_last_10,
                away_goals_for_rate_last_10=away_goals_for_rate_last_10,
                away_goals_against_rate_last_10=away_goals_against_rate_last_10,
                home_elo_overall=home_elo_overall,
                away_elo_overall=away_elo_overall,
                home_elo_home=home_elo_home,
                away_elo_away=away_elo_away,
                elo_diff_overall=elo_diff_overall,
                elo_diff_home_away=elo_diff_home_away,
                home_form_ema_points=home_form_ema_points,
                away_form_ema_points=away_form_ema_points,
                home_form_ema_goal_diff=home_form_ema_goal_diff,
                away_form_ema_goal_diff=away_form_ema_goal_diff,
                diff_form_ema_points=home_form_ema_points - away_form_ema_points,
                diff_form_ema_goal_diff=home_form_ema_goal_diff
                - away_form_ema_goal_diff,
                home_win_streak=team_win_streak[home_id],
                away_win_streak=team_win_streak[away_id],
                home_draw_streak=team_draw_streak[home_id],
                away_draw_streak=team_draw_streak[away_id],
                home_loss_streak=team_loss_streak[home_id],
                away_loss_streak=team_loss_streak[away_id],
                home_unbeaten_streak=team_unbeaten_streak[home_id],
                away_unbeaten_streak=team_unbeaten_streak[away_id],
                home_home_streak=team_home_streak[home_id],
                away_away_streak=team_away_streak[away_id],
            )

            session.merge(feature)
            features_written += 1

        for match in todays_matches:
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

            home_id = match.home_team_id
            away_id = match.away_team_id
            score_home = 0.5
            if home_record["win"]:
                score_home = 1.0
            elif home_record["loss"]:
                score_home = 0.0

            team_elo_overall[home_id], team_elo_overall[away_id] = update_elo(
                team_elo_overall[home_id], team_elo_overall[away_id], score_home
            )
            team_elo_home[home_id], team_elo_away[away_id] = update_elo(
                team_elo_home[home_id], team_elo_away[away_id], score_home
            )

            team_form_ema_points[home_id] = (
                (1 - EMA_ALPHA) * team_form_ema_points[home_id]
                + EMA_ALPHA * home_record["points"]
            )
            team_form_ema_points[away_id] = (
                (1 - EMA_ALPHA) * team_form_ema_points[away_id]
                + EMA_ALPHA * away_record["points"]
            )
            team_form_ema_goal_diff[home_id] = (
                (1 - EMA_ALPHA) * team_form_ema_goal_diff[home_id]
                + EMA_ALPHA * (home_record["goals_for"] - home_record["goals_against"])
            )
            team_form_ema_goal_diff[away_id] = (
                (1 - EMA_ALPHA) * team_form_ema_goal_diff[away_id]
                + EMA_ALPHA * (away_record["goals_for"] - away_record["goals_against"])
            )

            if home_record["win"]:
                team_win_streak[home_id] += 1
                team_draw_streak[home_id] = 0
                team_loss_streak[home_id] = 0
                team_unbeaten_streak[home_id] += 1
            elif home_record["draw"]:
                team_draw_streak[home_id] += 1
                team_win_streak[home_id] = 0
                team_loss_streak[home_id] = 0
                team_unbeaten_streak[home_id] += 1
            else:
                team_loss_streak[home_id] += 1
                team_win_streak[home_id] = 0
                team_draw_streak[home_id] = 0
                team_unbeaten_streak[home_id] = 0

            if away_record["win"]:
                team_win_streak[away_id] += 1
                team_draw_streak[away_id] = 0
                team_loss_streak[away_id] = 0
                team_unbeaten_streak[away_id] += 1
            elif away_record["draw"]:
                team_draw_streak[away_id] += 1
                team_win_streak[away_id] = 0
                team_loss_streak[away_id] = 0
                team_unbeaten_streak[away_id] += 1
            else:
                team_loss_streak[away_id] += 1
                team_win_streak[away_id] = 0
                team_draw_streak[away_id] = 0
                team_unbeaten_streak[away_id] = 0

            team_home_streak[home_id] += 1
            team_away_streak[home_id] = 0
            team_away_streak[away_id] += 1
            team_home_streak[away_id] = 0

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

    return FeatureSummary(
        total_matches=total_matches,
        eligible_matches=eligible_matches,
        features_written=features_written,
        missing_date=missing_date,
        leakage_warnings=leakage_warnings,
    )
