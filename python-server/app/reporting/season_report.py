from __future__ import annotations

import pandas as pd

from sqlalchemy.orm import aliased

from app.db.models import Match, Team


def build_season_report(session, season_id: int) -> dict:
    home_team = aliased(Team)
    away_team = aliased(Team)
    matches = (
        session.query(Match, home_team, away_team)
        .join(home_team, Match.home_team_id == home_team.id)
        .join(away_team, Match.away_team_id == away_team.id)
        .filter(Match.season_id == season_id)
        .all()
    )

    if not matches:
        return {
            "season_id": season_id,
            "total_matches": 0,
            "avg_home_score": 0.0,
            "avg_away_score": 0.0,
            "top_home_teams": [],
            "top_away_teams": [],
        }

    df = pd.DataFrame(
        [
            {
                "home_team": home.name,
                "away_team": away.name,
                "home_score": match.home_goals,
                "away_score": match.away_goals,
            }
            for match, home, away in matches
        ]
    )

    df = df.dropna(subset=["home_score", "away_score"]).copy()
    avg_home = float(df["home_score"].mean()) if not df.empty else 0.0
    avg_away = float(df["away_score"].mean()) if not df.empty else 0.0

    top_home = (
        df.groupby("home_team")["home_score"].mean().sort_values(ascending=False).head(5)
    )
    top_away = (
        df.groupby("away_team")["away_score"].mean().sort_values(ascending=False).head(5)
    )

    return {
        "season_id": season_id,
        "total_matches": len(matches),
        "avg_home_score": avg_home,
        "avg_away_score": avg_away,
        "top_home_teams": top_home.index.tolist(),
        "top_away_teams": top_away.index.tolist(),
    }
