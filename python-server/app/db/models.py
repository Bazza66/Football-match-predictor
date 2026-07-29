from __future__ import annotations

from datetime import date, datetime, time

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Time,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    seasons: Mapped[list["Season"]] = relationship(
        back_populates="league", cascade="all, delete-orphan"
    )
    teams: Mapped[list["Team"]] = relationship(
        back_populates="league", cascade="all, delete-orphan"
    )


class Season(Base):
    __tablename__ = "seasons"
    __table_args__ = (
        Index("ix_seasons_league_label", "league_id", "label"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), index=True)
    label: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    league: Mapped[League] = relationship(back_populates="seasons")
    matches: Mapped[list["Match"]] = relationship(
        back_populates="season", cascade="all, delete-orphan"
    )


class Team(Base):
    __tablename__ = "teams"
    __table_args__ = (
        UniqueConstraint("league_id", "name", name="uq_teams_league_name"),
        UniqueConstraint("league_id", "external_id", name="uq_teams_league_external_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), index=True)
    external_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    short_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    alt_names: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    country: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    league: Mapped[League] = relationship(back_populates="teams")
    home_matches: Mapped[list["Match"]] = relationship(
        back_populates="home_team", foreign_keys="Match.home_team_id"
    )
    away_matches: Mapped[list["Match"]] = relationship(
        back_populates="away_team", foreign_keys="Match.away_team_id"
    )


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint(
            "season_id",
            "match_date",
            "home_team_id",
            "away_team_id",
            name="uq_matches_season_date_teams",
        ),
        Index("ix_matches_season_date", "season_id", "match_date"),
        Index("ix_matches_date", "match_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id"), index=True)
    round: Mapped[str | None] = mapped_column(String(100), nullable=True)
    match_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    match_time: Mapped[time | None] = mapped_column(Time, nullable=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result: Mapped[str | None] = mapped_column(String(1), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="unknown")

    season: Mapped[Season] = relationship(back_populates="matches")
    home_team: Mapped[Team] = relationship(
        back_populates="home_matches", foreign_keys=[home_team_id]
    )
    away_team: Mapped[Team] = relationship(
        back_populates="away_matches", foreign_keys=[away_team_id]
    )
    features: Mapped[list["Feature"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )


class Feature(Base):
    __tablename__ = "features"
    __table_args__ = (
        UniqueConstraint("match_id", "version", name="uq_features_match_version"),
    )

    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id"), primary_key=True
    )
    version: Mapped[str] = mapped_column(String(20), primary_key=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    is_valid: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    notes: Mapped[str | None] = mapped_column(String(200), nullable=True)

    home_points_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_points_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_points_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_points_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_wins_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_draws_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_losses_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_wins_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_draws_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_losses_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_wins_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_draws_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_losses_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_wins_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_draws_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_losses_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_goals_for_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_goals_against_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_goal_diff_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals_for_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals_against_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goal_diff_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_goals_for_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_goals_against_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_goal_diff_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals_for_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals_against_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goal_diff_last_10: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_points_last_5_home_games: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    away_points_last_5_away_games: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    home_goals_for_home_last_5: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    away_goals_for_away_last_5: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    points_season_so_far_home: Mapped[int | None] = mapped_column(Integer, nullable=True)
    points_season_so_far_away: Mapped[int | None] = mapped_column(Integer, nullable=True)
    goal_diff_season_so_far_home: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    goal_diff_season_so_far_away: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    h2h_points_last_3: Mapped[int | None] = mapped_column(Integer, nullable=True)
    h2h_goal_diff_last_3: Mapped[int | None] = mapped_column(Integer, nullable=True)

    days_since_last_match_home: Mapped[int | None] = mapped_column(Integer, nullable=True)
    days_since_last_match_away: Mapped[int | None] = mapped_column(Integer, nullable=True)

    diff_points_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diff_goal_diff_last_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diff_days_rest: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_goals_for_rate_last_5: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    home_goals_against_rate_last_5: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    away_goals_for_rate_last_5: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    away_goals_against_rate_last_5: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    home_goals_for_rate_last_10: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    home_goals_against_rate_last_10: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    away_goals_for_rate_last_10: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    away_goals_against_rate_last_10: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    home_elo_overall: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_elo_overall: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_elo_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_elo_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    elo_diff_overall: Mapped[float | None] = mapped_column(Float, nullable=True)
    elo_diff_home_away: Mapped[float | None] = mapped_column(Float, nullable=True)

    home_form_ema_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_form_ema_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_form_ema_goal_diff: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    away_form_ema_goal_diff: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    diff_form_ema_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    diff_form_ema_goal_diff: Mapped[float | None] = mapped_column(Float, nullable=True)

    home_win_streak: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_win_streak: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_draw_streak: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_draw_streak: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_loss_streak: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_loss_streak: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_unbeaten_streak: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_unbeaten_streak: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_home_streak: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_away_streak: Mapped[int | None] = mapped_column(Integer, nullable=True)

    match: Mapped[Match] = relationship(back_populates="features")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "match_id", "model_name", "model_version", name="uq_predictions_match_model"
        ),
        Index("ix_predictions_model_created", "model_name", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    model_version: Mapped[str] = mapped_column(String(100), index=True)
    predicted_home_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_away_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_prob_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_prob_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_prob_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    match: Mapped[Match] = relationship(back_populates="predictions")


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    model_version: Mapped[str] = mapped_column(String(100), index=True)
    feature_version: Mapped[str] = mapped_column(String(20), index=True)
    commit_hash: Mapped[str | None] = mapped_column(String(40), nullable=True)
    league_code: Mapped[str | None] = mapped_column(String(50), index=True, nullable=True)
    season_label: Mapped[str | None] = mapped_column(String(20), index=True, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="completed")
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    data_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    artifacts: Mapped[list["ModelArtifact"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_run_id: Mapped[int] = mapped_column(
        ForeignKey("model_runs.id"), index=True
    )
    artifact_type: Mapped[str] = mapped_column(String(50), index=True)
    path: Mapped[str] = mapped_column(String(500))
    checksum: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    run: Mapped[ModelRun] = relationship(back_populates="artifacts")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
