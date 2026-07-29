from __future__ import annotations

from datetime import date, datetime, time

from pydantic import BaseModel, Field


class LeagueOut(BaseModel):
    id: int
    code: str
    name: str


class SeasonOut(BaseModel):
    id: int
    league_id: int
    label: str


class MatchOut(BaseModel):
    id: int
    season_id: int
    round: str | None
    date: date | None
    time: time | None
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None


class PredictionRequest(BaseModel):
    match_id: int | None = Field(default=None, description="Existing match id")
    home_team: str | None = None
    away_team: str | None = None
    league: str | None = None
    season: str | None = None


class PredictionOut(BaseModel):
    match_id: int | None
    model_name: str
    model_version: str
    predicted_home_score: float | None
    predicted_away_score: float | None
    win_prob_home: float | None = None
    win_prob_draw: float | None = None
    win_prob_away: float | None = None


class ModelArtifactOut(BaseModel):
    artifact_type: str
    path: str
    checksum: str
    size_bytes: int


class ModelRunOut(BaseModel):
    run_id: str
    model_name: str
    model_version: str
    feature_version: str
    commit_hash: str | None
    league_code: str | None
    season_label: str | None
    status: str
    metrics: dict | None = None
    params: dict | None = None
    data_summary: dict | None = None
    created_at: datetime
    artifacts: list[ModelArtifactOut] | None = None


class ModelRunSummaryOut(BaseModel):
    run_id: str
    model_name: str
    model_version: str
    feature_version: str
    league_code: str | None
    season_label: str | None
    metric: float | None = None
    metric_name: str | None = None
    created_at: datetime


class ModelRunBestOut(BaseModel):
    run_id: str | None = None
    model_name: str
    model_version: str
    feature_version: str
    league_code: str | None
    season_label: str | None
    metric_name: str | None = None
    metric_value: float | None = None
    direction: str | None = None
    updated_at: datetime | None = None


class DatasetIntegrityReportOut(BaseModel):
    report_id: str
    created_at: datetime
    summary_url: str
    missingness_url: str | None = None
    golden_sample_url: str | None = None


class LabelAuditReportOut(BaseModel):
    report_id: str
    run_id: str
    created_at: datetime
    report_url: str
    chart_url: str | None = None
    confusion_url: str | None = None
    league_code: str | None = None
    season_label: str | None = None


class TrainingDivergenceReportOut(BaseModel):
    report_id: str
    run_id: str
    created_at: datetime
    report_url: str
    loss_chart_url: str | None = None
    grad_chart_url: str | None = None
    league_code: str | None = None
    season_label: str | None = None


class ModelEvalReportOut(BaseModel):
    report_id: str
    run_id: str | None = None
    created_at: datetime
    status: str
    report_url: str | None = None
    metrics_json_url: str | None = None
    metrics_csv_url: str | None = None
    confusion_url: str | None = None
    confusion_baseline_url: str | None = None
    calibration_url: str | None = None
    entropy_hist_url: str | None = None
    max_prob_hist_url: str | None = None
    subsets_url: str | None = None
    league_code: str | None = None
    season_label: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    baseline_name: str | None = None
    baseline_version: str | None = None
    feature_version: str | None = None
    error: str | None = None


class TuningSweepReportOut(BaseModel):
    report_id: str
    created_at: datetime
    status: str
    started_at: datetime | None = None
    updated_at: datetime | None = None
    percent: float | None = None
    total_iterations: int | None = None
    completed_iterations: int | None = None
    current_league: str | None = None
    current_iteration: int | None = None
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    errors_count: int | None = None
    trainer: str | None = None
    model: str | None = None
    feature_version: str | None = None
    season_label: str | None = None
    league_codes: list[str] | None = None
    n_iter: int | None = None
    seed: int | None = None
    train_ratio: float | None = None
    val_ratio: float | None = None
    class_weight: str | None = None
    calibration: bool | None = None
    calibration_method: str | None = None
    n_estimators: int | None = None
    iterations: int | None = None
    fast_mode: bool | None = None
    results_csv_url: str | None = None
    results_tsv_url: str | None = None
    results_md_url: str | None = None
    summary_by_league_url: str | None = None
    best_by_league_url: str | None = None
    metadata_url: str | None = None
    errors_url: str | None = None
    error: str | None = None


class ReportOut(BaseModel):
    season_id: int
    total_matches: int
    avg_home_score: float
    avg_away_score: float
    top_home_teams: list[str]
    top_away_teams: list[str]
