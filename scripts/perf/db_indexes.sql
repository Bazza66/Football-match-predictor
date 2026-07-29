-- Add supporting indexes for frequent filters and ordering.
CREATE INDEX IF NOT EXISTS ix_seasons_league_label ON seasons (league_id, label);
CREATE INDEX IF NOT EXISTS ix_matches_date ON matches (match_date);
CREATE INDEX IF NOT EXISTS ix_predictions_model_created ON predictions (model_name, created_at DESC);
