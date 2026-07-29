-- Replace league/season values as needed.
EXPLAIN (ANALYZE, BUFFERS)
SELECT m.id, m.match_date, m.home_team_id, m.away_team_id
FROM matches m
JOIN seasons s ON m.season_id = s.id
JOIN leagues l ON s.league_id = l.id
WHERE l.code = 'en.1'
  AND s.label = '2010-11'
ORDER BY m.match_date DESC
LIMIT 200;

EXPLAIN (ANALYZE, BUFFERS)
SELECT p.id, p.match_id, p.model_name, p.model_version, p.created_at
FROM predictions p
WHERE p.model_name = 'main_xgboost'
ORDER BY p.created_at DESC
LIMIT 50;
