
## One-command start

```powershell
docker compose up --build
```

- API: http://localhost:8000
- UI: http://localhost:5173


## Seed sample data

```powershell
docker compose exec api python -m app.etl.seed --limit 1000
```


```powershell
docker compose exec api python -m app.etl.seed --limit 1000 --force
```

To load the full dataset:

```powershell
docker compose exec api python -m app.etl.seed --all --force
```

Feature version can be set explicitly:

```powershell
docker compose exec api python -m app.etl.seed --all --force --feature-version v1
```

Note: If you pull new feature columns, you must reset the database (drop the
volume) so the schema is recreated:

```powershell
docker compose down -v
docker compose up -d
```

## Training + tracking workflow

Train scripts do a temporal split (train/val/test), compute metrics (including baselines and per-class diagnostics), run leakage checks, and write a run record to DB with artifact links and checksums.

Summary:
- Features are built by ETL (`app.etl.seed`) and stored in DB with `feature_version` (e.g. `v1`).
- Training uses time-based split and logs `accuracy`, `macro-F1`, `logloss`, `brier`, and confusion matrix.
- Calibration is optional and uses the validation split.
- Leakage checks are enforced by default; use `--allow-leakage` only to override for debugging.
- Artifacts saved to `python-server/models`: model, meta, schema.

## Train baseline model

```powershell
docker compose exec api python -m app.ml.train_baseline --version v1
```

With tuning + calibration + league/season filters:

```powershell
docker compose exec api python -m app.ml.train_baseline --version v1 --league en.1 --season 2010-11 --tune --n-iter 12 --calibrate
```

With class weighting + CV:

```powershell
docker compose exec api python -m app.ml.train_baseline --version v1 --class-weight balanced --cv-splits 5
```

## Train main model (xgboost)

```powershell
docker compose exec api python -m app.ml.train_main --version v1 --model xgboost
```

With tuning + walk-forward + calibration:

```powershell
docker compose exec api python -m app.ml.train_main --version v1 --model xgboost --tune --walk-forward --n-iter 12 --calibrate
```

With class weighting + CV + verbose eval:

```powershell
docker compose exec api python -m app.ml.train_main --version v1 --model xgboost --class-weight balanced --cv-splits 5 --verbose-eval 50
```

Notes:
- `--tune` runs a random search over a broader grid and picks the best logloss.
- `--walk-forward` uses time-series CV during tuning (falls back to holdout if data is small).
- `--calibrate/--no-calibrate` controls validation-based calibration.
- `--class-weight balanced` applies inverse-frequency sample weighting.
- `--cv-splits N` reports rolling time-series CV summary metrics.
- `--early-stopping-rounds N` enables early stopping when supported by your XGBoost build.
- `--n-estimators N` overrides the number of boosting rounds (XGBoost only).
- Each training run writes a label audit report under `reports/label_audit_<run_id>/`.
- Each training run writes a divergence report under `reports/training_divergence_<run_id>/`.

## Train per-league (xgboost)

No calibration (avoids sklearn `cv="prefit"` issue) with verbose evaluation:

```powershell
docker compose exec -e PYTHONUNBUFFERED=1 api python -u -m app.ml.train_per_league --trainer main --model xgboost --continue-on-error --verbose-eval 1 --no-calibrate
```

Explicit 5 boosting rounds:

```powershell
docker compose exec -e PYTHONUNBUFFERED=1 api python -u -m app.ml.train_per_league --trainer main --model xgboost --continue-on-error --verbose-eval 1 --no-calibrate --n-estimators 5
```

Override boosting rounds and enable early stopping (if supported by XGBoost):

```powershell
docker compose exec -e PYTHONUNBUFFERED=1 api python -u -m app.ml.train_per_league --trainer main --model xgboost --continue-on-error --verbose-eval 1 --no-calibrate --n-estimators 50 --early-stopping-rounds 10
```

## Write predictions to DB

```powershell
docker compose exec api python -m app.ml.predict_to_db --version v1 --model main_xgboost
```

## Data integrity audit

Generate dataset integrity artifacts (canonical key alignment, step counts, missingness chart, golden sample):

```powershell
docker compose exec api python -m app.reporting.dataset_integrity --version v1
```

Outputs are written under `reports/<timestamp>` by default:
- `dataset_integrity.md` (row counts + key/label/null summaries per step)
- `missingness_train_val.png` (per-feature missing values histogram)
- `golden_sample.csv` (50 sampled rows with key, label, and features)

These audits are also visible in the UI under “Dataset integrity audits”.

### Step-by-step: dataset integrity + label audit + divergence monitor

1. Start services:
   ```powershell
   docker compose up --build
   ```
2. Seed data (if not already seeded):
   ```powershell
   docker compose exec api python -m app.etl.seed --limit 1000 --force
   ```
3. Generate a dataset integrity audit:
   ```powershell
   docker compose exec api python -m app.reporting.dataset_integrity --version v1
   ```
4. Open UI at `http://localhost:5173` and scroll to **Dataset integrity audits**.
5. Train a model to generate label audit + divergence reports:
   ```powershell
   docker compose exec api python -m app.ml.train_main --version v1 --model xgboost
   ```
6. Open UI at `http://localhost:5173` and scroll to **Label audits**.
7. Open UI at `http://localhost:5173` and scroll to **Training divergence monitor**.
8. Inspect label audit artifacts on disk:
   - `reports/label_audit_<run_id>/label_audit.md`
   - `reports/label_audit_<run_id>/class_counts.csv`
   - `reports/label_audit_<run_id>/confusion_matrix.csv`
   - `reports/label_audit_<run_id>/metrics.json`
   - `reports/label_audit_<run_id>/predicted_vs_true.png`
   - `reports/label_audit_<run_id>/confusion_matrix.png`
9. Inspect divergence artifacts on disk:
   - `reports/training_divergence_<run_id>/training_divergence.md`
   - `reports/training_divergence_<run_id>/feature_scaling.csv`
   - `reports/training_divergence_<run_id>/step_metrics.csv`
   - `reports/training_divergence_<run_id>/divergence_summary.csv`
   - `reports/training_divergence_<run_id>/loss_lr.png`
   - `reports/training_divergence_<run_id>/grad_norm.png` (if available)



```powershell
docker compose exec api python -m app.reporting.label_audit --run-id <run_id>
```



```powershell
docker compose exec api python -m app.reporting.training_divergence --run-id <run_id>
```

Label audits write an index file for the UI list:
- `reports/label_audits_index.json`

Label audits are filterable by league/season in the UI (uses `/label-audits?league=<code>&season=<label>`).

Training divergence reports write an index file for the UI list:
- `reports/training_divergence_index.json`

Training divergence reports are filterable by league/season in the UI (uses `/training-divergence?league=<code>&season=<label>`).

## Test / evaluation

Training already logs validation/test metrics into `model_runs`. You can also generate a report from stored predictions:

```powershell
docker compose exec api python -m app.reporting.prediction_report
```

## 6.1 Testy modelu predykcyjnego



Artefakty:
- `reports/model_eval/<timestamp>/metrics.json`
- `reports/model_eval/<timestamp>/metrics.csv`
- `reports/model_eval/<timestamp>/confusion_matrix.png`
- `reports/model_eval/<timestamp>/calibration.png` (jeśli wdrożone)
- `reports/model_eval/<timestamp>/subsets.csv`



Uruchomienie (CLI/CI):

```powershell
docker compose exec api python -m app.reporting.model_eval --version v1 --model main_xgboost --baseline baseline_classifier
```

Z filtrowaniem ligi/sezonu:

```powershell
docker compose exec api python -m app.reporting.model_eval --version v1 --model main_xgboost --baseline baseline_classifier --league en.1 --season 2010-11
```

Z konkretną wersją modelu:

```powershell
docker compose exec api python -m app.reporting.model_eval --version v1 --model main_xgboost --model-version 20260207_011921_nogit --baseline baseline_classifier --baseline-version 20260207_011558_nogit
```

- Domyślnie `--lookup best` (najlepszy model z `models/best_models.json`, jeśli istnieje).
- `--lookup latest` wybiera najnowszy artefakt w `python-server/models`.
- `--model-version` / `--baseline-version` wymuszają konkretny artefakt (dokładna reprodukcja).

Przykłady:

Najnowsze artefakty (latest):
```powershell
docker compose exec -T api python -m app.reporting.model_eval --version v1 --model main_xgboost --baseline baseline_classifier --lookup latest
```

Najlepsze z registry (best):
```powershell
docker compose exec -T api python -m app.reporting.model_eval --version v1 --model main_xgboost --baseline baseline_classifier --lookup best
```

Konkretny model główny + konkretna wersja baseline:
```powershell
docker compose exec -T api python -m app.reporting.model_eval --version v1 --model main_xgboost --model-version 20260207_011921_nogit --baseline baseline_classifier --baseline-version 20260207_011558_nogit
```


```powershell
Get-ChildItem python-server/models -Filter "main_xgboost_*.json" | Select-Object Name
Get-ChildItem python-server/models -Filter "baseline_classifier_*.json" | Select-Object Name
```

Wiele treningów per liga/sezon:
- `--lookup best` jest zależny od `--league` i `--season` (najlepszy model dla tego segmentu).
- `--lookup latest` ignoruje ligę/sezon (bierze najnowszy artefakt globalnie).
- Dla pełnej kontroli użyj `--model-version` i `--baseline-version`.

Przykład (best dla konkretnej ligi/sezonu):
```powershell
docker compose exec -T api python -m app.reporting.model_eval --version v1 --model main_xgboost --baseline baseline_classifier --lookup best --league en.1 --season 2010-11
```

Trening na wszystkich sezonach wybranej ligi:
```powershell
docker compose exec -T api python -m app.ml.train_main --version v1 --model xgboost --league en.1
docker compose exec -T api python -m app.ml.train_baseline --version v1 --league en.1
```

Ewaluacja na tej samej lidze (wszystkie sezony):
```powershell
docker compose exec -T api python -m app.reporting.model_eval --version v1 --model main_xgboost --baseline baseline_classifier --league en.1
```

Różnica między baseline i main:
- `app.ml.train_baseline` = model bazowy (logistyczna regresja).
- `app.ml.train_main` = model główny (XGBoost/CatBoost).

Przykład (ta sama liga, wszystkie sezony):
```powershell
docker compose exec -T api python -m app.ml.train_baseline --version v1 --league en.1
docker compose exec -T api python -m app.ml.train_main --version v1 --model xgboost --league en.1
```

Jak rozpoznać ligę/sezon modelu:
- Nazwa pliku mówi o typie modelu (np. `main_xgboost_*`, `baseline_classifier_*`), ale **nie** o lidze.
- Liga/sezon są zapisane w metadanych modelu (`*.json`) i w `model_runs`/`best_models.json`.

Przykład odczytu metadanych:
```powershell
Get-Content python-server/models/main_xgboost_20260207_011921_nogit.json
```

## 6.2 Testy integracyjne aplikacji


```powershell
docker compose exec api python -m unittest discover -s tests -p "test_*.py"
```


## 6.3 Testy funkcjonalne interfejsu (UI)



Uruchomienie (komponenty):
```powershell
cd react-server
npm run test:unit
```

Uruchomienie (Playwright):
```powershell
cd react-server
npx playwright install
npm run test:e2e
```

Wymagane: działający API na `http://localhost:8000` oraz UI na `http://localhost:5173`.

## 6.4 Testy scenariuszy użytkownika (E2E)



Uruchomienie:
```powershell
cd react-server
npx playwright install
npm run test:e2e
```



## 6.5 Analiza stabilności działania



Sprawdzenie healthchecków:
```powershell
docker compose ps
curl http://localhost:8000/health
```

Deterministyczność predykcji (ten sam `match_id` + `model_version`):
```powershell
curl -X POST "http://localhost:8000/predict?feature_version=v1&model_name=main_xgboost&model_version=20260207_011921_nogit" ^
  -H "Content-Type: application/json" ^
  -d "{\"match_id\": 123}"
```

Powtórne wywołanie (powinien zwrócić ten sam rekord z DB):
```powershell
curl -X POST "http://localhost:8000/predict?feature_version=v1&model_name=main_xgboost&model_version=20260207_011921_nogit" ^
  -H "Content-Type: application/json" ^
  -d "{\"match_id\": 123}"
```

Podgląd historii predykcji:
```powershell
curl "http://localhost:8000/predictions?match_id=123&model_name=main_xgboost"
```

## 6.6 Testy wydajnościowe


Narzędzie:
- k6 (standard)

Pliki:
- `scripts/perf/k6.js`
- `scripts/perf/db_indexes.sql`
- `scripts/perf/db_explain.sql`

Przygotowanie:
1. Start usług:
   ```powershell
   docker compose up -d --build
   ```
2. Seed danych:
   ```powershell
   docker compose exec -T api python -m app.etl.seed --limit 1000 --force
   ```
3. Trening modelu:
   ```powershell
   docker compose exec -T api python -m app.ml.train_main --version v1 --model xgboost
   ```
4. Opcjonalnie wyczyść predykcje, aby wymusić zapis w DB:
   ```powershell
   docker compose exec -T db psql -U app -d football -c "TRUNCATE TABLE predictions;"
   ```

Uruchomienie k6 (lokalnie):
```powershell
k6 run scripts/perf/k6.js
```

Uruchomienie k6 (Docker):
```powershell
docker run --rm -i -v ${PWD}:/work -w /work grafana/k6 run scripts/perf/k6.js -e BASE_URL=http://host.docker.internal:8000
```

Uruchomienie k6 (Docker Compose):
```powershell
docker compose --profile perf run --rm k6
```

Zmienne środowiskowe (opcjonalne):
- `BASE_URL` (domyślnie `http://localhost:8000`)
- `LEAGUE` (np. `en.1`)
- `SEASON` (np. `2010-11`)
- `FEATURE_VERSION` (domyślnie `v1`)
- `MODEL_NAME` (domyślnie `main_xgboost`)
- `MATCH_LIMIT` (domyślnie `200`)
- `PREDICT_TIMEOUT` (domyślnie `20s`)

Sterowanie obciążeniem (opcjonalne):
- `LIST_MATCHES_VUS` (domyślnie `3`)
- `LIST_PREDICTIONS_VUS` (domyślnie `2`)
- `LIST_DURATION` (domyślnie `30s`)
- `PREDICT_STEADY_RATE` (domyślnie `2`/s)
- `PREDICT_STEADY_DURATION` (domyślnie `60s`)
- `PREDICT_STEADY_VUS` (domyślnie `10`)
- `PREDICT_STEADY_MAX_VUS` (domyślnie `30`)
- `PREDICT_BURST` (domyślnie `false`, ustaw `true` aby włączyć burst)
- `PREDICT_BURST_RATE` (domyślnie `8`/s)
- `PREDICT_BURST_DURATION` (domyślnie `15s`)
- `PREDICT_BURST_VUS` (domyślnie `15`)
- `PREDICT_BURST_MAX_VUS` (domyślnie `40`)

Progi (opcjonalne):
- `HTTP_REQ_FAILED_THRESHOLD` (domyślnie `rate<0.05`)
- `API_FAILURE_THRESHOLD` (domyślnie `rate<0.05`)
- `SERVER_ERROR_THRESHOLD` (domyślnie `rate<0.01`)
- `MATCHES_P95_THRESHOLD` (domyślnie `p(95)<1500`)
- `PREDICTIONS_P95_THRESHOLD` (domyślnie `p(95)<1500`)
- `PREDICT_P95_THRESHOLD` (domyślnie `p(95)<6000`)

Eksport wyników:
```powershell
k6 run scripts/perf/k6.js --summary-export reports/k6_summary.json
```

Indeksy DB (opcjonalnie, dla istniejącej bazy):
```powershell
Get-Content scripts/perf/db_indexes.sql | docker compose exec -T db psql -U app -d football
```

EXPLAIN (opcjonalnie):
```powershell
Get-Content scripts/perf/db_explain.sql | docker compose exec -T db psql -U app -d football
```



Indeksy DB zastosowane:
- `ix_seasons_league_label`, `ix_matches_date`, `ix_predictions_model_created`.

EXPLAIN before/after (skrócony):
- matches list: Before = Seq Scan `seasons` + Index Scan `matches` + Sort; ~10.251 ms. After = ten sam plan, ale ~0.176 ms (prawdopodobnie cache).
- predictions list: Before = Index Scan `ix_predictions_model_name` + Sort; ~0.029 ms. After = Index Scan `ix_predictions_model_created` bez sortu; ~0.013 ms.

Uwaga:
- p95 ~30s oznacza timeouty pod obciążeniem — zbyt agresywne scenariusze saturują API/DB pool.

## Quick run (end-to-end)

1. Start services:
   ```powershell
   docker compose up -d --build
   ```
2. Seed data:
   ```powershell
   docker compose exec -T api python -m app.etl.seed --limit 1000 --force
   ```
3. Train models:
   ```powershell
   docker compose exec -T api python -m app.ml.train_main --version v1 --model xgboost
   docker compose exec -T api python -m app.ml.train_baseline --version v1
   ```
4. Run 6.1 model evaluation harness:
   ```powershell
   docker compose exec -T api python -m app.reporting.model_eval --version v1 --model main_xgboost --baseline baseline_classifier
   ```
5. Run integration tests:
   ```powershell
   docker compose exec -T api python -m unittest discover -s tests -p "test_*.py"
   ```

Single command (pipeline):

```powershell
.\pipeline.ps1
```

## API endpoints

- `GET /health`
- `GET /leagues`
- `GET /seasons?league=en.1`
- `GET /matches?league=en.1&season=2010-11&limit=200`
- `GET /model-runs?league=en.1&season=2010-11&feature_version=v1`
- `GET /model-runs/top?league=en.1&season=2010-11&feature_version=v1&metric=test_log_loss&limit=5`
- `GET /dataset-integrity`
- `GET /label-audits?league=en.1&season=2010-11`
- `GET /training-divergence?league=en.1&season=2010-11`
- `POST /predict?feature_version=v1&model_name=main_xgboost&model_version=<optional>`
- `GET /predictions?match_id=123`

## How to use run tracking

Example flow:
1. Seed data (`app.etl.seed`).
2. Train a model (`app.ml.train_main` or `app.ml.train_baseline`).
3. Fetch runs or top models via API.

Useful metrics for sorting:
- `test_log_loss` (lower is better)
- `test_brier_score` (lower is better)
- `test_f1_macro` (higher is better)
- `test_accuracy` (higher is better)
