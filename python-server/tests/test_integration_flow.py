import json
import unittest

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import MODEL_DIR
from app.db.models import Prediction
from app.db.session import SessionLocal
from app.main import app
from app.ml.dataset import FEATURE_COLUMNS, load_features_for_prediction
from app.ml.registry import load_latest_model_by_name


class TestIntegrationFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.feature_version = "v1"

        cls.main_model, cls.main_meta = load_latest_model_by_name(
            MODEL_DIR, "main_xgboost"
        )
        cls.baseline_model, cls.baseline_meta = load_latest_model_by_name(
            MODEL_DIR, "baseline_classifier"
        )

        if cls.main_model is None or cls.main_meta is None:
            raise AssertionError("Missing main_xgboost model artifacts.")
        if cls.baseline_model is None or cls.baseline_meta is None:
            raise AssertionError("Missing baseline_classifier model artifacts.")
        with SessionLocal() as session:
            match_id = session.execute(
                text(
                    "SELECT m.id FROM matches m "
                    "JOIN features f ON f.match_id = m.id AND f.version = :version "
                    "WHERE m.status = 'played' AND m.result IS NOT NULL "
                    "ORDER BY m.match_date DESC NULLS LAST "
                    "LIMIT 1"
                ),
                {"version": cls.feature_version},
            ).scalar()
            if match_id is None:
                raise AssertionError("No match with features found; seed data first.")
            cls.match_id = int(match_id)

    def test_api_health_and_lists(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("status"), "ok")

        leagues = self.client.get("/leagues").json()
        self.assertIsInstance(leagues, list)
        self.assertGreater(len(leagues), 0, "No leagues returned; seed data first.")

        league_code = leagues[0]["code"]
        seasons = self.client.get(f"/seasons?league={league_code}").json()
        self.assertIsInstance(seasons, list)
        self.assertGreater(len(seasons), 0, "No seasons returned for seeded league.")

        matches = self.client.get(f"/matches?league={league_code}&limit=5").json()
        self.assertIsInstance(matches, list)
        self.assertGreater(len(matches), 0, "No matches returned for seeded league.")

    def test_features_exist_for_played_matches(self):
        with SessionLocal() as session:
            total_played = session.execute(
                text(
                    "SELECT COUNT(*) FROM matches "
                    "WHERE status = 'played' AND result IS NOT NULL"
                )
            ).scalar()
            self.assertGreater(
                int(total_played or 0), 0, "No played matches found; seed data first."
            )

            missing = session.execute(
                text(
                    "SELECT COUNT(*) FROM matches m "
                    "LEFT JOIN features f "
                    "ON f.match_id = m.id AND f.version = :version "
                    "WHERE m.status = 'played' AND m.result IS NOT NULL "
                    "AND f.match_id IS NULL"
                ),
                {"version": self.feature_version},
            ).scalar()
            self.assertEqual(
                int(missing or 0),
                0,
                "Found matches without feature rows; ETL integrity issue.",
            )

    def test_model_schema_matches_feature_columns(self):
        for meta in [self.main_meta, self.baseline_meta]:
            schema_name = meta.get("feature_schema")
            self.assertIsNotNone(schema_name, "Model meta missing feature_schema.")
            schema_path = MODEL_DIR / schema_name
            self.assertTrue(schema_path.exists(), f"Schema file missing: {schema_path}")
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            self.assertEqual(
                schema.get("columns"),
                FEATURE_COLUMNS,
                "Feature schema does not match expected columns.",
            )

    def test_model_predict_proba_shape(self):
        with SessionLocal() as session:
            df = load_features_for_prediction(
                session, self.feature_version, only_played=True, include_match_id=True
            )
        self.assertFalse(df.empty, "No feature rows available for prediction.")
        X = df[FEATURE_COLUMNS].head(5)
        for model in [self.main_model, self.baseline_model]:
            probs = np.asarray(model.predict_proba(X))
            self.assertEqual(probs.shape[1], 3, "Expected 3-class probabilities.")
            self.assertEqual(probs.shape[0], len(X))
            self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
            sums = probs.sum(axis=1)
            self.assertTrue(np.allclose(sums, 1.0, atol=1e-3))

    def test_predictions_written(self):
        resp = self.client.post(
            f"/predict?feature_version={self.feature_version}&model_name=main_xgboost",
            json={"match_id": self.match_id},
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload.get("match_id"), self.match_id)
        self.assertEqual(payload.get("model_name"), "main_xgboost")
        model_version = payload.get("model_version")
        self.assertIsNotNone(model_version)
        with SessionLocal() as session:
            count = (
                session.query(Prediction)
                .filter(
                    Prediction.model_name == "main_xgboost",
                    Prediction.model_version == model_version,
                )
                .count()
            )
            self.assertGreater(count, 0, "No predictions stored in DB.")

            sample = (
                session.query(Prediction)
                .filter(
                    Prediction.model_name == "main_xgboost",
                    Prediction.model_version == model_version,
                )
                .first()
            )
            self.assertIsNotNone(sample)
            self.assertIsNotNone(sample.created_at)
            self.assertIsNotNone(sample.win_prob_home)
            self.assertIsNotNone(sample.win_prob_draw)
            self.assertIsNotNone(sample.win_prob_away)

    def test_end_to_end_prediction_history(self):
        resp = self.client.post(
            f"/predict?feature_version={self.feature_version}&model_name=main_xgboost",
            json={"match_id": self.match_id},
        )
        self.assertEqual(resp.status_code, 200)
        history = self.client.get(f"/predictions?match_id={self.match_id}").json()
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0, "No prediction history found.")


if __name__ == "__main__":
    unittest.main()
