import unittest

from app.ml.labels import LABELS, predict_class_ids, validate_labels


class TestLabelEncoding(unittest.TestCase):
    def test_predict_class_ids_from_logits(self):
        logits = [
            [0.1, 0.2, 0.3],
            [3.0, 1.0, -1.0],
            [-5.0, 2.5, 2.4],
        ]
        preds = predict_class_ids(logits)
        self.assertEqual(preds, [2, 0, 1])
        validate_labels(preds)


if __name__ == "__main__":
    unittest.main()
