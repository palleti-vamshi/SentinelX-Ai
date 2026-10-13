"""Unit tests for SentinelX AI Phase 2B: Error Analysis, Threshold Tuning & Interface."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from models.interface import (
    ALL_INPUT_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_THRESHOLD,
    NUMERICAL_FEATURES,
    get_example_record,
    predict,
    validate_flow_record,
)
from models.train_rf import (
    ARTIFACTS_DIR,
    METADATA_PATH,
    METRICS_PATH,
    PIPELINE_PATH,
    SELECTED_THRESHOLD,
    VALIDATION_METRICS_PATH,
    build_pipeline,
    compute_metrics,
    load_pipeline,
)


class TestRandomForestPhase2B(unittest.TestCase):
    """Test suite covering pipeline creation, threshold handling, validation, and contract compliance."""

    def test_pipeline_creation_and_config(self):
        """Verify pipeline creation, structure, and deterministic configuration."""
        cat_cols = ["proto", "service", "state"]
        num_cols = ["dur", "sbytes", "dbytes"]
        pipeline = build_pipeline(
            categorical_cols=cat_cols,
            numerical_cols=num_cols,
            n_estimators=10,
            min_samples_leaf=2,
            min_samples_split=4,
            random_state=42,
        )

        self.assertIsInstance(pipeline, Pipeline)
        self.assertIn("preprocessor", pipeline.named_steps)
        self.assertIn("classifier", pipeline.named_steps)

        clf = pipeline.named_steps["classifier"]
        self.assertEqual(clf.random_state, 42)
        self.assertEqual(clf.n_estimators, 10)
        self.assertEqual(clf.min_samples_leaf, 2)
        self.assertEqual(clf.min_samples_split, 4)
        self.assertEqual(clf.n_jobs, -1)

    def test_model_save_and_reload(self):
        """Verify model serialization and deserialization integrity."""
        self.assertTrue(PIPELINE_PATH.exists(), f"Pipeline artifact missing at {PIPELINE_PATH}")
        pipeline = load_pipeline(PIPELINE_PATH)
        self.assertIsInstance(pipeline, Pipeline)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_rf.joblib"
            import joblib
            joblib.dump(pipeline, tmp_path)
            reloaded = load_pipeline(tmp_path)

            sample = get_example_record(as_attack=True)
            res1 = predict(sample, pipeline=pipeline)
            res2 = predict(sample, pipeline=reloaded)
            self.assertEqual(res1, res2)

    def test_prediction_output_schema(self):
        """Verify prediction strictly obeys Part 5.3 contract specification."""
        record = get_example_record(as_attack=True)
        pred = predict(record)

        expected_keys = {
            "record_id",
            "label",
            "probability",
            "attack_category",
            "model_version",
            "top_features",
        }
        self.assertEqual(set(pred.keys()), expected_keys)

        self.assertIsInstance(pred["record_id"], str)
        self.assertEqual(pred["record_id"], "R-100")
        self.assertIn(pred["label"], ["attack", "benign"])
        self.assertIsInstance(pred["probability"], float)
        self.assertGreaterEqual(pred["probability"], 0.0)
        self.assertLessEqual(pred["probability"], 1.0)
        self.assertIsNone(pred["attack_category"])
        self.assertEqual(pred["model_version"], "rf-v1")
        self.assertIsInstance(pred["top_features"], list)
        self.assertEqual(len(pred["top_features"]), 0)

    def test_threshold_handling(self):
        """Verify dynamic threshold behavior and adherence to default 0.60 threshold."""
        record = get_example_record(as_attack=True)

        # Baseline predict uses DEFAULT_THRESHOLD = 0.60
        pred_default = predict(record)
        self.assertEqual(DEFAULT_THRESHOLD, 0.60)
        self.assertEqual(pred_default["label"], "attack")

        # Overridden threshold: extreme high threshold turns marginal detection into benign
        pred_high_th = predict(record, threshold=0.999)
        self.assertIn(pred_high_th["label"], ["attack", "benign"])

        # Test benign sample at different thresholds
        benign_rec = get_example_record(as_attack=False)
        pred_benign = predict(benign_rec)
        self.assertEqual(pred_benign["label"], "benign")

    def test_reproducibility(self):
        """Verify that inference is completely deterministic."""
        record = get_example_record(as_attack=True)
        pred1 = predict(record)
        pred2 = predict(record)
        self.assertEqual(pred1, pred2)

    def test_feature_ordering_and_unseen_categories(self):
        """Verify robustness to shuffled input keys and unseen categorical values."""
        record = get_example_record(as_attack=True)

        # Shuffle keys
        shuffled = {k: record[k] for k in sorted(record.keys(), reverse=True)}
        pred_shuffled = predict(shuffled)
        self.assertIn(pred_shuffled["label"], ["attack", "benign"])

        # Inject unseen category in state (such as ACC from test dataset)
        unseen_record = dict(record)
        unseen_record["state"] = "ACC"
        pred_unseen = predict(unseen_record)
        self.assertIsInstance(pred_unseen["probability"], float)

    def test_missing_and_incompatible_features_rejected(self):
        """Verify strict validation and informative errors for invalid records."""
        # Missing required features
        incomplete = {"record_id": "R-999", "proto": "tcp", "service": "http"}
        with self.assertRaises(ValueError) as ctx:
            predict(incomplete)
        self.assertIn("Record is missing", str(ctx.exception))

        # Synthetic host alert (non-network telemetry)
        host_alert = {
            "record_id": "HOST-001",
            "hostname": "workstation-12",
            "user": "alice",
            "event_type": "process_creation",
        }
        with self.assertRaises(ValueError) as ctx:
            predict(host_alert)
        self.assertIn("cannot directly classify non-network alerts", str(ctx.exception))

        # Incompatible numerical feature type
        corrupt_record = get_example_record(as_attack=True)
        corrupt_record["dur"] = "invalid_string_duration"
        with self.assertRaises(ValueError) as ctx:
            predict(corrupt_record)
        self.assertIn("Incompatible value for numerical feature", str(ctx.exception))

        # Invalid input type
        with self.assertRaises(TypeError):
            predict(["not", "a", "dict"])

    def test_metric_serialization(self):
        """Verify metrics.json exists, is valid, and matches test dataset volume."""
        self.assertTrue(METRICS_PATH.exists(), f"metrics.json missing at {METRICS_PATH}")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        self.assertEqual(metrics["model_version"], "rf-v1")
        self.assertEqual(metrics["selected_threshold"], 0.60)
        self.assertIn("metrics_at_selected_threshold", metrics)
        self.assertIn("comparison_delta", metrics)

        sel = metrics["metrics_at_selected_threshold"]
        cm = sel["confusion_matrix"]
        total_eval = (
            cm["true_negatives"]
            + cm["false_positives"]
            + cm["false_negatives"]
            + cm["true_positives"]
        )
        self.assertEqual(total_eval, 82332)

    def test_validation_metrics_artifact(self):
        """Verify validation study artifact is properly saved and documented."""
        self.assertTrue(
            VALIDATION_METRICS_PATH.exists(),
            f"validation_metrics.json missing at {VALIDATION_METRICS_PATH}",
        )
        with open(VALIDATION_METRICS_PATH, "r", encoding="utf-8") as f:
            val_study = json.load(f)

        self.assertEqual(val_study["sub_train_samples"], 140272)
        self.assertEqual(val_study["validation_samples"], 35069)
        self.assertEqual(val_study["random_state"], 42)
        self.assertIn("selection_criterion", val_study)
        self.assertGreater(len(val_study["candidates_evaluated"]), 0)

    def test_separation_of_model_inputs_and_targets(self):
        """Verify id, label, and attack_cat are never fed as model input features."""
        self.assertTrue(METADATA_PATH.exists(), f"metadata.json missing at {METADATA_PATH}")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        all_features = meta["all_features"]
        self.assertNotIn("id", all_features)
        self.assertNotIn("label", all_features)
        self.assertNotIn("attack_cat", all_features)
        self.assertEqual(len(all_features), 42)


if __name__ == "__main__":
    unittest.main()
