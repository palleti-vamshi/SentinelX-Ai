"""Unit tests for SentinelX AI Phase 3: SHAP Feature Explainability."""

import unittest
import numpy as np
import pandas as pd

from models.explain import (
    explain,
    extract_attack_shap_values,
    get_tree_explainer,
    map_feature_metadata,
)
from models.interface import (
    ALL_INPUT_FEATURES,
    get_example_record,
    predict,
)


class TestSHAPExplainability(unittest.TestCase):
    """Test suite for local SHAP explanations, additivity checks, and contract integrity."""

    def test_explain_attack_record(self):
        """Verify SHAP explanation structure and values for an attack record."""
        rec = get_example_record(as_attack=True)
        exp = explain(rec, top_k=8)

        # Check top-level contract keys
        required_keys = {
            "record_id",
            "model_version",
            "label",
            "predicted_label",
            "probability",
            "predicted_probability",
            "base_value",
            "decision_threshold",
            "attack_category",
            "top_features",
            "summary",
            "additivity_check_passed",
        }
        self.assertTrue(required_keys.issubset(set(exp.keys())))

        self.assertEqual(exp["record_id"], "R-100")
        self.assertEqual(exp["predicted_label"], "attack")
        self.assertEqual(exp["label"], "attack")
        self.assertGreaterEqual(exp["predicted_probability"], 0.60)
        self.assertEqual(exp["decision_threshold"], 0.60)
        self.assertIsNone(exp["attack_category"])
        self.assertTrue(exp["additivity_check_passed"])
        self.assertIn("R-100 is classified as ATTACK", exp["summary"])

        # Check top features
        top_feats = exp["top_features"]
        self.assertEqual(len(top_feats), 8)

        # Check sorting by absolute SHAP value
        abs_shap_vals = [abs(f["shap_value"]) for f in top_feats]
        self.assertEqual(abs_shap_vals, sorted(abs_shap_vals, reverse=True))

        # Check item schema
        for f in top_feats:
            self.assertIn("feature", f)
            self.assertIn("original_feature", f)
            self.assertIn("value", f)
            self.assertIn("shap_value", f)
            self.assertIn("contribution", f)
            self.assertIn("direction", f)
            self.assertIn("impact", f)
            self.assertIn("explanation", f)
            self.assertIn(f["direction"], ["attack", "benign"])

    def test_explain_benign_record(self):
        """Verify SHAP explanation for a benign record shows mitigating normal indicators."""
        rec = get_example_record(as_attack=False)
        exp = explain(rec, top_k=6)

        self.assertEqual(exp["record_id"], "R-200")
        self.assertEqual(exp["predicted_label"], "benign")
        self.assertEqual(exp["label"], "benign")
        self.assertLess(exp["predicted_probability"], 0.60)
        self.assertTrue(exp["additivity_check_passed"])
        self.assertIn("R-200 is classified as BENIGN", exp["summary"])

        # Top feature should push toward benign (negative SHAP value)
        top_f = exp["top_features"][0]
        self.assertEqual(top_f["direction"], "benign")
        self.assertEqual(top_f["impact"], "pushed_toward_benign")
        self.assertLess(top_f["shap_value"], 0.0)

    def test_additivity_check(self):
        """Verify mathematical additivity of TreeExplainer (base_value + sum(shap) == prob)."""
        attack_rec = get_example_record(as_attack=True)
        exp_attack = explain(attack_rec)
        self.assertTrue(exp_attack["additivity_check_passed"])

        benign_rec = get_example_record(as_attack=False)
        exp_benign = explain(benign_rec)
        self.assertTrue(exp_benign["additivity_check_passed"])

    def test_consistency_with_predict_interface(self):
        """Verify that explain() matches predict() in predicted probability and label."""
        for as_att in [True, False]:
            rec = get_example_record(as_attack=as_att)
            pred = predict(rec)
            exp = explain(rec)

            self.assertEqual(pred["label"], exp["predicted_label"])
            self.assertEqual(pred["label"], exp["label"])
            self.assertAlmostEqual(pred["probability"], exp["predicted_probability"], places=3)

    def test_predict_include_explanations_flag(self):
        """Verify predict(..., include_explanations=True) embeds SHAP top_features."""
        rec = get_example_record(as_attack=True)

        # Default: fast inference, empty top_features
        pred_fast = predict(rec, include_explanations=False)
        self.assertEqual(pred_fast["top_features"], [])

        # With explanations enabled
        pred_explained = predict(rec, include_explanations=True)
        self.assertGreater(len(pred_explained["top_features"]), 0)
        self.assertEqual(pred_explained["label"], pred_fast["label"])

    def test_feature_name_alignment(self):
        """Verify that original_feature accurately maps to one of ALL_INPUT_FEATURES."""
        rec = get_example_record(as_attack=True)
        exp = explain(rec, top_k=10)

        for f in exp["top_features"]:
            self.assertIn(f["original_feature"], ALL_INPUT_FEATURES)
            self.assertIn(f["original_feature"], rec)

    def test_missing_and_invalid_inputs(self):
        """Verify explain() rejects incomplete records or non-network alerts."""
        # Missing features
        with self.assertRaises(ValueError):
            explain({"record_id": "R-1", "proto": "tcp"})

        # Host-only alert
        with self.assertRaises(ValueError):
            explain({"hostname": "server01", "event_id": 4624})

        # Non-mapping input
        with self.assertRaises(TypeError):
            explain(["not", "a", "record"])

    def test_top_k_parameter(self):
        """Verify top_k parameter controls the number of returned feature attributions."""
        rec = get_example_record(as_attack=True)
        exp_4 = explain(rec, top_k=4)
        self.assertEqual(len(exp_4["top_features"]), 4)

        exp_12 = explain(rec, top_k=12)
        self.assertEqual(len(exp_12["top_features"]), 12)


if __name__ == "__main__":
    unittest.main()
