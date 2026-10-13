"""Unit tests for UNSW-NB15 dataset preparation module."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.detection.prepare_data import (
    identify_feature_types,
    validate_datasets,
    build_preprocessor,
    prepare_data,
    load_dataset,
)


class TestDataPreparation(unittest.TestCase):
    """Test suite for data loading, validation, and preprocessing pipeline."""

    def test_identify_feature_types(self):
        sample_df = pd.DataFrame({
            "id": [1, 2],
            "dur": [0.1, 0.5],
            "proto": ["tcp", "udp"],
            "service": ["http", "-"],
            "state": ["FIN", "INT"],
            "sbytes": [100, 200],
            "attack_cat": ["Normal", "Exploits"],
            "label": [0, 1],
        })
        num_cols, cat_cols = identify_feature_types(sample_df)
        self.assertEqual(sorted(cat_cols), ["proto", "service", "state"])
        self.assertEqual(sorted(num_cols), ["dur", "sbytes"])

    def test_validate_datasets_mock(self):
        train = pd.DataFrame({
            "id": [1],
            "dur": [0.1],
            "proto": ["tcp"],
            "service": ["http"],
            "state": ["FIN"],
            "attack_cat": ["Normal"],
            "label": [0],
        })
        test = pd.DataFrame({
            "id": [2],
            "dur": [0.2],
            "proto": ["tcp"],
            "service": ["http"],
            "state": ["ACC"],  # unseen category
            "attack_cat": ["Exploits"],
            "label": [1],
        })
        res = validate_datasets(train, test)
        self.assertTrue(res["columns_match"])
        self.assertTrue(res["labels_valid"])
        self.assertEqual(res["train_missing_total"], 0)
        self.assertEqual(res["test_missing_total"], 0)
        self.assertIn("state", res["unseen_categories_in_test"])
        self.assertEqual(res["unseen_categories_in_test"]["state"], ["ACC"])

    def test_preprocessor_fitted_on_train_only(self):
        # Ensure unseen categories in test don't crash and don't leak into train
        train = pd.DataFrame({
            "proto": ["tcp", "udp"],
            "service": ["http", "dns"],
            "state": ["FIN", "INT"],
            "dur": [0.1, 0.2],
            "label": [0, 1],
            "attack_cat": ["Normal", "Generic"],
        })
        test = pd.DataFrame({
            "proto": ["tcp", "udp"],
            "service": ["http", "dns"],
            "state": ["ACC", "CLO"],  # unseen in train
            "dur": [0.3, 0.4],
            "label": [1, 1],
            "attack_cat": ["Exploits", "Exploits"],
        })

        prep = prepare_data(train, test)
        self.assertEqual(prep["X_train"].shape[0], 2)
        self.assertEqual(prep["X_test"].shape[0], 2)
        self.assertEqual(prep["X_train"].shape[1], prep["X_test"].shape[1])
        # Verify that attack_cat and label are not in feature names
        for col in ["id", "label", "attack_cat"]:
            self.assertNotIn(col, prep["feature_names"])


if __name__ == "__main__":
    unittest.main()
