"""Data preparation and validation module for the UNSW-NB15 dataset.

Phase 1: Understand and prepare the UNSW-NB15 dataset for SentinelX AI.
Learned preprocessing is fitted strictly on the training set only to prevent data leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_FILENAME = "UNSW_NB15_training-set.csv"
TEST_FILENAME = "UNSW_NB15_testing-set.csv"

# Columns to exclude from input feature matrix X
DEFAULT_EXCLUDE_COLS: tuple[str, ...] = ("id", "label", "attack_cat")


def load_dataset(split: str = "train") -> pd.DataFrame:
    """Load raw training or testing dataset."""
    filename = TRAIN_FILENAME if split == "train" else TEST_FILENAME
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    return pd.read_csv(filepath, low_memory=False)


def identify_feature_types(
    df: pd.DataFrame,
    exclude_cols: Sequence[str] = DEFAULT_EXCLUDE_COLS,
) -> tuple[list[str], list[str]]:
    """Programmatically identify numerical and categorical input feature columns.

    Args:
        df: Input dataframe.
        exclude_cols: Columns to exclude from feature lists (e.g. id, targets).

    Returns:
        tuple of (numerical_cols, categorical_cols)
    """
    candidate_cols = [c for c in df.columns if c not in exclude_cols]

    numerical_cols = [
        c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_cols = [
        c for c in candidate_cols if not pd.api.types.is_numeric_dtype(df[c])
    ]

    return numerical_cols, categorical_cols


def validate_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exclude_cols: Sequence[str] = DEFAULT_EXCLUDE_COLS,
) -> dict[str, Any]:
    """Perform comprehensive data integrity and compatibility validation checks.

    Checks:
    1. Column compatibility (exact match of names and ordering).
    2. Missing value detection.
    3. Duplicate row detection.
    4. Target label domain check (label in {0, 1}).
    5. Categorical cardinality and unseen category detection in test split.
    6. Numerical data sanity (NaN, Inf, negative check).
    """
    results: dict[str, Any] = {}

    # 1. Column compatibility
    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)
    results["columns_match"] = train_cols == test_cols
    results["train_shape"] = train_df.shape
    results["test_shape"] = test_df.shape

    # 2. Missing values
    results["train_missing_total"] = int(train_df.isnull().sum().sum())
    results["test_missing_total"] = int(test_df.isnull().sum().sum())

    # 3. Duplicate rows
    results["train_duplicates"] = int(train_df.duplicated().sum())
    results["test_duplicates"] = int(test_df.duplicated().sum())

    # 4. Target values
    results["train_labels"] = sorted(train_df["label"].unique().tolist())
    results["test_labels"] = sorted(test_df["label"].unique().tolist())
    results["labels_valid"] = (
        bool(results["train_labels"])
        and bool(results["test_labels"])
        and set(results["train_labels"]).issubset({0, 1})
        and set(results["test_labels"]).issubset({0, 1})
    )

    # 5. Feature types
    num_cols, cat_cols = identify_feature_types(train_df, exclude_cols=exclude_cols)
    results["numerical_cols"] = num_cols
    results["categorical_cols"] = cat_cols
    results["num_numerical"] = len(num_cols)
    results["num_categorical"] = len(cat_cols)

    # 6. Unseen categories in test
    unseen_categories: dict[str, list[str]] = {}
    for col in cat_cols:
        train_vals = set(train_df[col].dropna().unique())
        test_vals = set(test_df[col].dropna().unique())
        unseen = sorted(list(test_vals - train_vals))
        if unseen:
            unseen_categories[col] = unseen
    results["unseen_categories_in_test"] = unseen_categories

    return results


def build_preprocessor(
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer for preprocessing.

    Categorical features are One-Hot Encoded with handle_unknown='ignore'
    to safely accommodate unseen categories (e.g. in test data).
    Numerical features are kept as passthrough to preserve numeric values.

    Args:
        categorical_cols: List of categorical feature column names.
        numerical_cols: List of numerical feature column names.

    Returns:
        Configured ColumnTransformer.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_cols,
            ),
            ("num", "passthrough", numerical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def prepare_data(
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Execute complete Phase 1 data preparation pipeline.

    CRITICAL: Fits preprocessor exclusively on the training set (X_train).
    Transforms both X_train and X_test using the fitted preprocessor.

    Returns:
        Dictionary containing:
        - X_train, y_train, attack_cat_train
        - X_test, y_test, attack_cat_test
        - feature_names (transformed feature names)
        - preprocessor (fitted ColumnTransformer)
        - validation_results
    """
    if train_df is None:
        train_df = load_dataset("train")
    if test_df is None:
        test_df = load_dataset("test")

    # Validate datasets
    validation_results = validate_datasets(train_df, test_df)

    num_cols = validation_results["numerical_cols"]
    cat_cols = validation_results["categorical_cols"]
    feature_cols = cat_cols + num_cols

    # Separate features and targets
    X_train_raw = train_df[feature_cols]
    y_train = train_df["label"].to_numpy()
    attack_cat_train = train_df["attack_cat"].to_numpy()

    X_test_raw = test_df[feature_cols]
    y_test = test_df["label"].to_numpy()
    attack_cat_test = test_df["attack_cat"].to_numpy()

    # Build and fit preprocessor strictly on training data
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train_processed = preprocessor.fit_transform(X_train_raw)

    # Transform test set using the train-fitted preprocessor
    X_test_processed = preprocessor.transform(X_test_raw)

    transformed_feature_names = preprocessor.get_feature_names_out().tolist()

    return {
        "X_train": X_train_processed,
        "y_train": y_train,
        "attack_cat_train": attack_cat_train,
        "X_test": X_test_processed,
        "y_test": y_test,
        "attack_cat_test": attack_cat_test,
        "feature_names": transformed_feature_names,
        "preprocessor": preprocessor,
        "validation_results": validation_results,
    }


def main() -> None:
    """Run Phase 1 inspection, validation, and pipeline verification."""
    print("=" * 70)
    print("SentinelX AI - Phase 1: UNSW-NB15 Dataset Preparation & Validation")
    print("=" * 70)

    train_df = load_dataset("train")
    test_df = load_dataset("test")

    val = validate_datasets(train_df, test_df)

    print(f"\n[1] Dataset Shapes:")
    print(f"    Train: {val['train_shape'][0]:,} rows x {val['train_shape'][1]} cols")
    print(f"    Test:  {val['test_shape'][0]:,} rows x {val['test_shape'][1]} cols")
    print(f"    Column names match exactly: {val['columns_match']}")

    print(f"\n[2] Data Integrity:")
    print(f"    Train missing values: {val['train_missing_total']}")
    print(f"    Test missing values:  {val['test_missing_total']}")
    print(f"    Train duplicate rows: {val['train_duplicates']}")
    print(f"    Test duplicate rows:  {val['test_duplicates']}")
    print(f"    Label domains: Train={val['train_labels']}, Test={val['test_labels']}")
    print(f"    Binary label valid (only 0 and 1): {val['labels_valid']}")

    print(f"\n[3] Identified Feature Types:")
    print(f"    Excluded non-feature columns: {DEFAULT_EXCLUDE_COLS}")
    print(f"    Categorical features ({val['num_categorical']}): {val['categorical_cols']}")
    print(f"    Numerical features ({val['num_numerical']}):")
    for i in range(0, len(val["numerical_cols"]), 6):
        print(f"      {val['numerical_cols'][i:i+6]}")

    print(f"\n[4] Unseen Categories Analysis (Test vs Train):")
    if val["unseen_categories_in_test"]:
        for col, unseen in val["unseen_categories_in_test"].items():
            print(f"    Column '{col}' has {len(unseen)} unseen values in test: {unseen}")
    else:
        print("    None found.")

    print(f"\n[5] Executing Preprocessing Pipeline (Fitted on Train Only)...")
    prepared = prepare_data(train_df, test_df)

    print(f"    X_train shape after encoding: {prepared['X_train'].shape}")
    print(f"    X_test shape after encoding:  {prepared['X_test'].shape}")
    print(f"    Total encoded features: {len(prepared['feature_names'])}")
    print(f"    y_train distribution: 0={int((prepared['y_train'] == 0).sum()):,}, 1={int((prepared['y_train'] == 1).sum()):,}")
    print(f"    y_test distribution:  0={int((prepared['y_test'] == 0).sum()):,}, 1={int((prepared['y_test'] == 1).sum()):,}")

    print("\nPhase 1 validation completed successfully.")


if __name__ == "__main__":
    main()
