from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

for filename in [
    "UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv",
]:
    path = DATA_DIR / filename

    print(f"\n{'=' * 60}")
    print(f"Dataset: {filename}")
    print("=" * 60)

    if not path.exists():
        print(f"File not found: {path}")
        continue

    df = pd.read_csv(path, low_memory=False)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string()
          if missing.any() else "No missing values")

    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    for target in ["label", "attack_cat"]:
        if target in df.columns:
            print(f"\n{target} distribution:")
            print(df[target].value_counts(dropna=False).to_string())
