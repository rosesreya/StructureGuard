import numpy as np
import pandas as pd

from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)

FOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vibration"
    / "blocked_folds.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "vibration"
    / "stage2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_COLUMNS = [
    "Accel_X (m/s^2)",
    "Accel_Y (m/s^2)",
    "Accel_Z (m/s^2)",
    "Strain (με)",
    "Temp (°C)"
]


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)
fold_df = pd.read_csv(FOLD_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# Feature statistics by chronological fold
# --------------------------------------------------

feature_results = []
class_results = []

for _, fold in fold_df.iterrows():

    fold_number = int(fold["fold"])

    start = int(fold["val_start"])
    end = int(fold["val_end"])

    partition = df.iloc[start:end + 1].copy()

    print("\n" + "=" * 60)
    print(f"FOLD {fold_number}")
    print("=" * 60)

    print("Rows:", start, "to", end)

    # Class proportions
    counts = (
        partition["Condition Label"]
        .value_counts()
        .reindex([0, 1, 2], fill_value=0)
    )

    proportions = counts / counts.sum()

    print("\nClass distribution:")

    for cls in [0, 1, 2]:

        print(
            f"Class {cls}: "
            f"{counts[cls]} "
            f"({proportions[cls]:.2%})"
        )

        class_results.append({
            "Fold": fold_number,
            "Class": cls,
            "Count": counts[cls],
            "Proportion": proportions[cls]
        })

    # Sensor statistics
    print("\nSensor statistics:")

    for feature in SENSOR_COLUMNS:

        values = partition[feature].dropna()

        mean = values.mean()
        std = values.std()
        median = values.median()
        minimum = values.min()
        maximum = values.max()

        print(
            f"{feature}: "
            f"Mean={mean:.4f}, "
            f"Std={std:.4f}, "
            f"Median={median:.4f}"
        )

        feature_results.append({
            "Fold": fold_number,
            "Feature": feature,
            "Mean": mean,
            "Std": std,
            "Median": median,
            "Minimum": minimum,
            "Maximum": maximum
        })


# --------------------------------------------------
# Save results
# --------------------------------------------------

feature_df = pd.DataFrame(feature_results)
class_df = pd.DataFrame(class_results)

feature_df.to_csv(
    OUTPUT_DIR / "temporal_feature_statistics.csv",
    index=False
)

class_df.to_csv(
    OUTPUT_DIR / "temporal_class_distribution.csv",
    index=False
)


# --------------------------------------------------
# Print comparison tables
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE STATISTICS ACROSS FOLDS")
print("=" * 60)

print(
    feature_df.pivot(
        index="Feature",
        columns="Fold",
        values="Mean"
    ).round(4).to_string()
)

print("\n" + "=" * 60)
print("CLASS PROPORTIONS ACROSS FOLDS")
print("=" * 60)

print(
    class_df.pivot(
        index="Class",
        columns="Fold",
        values="Proportion"
    ).round(4).to_string()
)

print("\nResults saved to:", OUTPUT_DIR)