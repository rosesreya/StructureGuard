from pathlib import Path

import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = Path("data/VIBRATIONAL/building_health_monitoring_dataset.csv")

OUTPUT_DIR = Path("results/vibration/stage2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLUMN = "Condition Label"

FEATURES = [
    "Accel_X (m/s^2)",
    "Accel_Y (m/s^2)",
    "Accel_Z (m/s^2)",
    "Strain (με)",
    "Temp (°C)",
]

# Same chronological validation folds used previously.
FOLDS = {
    "Fold 1": (200, 399),
    "Fold 2": (400, 599),
    "Fold 3": (600, 799),
    "Fold 4": (800, 999),
}


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

results = []

# --------------------------------------------------
# Calculate statistics for each class in each fold
# --------------------------------------------------

for fold_name, (start, end) in FOLDS.items():

    fold_df = df.iloc[start:end + 1].copy()

    print("\n" + "=" * 65)
    print(f"{fold_name}: Rows {start} to {end}")
    print("=" * 65)

    for class_label in sorted(df[LABEL_COLUMN].dropna().unique()):

        class_df = fold_df[
            fold_df[LABEL_COLUMN] == class_label
        ]

        print(f"\nClass {class_label} — {len(class_df)} observations")

        if class_df.empty:
            print("No observations for this class.")
            continue

        stats = class_df[FEATURES].agg(
            ["count", "mean", "std", "median"]
        )

        print(stats.round(4))

        for feature in FEATURES:

            results.append({
                "Fold": fold_name,
                "Class": class_label,
                "Feature": feature,
                "Count": class_df[feature].count(),
                "Mean": class_df[feature].mean(),
                "Std": class_df[feature].std(),
                "Median": class_df[feature].median(),
            })


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_file = (
    OUTPUT_DIR / "class_conditional_sensor_statistics.csv"
)

results_df.to_csv(output_file, index=False)

print("\n" + "=" * 65)
print("DIAGNOSTIC COMPLETED")
print("=" * 65)
print("Results saved to:", output_file)